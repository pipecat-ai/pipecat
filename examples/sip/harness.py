#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Provision the accounts and trunks the SIP test harness calls through.

Tier 0 is Daily only: two SIP clients on your Daily domain, one for the bot
and one for a softphone, so a call never leaves Daily's SIP service. Tier 1
adds a Twilio Elastic SIP trunk for real phones.

Provision Tier 0, then read back what a softphone needs::

    uv run python examples/sip/harness.py provision
    uv run python examples/sip/harness.py status

Tier 1 adds the trunk, authorizes this machine's public IP, and routes a
number you already own to the trunk::

    uv run python examples/sip/harness.py provision --tier 1 --number +15551234567

Have Twilio read a sentence to a running bot, or write the body the bot dials
out with::

    uv run python examples/sip/harness.py call dial-in --via sip --duration 30
    uv run python examples/sip/harness.py call dial-out --to softphone

``--dry-run`` prints the REST calls a command would make and touches nothing.

Credentials live in ``.env`` next to this file; provisioned resources are
cached in ``.state.json`` beside it. Both are gitignored — the cache holds
SIP passwords, which Daily returns only once and can otherwise only be
rotated. ``status`` always re-reads the live resources, so deleting the cache
loses passwords but never the ability to recover: the next ``provision``
rotates them.
"""

import argparse
import asyncio
import json
import os
import secrets
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp
from dotenv import load_dotenv
from twilio_api import TwilioAPI, TwilioError, get_public_ip

from pipecat.transports.daily.utils import (
    DailyRESTHelper,
    DailySIPClientObject,
    DailySIPClientParams,
)

load_dotenv(Path(__file__).parent / ".env", override=True)

HERE = Path(__file__).parent
STATE_PATH = HERE / ".state.json"

DAILY_API_URL = "https://api.daily.co/v1"

# Every resource the harness owns carries this prefix, so status can find them
# live and teardown never touches anything else.
NAME_PREFIX = "pipecat-harness"
BOT_USER = f"{NAME_PREFIX}-bot"
DEV_USER = f"{NAME_PREFIX}-dev"

# Creation order. Teardown walks it backwards.
TIER0_USERS = (BOT_USER, DEV_USER)

# Tier 1 resource names. Twilio scopes FriendlyName per account, so the prefix
# alone is enough to find a resource again and safe to reuse across runs.
TRUNK_NAME = NAME_PREFIX
ORIGINATION_NAME = f"{NAME_PREFIX}-origination"
ACL_NAME = NAME_PREFIX
ACL_ENTRY_NAME = f"{NAME_PREFIX}-dev-ip"
CRED_LIST_NAME = NAME_PREFIX
CRED_USER = NAME_PREFIX

TWILIO_API = "https://api.twilio.com/2010-04-01/Accounts/<sid>"
TWILIO_TRUNKING = "https://trunking.twilio.com/v1"


class HarnessError(Exception):
    """A problem the developer can fix — printed without a traceback."""


# ---------------------------------------------------------------- local state


def read_state() -> dict[str, Any]:
    """Read the resource cache.

    Returns:
        Cached state, or an empty dict when no cache exists.
    """
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text())
    except json.JSONDecodeError as e:
        raise HarnessError(f"{STATE_PATH} is not valid JSON ({e}). Delete it and re-provision.")


def write_state(state: dict[str, Any]) -> None:
    """Write the resource cache, readable only by its owner.

    Args:
        state: State to cache.
    """
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    STATE_PATH.chmod(0o600)


def cached_clients(state: dict[str, Any]) -> dict[str, Any]:
    """The cached SIP clients, creating the nesting on the way in.

    Args:
        state: State to read.

    Returns:
        Mutable mapping of username to cached client fields.
    """
    return state.setdefault("daily", {}).setdefault("sip_clients", {})


def cache_client(state: dict[str, Any], client: DailySIPClientObject) -> None:
    """Merge a live SIP client into the cache, keeping a known password.

    Args:
        state: State to update.
        client: Live client as Daily returned it.
    """
    entry = cached_clients(state).setdefault(client.username, {})
    entry.update(
        username=client.username,
        domain=client.domain,
        sip_uri=client.sip_uri,
    )
    if client.password:
        entry["password"] = client.password


def twilio_state(state: dict[str, Any]) -> dict[str, Any]:
    """The cached Twilio resources, creating the nesting on the way in.

    Args:
        state: State to read.

    Returns:
        Mutable mapping of resource name to cached fields.
    """
    return state.setdefault("twilio", {})


def gen_password() -> str:
    """Generate a password Twilio and Daily both accept.

    Twilio SIP credentials need 12+ characters with a digit and both letter
    cases; Daily needs 12-64 with a letter and a digit.

    Returns:
        Fresh 20-character password.
    """
    alphabet = "abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    while True:
        candidate = "".join(secrets.choice(alphabet) for _ in range(20))
        if any(c.isdigit() for c in candidate) and any(c.isupper() for c in candidate):
            if any(c.islower() for c in candidate):
                return candidate


def sip_scheme(uri: str) -> str:
    """Add the ``sip:`` scheme Twilio requires, if it is missing.

    Args:
        uri: SIP URI, with or without a scheme.

    Returns:
        URI with a ``sip:`` scheme.
    """
    return uri if uri.startswith("sip:") else f"sip:{uri}"


def bot_sip_target(state: dict[str, Any]) -> str:
    """The bot's SIP URI in the ``sip:`` form Twilio origination wants.

    Args:
        state: State to read.

    Returns:
        Bot's dialable URI with a ``sip:`` scheme.
    """
    uri = cached_clients(state).get(BOT_USER, {}).get("sip_uri")
    if not uri:
        raise HarnessError(
            f"{BOT_USER} is not provisioned, so the trunk has nowhere to send calls. "
            "Run: uv run python examples/sip/harness.py provision"
        )
    return sip_scheme(uri)


def origination_sip_url(state: dict[str, Any]) -> str:
    """Where the trunk sends inbound calls.

    Args:
        state: State to read.

    Returns:
        Bot's SIP URI with the TLS transport parameter Daily's SIP service
        expects from Twilio.
    """
    return f"{bot_sip_target(state)};transport=tls"


# ------------------------------------------------------------------ dry runs


def plan(method: str, path: str, payload: dict | None = None, when: str = "") -> None:
    """Print one REST call a command would make.

    Args:
        method: HTTP method.
        path: Full request URL.
        payload: Request body, when there is one.
        when: Condition under which the call happens.
    """
    line = f"  {method:6} {path}"
    if payload is not None:
        line += f"  {json.dumps(payload, sort_keys=True)}"
    if when:
        line += f"   [{when}]"
    print(line)


def plan_provision_tier0() -> None:
    """Print the Daily REST calls Tier 0 provisioning would make."""
    print("Tier 0 — Daily SIP clients:")
    plan("GET", f"{DAILY_API_URL}/sip-clients", {"limit": 100}, when="to find existing clients")
    for username in TIER0_USERS:
        plan("POST", f"{DAILY_API_URL}/sip-clients", {"username": username}, when="when missing")
        plan(
            "POST",
            f"{DAILY_API_URL}/sip-clients/{username}/rotate",
            when="when it exists but no password is cached",
        )


def plan_provision_tier1(args: argparse.Namespace) -> None:
    """Print the Twilio REST calls Tier 1 provisioning would make.

    Args:
        args: Parsed arguments, for the number flags.
    """
    print()
    print("Tier 1 — Twilio Elastic SIP trunk:")
    plan("GET", "https://api.ipify.org", when="to authorize this machine's public IP")
    plan("GET", f"{TWILIO_TRUNKING}/Trunks", when=f"to find the {TRUNK_NAME} trunk")
    plan(
        "POST",
        f"{TWILIO_TRUNKING}/Trunks",
        {
            "DomainName": f"{NAME_PREFIX}-<8 hex>.pstn.twilio.com",
            "FriendlyName": TRUNK_NAME,
            "Secure": "false",
        },
        when="when missing; a new suffix is drawn if the domain collides",
    )
    plan(
        "POST",
        f"{TWILIO_TRUNKING}/Trunks/<trunk>",
        {"Secure": "false"},
        when="when Secure drifted on",
    )
    plan("GET", f"{TWILIO_TRUNKING}/Trunks/<trunk>/OriginationUrls")
    plan(
        "POST",
        f"{TWILIO_TRUNKING}/Trunks/<trunk>/OriginationUrls[/<url>]",
        {
            "Enabled": "true",
            "FriendlyName": ORIGINATION_NAME,
            "Priority": "10",
            "SipUrl": "sip:<bot sip uri>;transport=tls",
            "Weight": "10",
        },
        when="create when missing, update when it no longer matches the bot",
    )
    plan("GET", f"{TWILIO_API}/SIP/IpAccessControlLists.json")
    plan(
        "POST",
        f"{TWILIO_API}/SIP/IpAccessControlLists.json",
        {"FriendlyName": ACL_NAME},
        when="when missing",
    )
    plan("GET", f"{TWILIO_API}/SIP/IpAccessControlLists/<acl>/IpAddresses.json")
    plan(
        "POST",
        f"{TWILIO_API}/SIP/IpAccessControlLists/<acl>/IpAddresses[/<entry>].json",
        {"CidrPrefixLength": "32", "FriendlyName": ACL_ENTRY_NAME, "IpAddress": "<public ip>"},
        when="create when missing, update when this machine's IP moved",
    )
    plan("GET", f"{TWILIO_API}/SIP/CredentialLists.json")
    plan(
        "POST",
        f"{TWILIO_API}/SIP/CredentialLists.json",
        {"FriendlyName": CRED_LIST_NAME},
        when="when missing",
    )
    plan("GET", f"{TWILIO_API}/SIP/CredentialLists/<list>/Credentials.json")
    plan(
        "POST",
        f"{TWILIO_API}/SIP/CredentialLists/<list>/Credentials[/<cred>].json",
        {"Password": "<generated>", "Username": CRED_USER},
        when="create when missing, reset when no password is cached",
    )
    plan("GET", f"{TWILIO_TRUNKING}/Trunks/<trunk>/IpAccessControlLists")
    plan(
        "POST",
        f"{TWILIO_TRUNKING}/Trunks/<trunk>/IpAccessControlLists",
        {"IpAccessControlListSid": "<acl>"},
        when="when not attached",
    )
    plan("GET", f"{TWILIO_TRUNKING}/Trunks/<trunk>/CredentialLists")
    plan(
        "POST",
        f"{TWILIO_TRUNKING}/Trunks/<trunk>/CredentialLists",
        {"CredentialListSid": "<list>"},
        when="when not attached",
    )
    if args.buy_number:
        plan(
            "GET",
            f"{TWILIO_API}/AvailablePhoneNumbers/US/Local.json",
            {"AreaCode": args.area_code or "<any>", "PageSize": "5", "VoiceEnabled": "true"},
        )
        plan(
            "POST",
            f"{TWILIO_API}/IncomingPhoneNumbers.json",
            {"FriendlyName": NAME_PREFIX, "PhoneNumber": "<chosen>"},
            when="COSTS MONEY — only after you confirm",
        )
    if args.number or args.buy_number:
        plan(
            "GET",
            f"{TWILIO_API}/IncomingPhoneNumbers.json",
            {"PhoneNumber": args.number or "<bought>"},
        )
        plan("GET", f"{TWILIO_TRUNKING}/Trunks/<trunk>/PhoneNumbers")
        plan(
            "POST",
            f"{TWILIO_TRUNKING}/Trunks/<trunk>/PhoneNumbers",
            {"PhoneNumberSid": "<number>"},
            when="when not already routed to the trunk",
        )
    else:
        print("  (no number flag: dial-in from the PSTN stays unavailable)")


# ------------------------------------------------------------------ Daily API


def make_helper(session: aiohttp.ClientSession) -> DailyRESTHelper:
    """Build a Daily REST helper from ``DAILY_API_KEY``.

    Args:
        session: Session the helper issues requests on.

    Returns:
        Helper bound to the Daily API.
    """
    key = os.getenv("DAILY_API_KEY")
    if not key:
        raise HarnessError(
            "DAILY_API_KEY is not set. Copy examples/sip/env.example to "
            "examples/sip/.env and fill it in."
        )
    return DailyRESTHelper(daily_api_key=key, daily_api_url=DAILY_API_URL, aiohttp_session=session)


async def live_harness_clients(helper: DailyRESTHelper) -> dict[str, DailySIPClientObject]:
    """Every SIP client on the domain whose name the harness owns.

    Args:
        helper: Daily REST helper.

    Returns:
        Mapping of username to live client (passwords never included).
    """
    found: dict[str, DailySIPClientObject] = {}
    offset = 0
    while True:
        page = await helper.list_sip_clients(limit=100, offset=offset)
        for client in page.sip_clients:
            if client.username.startswith(NAME_PREFIX):
                found[client.username] = client
        offset += len(page.sip_clients)
        if not page.sip_clients or offset >= page.total:
            return found


# ------------------------------------------------------------------ provision


async def provision_tier0(helper: DailyRESTHelper, state: dict[str, Any]) -> None:
    """Bring the two Tier 0 SIP clients up to date.

    Args:
        helper: Daily REST helper.
        state: State to update in place.
    """
    live = await live_harness_clients(helper)
    for username in TIER0_USERS:
        cached = cached_clients(state).get(username, {})
        if username not in live:
            client = await helper.create_sip_client(DailySIPClientParams(username=username))
            print(f"[new]  SIP client {username} created at {client.sip_uri}")
        elif not cached.get("password"):
            client = await helper.rotate_sip_client_password(username)
            print(f"[new]  SIP client {username} password rotated (none was cached)")
        else:
            client = live[username]
            print(f"[keep] SIP client {username} unchanged at {client.sip_uri}")
        cache_client(state, client)
        write_state(state)


async def ensure_trunk(api: TwilioAPI, state: dict[str, Any]) -> dict[str, Any]:
    """Find or create the harness trunk, with Symmetric RTP on and TLS off.

    Args:
        api: Twilio client.
        state: State to update in place.

    Returns:
        Cached trunk entry.
    """
    tw = twilio_state(state)
    trunks = await api.list_trunks()
    trunk = next((t for t in trunks if t.get("friendly_name") == TRUNK_NAME), None)

    if trunk is None:
        # Termination domains are globally unique, so a random suffix is drawn
        # until Twilio accepts one.
        for attempt in range(5):
            suffix = secrets.token_hex(4)
            domain = f"{NAME_PREFIX}-{suffix}.pstn.twilio.com"
            try:
                trunk = await api.create_trunk(TRUNK_NAME, domain)
            except TwilioError as e:
                if attempt == 4 or "21244" not in str(e) and " 409 " not in str(e):
                    raise
                print(f"[retry] trunk domain {domain} taken, drawing another suffix")
                continue
            tw["domain_suffix"] = suffix
            print(f"[new]  trunk {TRUNK_NAME} created at {domain}")
            break
    else:
        fixes = {}
        if trunk.get("secure"):
            fixes["Secure"] = False
        if fixes:
            trunk = await api.update_trunk(trunk["sid"], **fixes)
            print(f"[set]  trunk {TRUNK_NAME} updated: {', '.join(sorted(fixes))}")
        else:
            print(f"[keep] trunk {TRUNK_NAME} unchanged at {trunk.get('domain_name')}")

    entry = tw.setdefault("trunk", {})
    entry.update(sid=trunk["sid"], domain_name=trunk.get("domain_name"))
    write_state(state)
    return entry


async def ensure_origination(api: TwilioAPI, state: dict[str, Any]) -> None:
    """Point the trunk's inbound calls at the bot's Daily SIP client.

    Args:
        api: Twilio client.
        state: State to update in place.
    """
    tw = twilio_state(state)
    trunk_sid = tw["trunk"]["sid"]
    sip_url = origination_sip_url(state)
    urls = await api.list_origination_urls(trunk_sid)
    mine = next((u for u in urls if u.get("friendly_name") == ORIGINATION_NAME), None)

    if mine is None:
        mine = await api.create_origination_url(
            trunk_sid, friendly_name=ORIGINATION_NAME, sip_url=sip_url
        )
        print(f"[new]  origination -> {sip_url}")
    elif mine.get("sip_url") != sip_url or not mine.get("enabled"):
        mine = await api.update_origination_url(
            trunk_sid, mine["sid"], SipUrl=sip_url, Enabled=True, Priority=10, Weight=10
        )
        print(f"[set]  origination repointed -> {sip_url}")
    else:
        print(f"[keep] origination unchanged -> {sip_url}")

    tw["origination"] = {"sid": mine["sid"], "sip_url": sip_url}
    write_state(state)


async def ensure_acl(api: TwilioAPI, state: dict[str, Any]) -> None:
    """Authorize this machine's current public IP on the trunk.

    Twilio checks the source address of every INVITE the trunk terminates, and
    a developer's address moves, so the entry is refreshed on every run.

    Args:
        api: Twilio client.
        state: State to update in place.
    """
    tw = twilio_state(state)
    trunk_sid = tw["trunk"]["sid"]
    public_ip = await get_public_ip()

    acls = await api.list_ip_acls()
    acl = next((a for a in acls if a.get("friendly_name") == ACL_NAME), None)
    if acl is None:
        acl = await api.create_ip_acl(ACL_NAME)
        print(f"[new]  IP ACL {ACL_NAME} created")
    acl_sid = acl["sid"]

    entries = await api.list_ip_acl_addresses(acl_sid)
    entry = next((e for e in entries if e.get("friendly_name") == ACL_ENTRY_NAME), None)
    if entry is None:
        await api.create_ip_acl_address(acl_sid, friendly_name=ACL_ENTRY_NAME, ip_address=public_ip)
        print(f"[new]  IP ACL authorizes {public_ip}")
    elif entry.get("ip_address") != public_ip:
        await api.update_ip_acl_address(acl_sid, entry["sid"], IpAddress=public_ip)
        print(f"[set]  IP ACL moved {entry.get('ip_address')} -> {public_ip}")
    else:
        print(f"[keep] IP ACL already authorizes {public_ip}")

    attached = await api.list_trunk_ip_acls(trunk_sid)
    if not any(a.get("sid") == acl_sid for a in attached):
        await api.attach_ip_acl(trunk_sid, acl_sid)
        print(f"[new]  IP ACL {ACL_NAME} attached to the trunk")

    tw["acl"] = {"sid": acl_sid, "ip_address": public_ip}
    write_state(state)


async def ensure_credentials(api: TwilioAPI, state: dict[str, Any]) -> None:
    """Keep a SIP credential the bot registers to the trunk with.

    Args:
        api: Twilio client.
        state: State to update in place.
    """
    tw = twilio_state(state)
    trunk_sid = tw["trunk"]["sid"]
    cached = tw.get("credential", {})

    lists = await api.list_credential_lists()
    cred_list = next((c for c in lists if c.get("friendly_name") == CRED_LIST_NAME), None)
    if cred_list is None:
        cred_list = await api.create_credential_list(CRED_LIST_NAME)
        print(f"[new]  credential list {CRED_LIST_NAME} created")
    list_sid = cred_list["sid"]

    creds = await api.list_credentials(list_sid)
    cred = next((c for c in creds if c.get("username") == CRED_USER), None)
    password = cached.get("password")
    if cred is None:
        password = gen_password()
        cred = await api.create_credential(list_sid, CRED_USER, password)
        print(f"[new]  credential {CRED_USER} created")
    elif not password:
        password = gen_password()
        await api.update_credential(list_sid, cred["sid"], password)
        print(f"[new]  credential {CRED_USER} password reset (none was cached)")
    else:
        print(f"[keep] credential {CRED_USER} unchanged")

    attached = await api.list_trunk_credential_lists(trunk_sid)
    if not any(c.get("sid") == list_sid for c in attached):
        await api.attach_credential_list(trunk_sid, list_sid)
        print(f"[new]  credential list {CRED_LIST_NAME} attached to the trunk")

    tw["credential"] = {
        "list_sid": list_sid,
        "sid": cred["sid"],
        "username": CRED_USER,
        "password": password,
    }
    write_state(state)


async def buy_number(api: TwilioAPI, area_code: str | None, assume_yes: bool) -> dict[str, Any]:
    """Search Twilio's inventory and purchase one number.

    Costs money every month it is held, so the purchase waits on an explicit
    ``y`` unless ``--yes`` was passed.

    Args:
        api: Twilio client.
        area_code: Area code to search within.
        assume_yes: Skip the interactive confirmation.

    Returns:
        Purchased incoming phone number resource.
    """
    available = await api.list_available_local_numbers(area_code=area_code, limit=5)
    if not available:
        raise HarnessError(
            f"Twilio has no local numbers available in area code {area_code or '(any)'}."
        )
    chosen = available[0]["phone_number"]
    print(f"Twilio offers: {', '.join(n['phone_number'] for n in available)}")
    print(f"About to BUY {chosen} — roughly $1.15 per month until released.")
    if not assume_yes:
        answer = await asyncio.to_thread(input, "Type y to buy it: ")
        if answer.strip().lower() != "y":
            raise HarnessError("purchase declined; nothing was bought.")
    number = await api.buy_phone_number(chosen, NAME_PREFIX)
    print(f"[new]  bought {number['phone_number']}")
    return number


async def ensure_number(api: TwilioAPI, state: dict[str, Any], args: argparse.Namespace) -> None:
    """Route a phone number to the trunk, buying one only when asked.

    Args:
        api: Twilio client.
        state: State to update in place.
        args: Parsed arguments, for ``--number``, ``--buy-number``, ``--yes``.
    """
    tw = twilio_state(state)
    trunk_sid = tw["trunk"]["sid"]

    if args.buy_number:
        number = await buy_number(api, args.area_code, args.yes)
    elif args.number:
        owned = await api.list_incoming_phone_numbers(args.number)
        if not owned:
            raise HarnessError(
                f"{args.number} is not on this Twilio account. Buy it in the Twilio "
                "console, or run provision with --buy-number."
            )
        number = owned[0]
    else:
        cached = tw.get("number", {})
        if cached.get("phone_number"):
            print(f"[keep] number {cached['phone_number']} stays routed to the trunk")
        else:
            print("no number attached; dial-in from PSTN unavailable")
            print("  attach one: provision --tier 1 --number +1XXXXXXXXXX")
        return

    attached = await api.list_trunk_phone_numbers(trunk_sid)
    if any(n.get("sid") == number["sid"] for n in attached):
        print(f"[keep] number {number['phone_number']} already routed to the trunk")
    else:
        await api.attach_phone_number(trunk_sid, number["sid"])
        print(f"[new]  number {number['phone_number']} routed to the trunk")

    tw["number"] = {"sid": number["sid"], "phone_number": number["phone_number"]}
    write_state(state)


async def provision_tier1(api: TwilioAPI, state: dict[str, Any], args: argparse.Namespace) -> None:
    """Bring the Twilio trunk and everything it points at up to date.

    Args:
        api: Twilio client.
        state: State to update in place.
        args: Parsed arguments, for the number flags.
    """
    await ensure_trunk(api, state)
    await ensure_origination(api, state)
    await ensure_acl(api, state)
    await ensure_credentials(api, state)
    await ensure_number(api, state, args)


def print_tier1_summary(state: dict[str, Any]) -> None:
    """Print what the trunk routes and what the bot registers with.

    Args:
        state: State to read.
    """
    tw = twilio_state(state)
    if not tw.get("trunk"):
        return
    cred = tw.get("credential", {})
    print()
    print("Twilio trunk:")
    print(f"  Domain:     {tw['trunk'].get('domain_name', '?')}")
    print(f"  Number:     {tw.get('number', {}).get('phone_number') or '(none attached)'}")
    print(f"  Origination:{' ' + tw.get('origination', {}).get('sip_url', '?')}")
    print(f"  Authorized: {tw.get('acl', {}).get('ip_address', '?')} (this machine)")
    print(f"  SIP user:   {cred.get('username', '?')}")
    print(f"  SIP pass:   {cred.get('password') or '(not cached — re-run provision --tier 1)'}")


def print_tier0_credentials(state: dict[str, Any]) -> None:
    """Print what a softphone needs and what to dial.

    Args:
        state: State to read.
    """
    clients = cached_clients(state)
    dev = clients.get(DEV_USER, {})
    bot = clients.get(BOT_USER, {})
    unknown = "(not cached — run provision to rotate it)"
    print()
    print("Softphone account — Telephone.app > Settings > Accounts > +:")
    print(f"  Username:   {dev.get('username', '?')}")
    print(f"  Domain:     {dev.get('domain', '?')}")
    print(f"  Password:   {dev.get('password') or unknown}")
    print("  Transport:  UDP")
    print()
    print("Bot's dialable URI — call this from the softphone:")
    print(f"  {bot.get('sip_uri', '?')}")


async def cmd_provision(args: argparse.Namespace) -> int:
    """Create or refresh the resources the harness calls through.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    if args.dry_run:
        plan_provision_tier0()
        if args.tier == "1":
            plan_provision_tier1(args)
        print()
        print(f"--dry-run: no requests sent, {STATE_PATH.name} not written.")
        return 0

    state = read_state()
    async with aiohttp.ClientSession() as session:
        helper = make_helper(session)
        await provision_tier0(helper, state)
    if args.tier == "1":
        async with TwilioAPI.from_env() as api:
            await provision_tier1(api, state, args)
    print_tier0_credentials(state)
    print_tier1_summary(state)
    return 0


# --------------------------------------------------------------------- status


def note(text: str, detail: str = "") -> None:
    """Print one advisory checklist line, outside the pass/fail tally.

    Args:
        text: What the line reports.
        detail: Follow-up line, indented under it.
    """
    print(f"  [WARNING] {text}")
    if detail:
        print(f"         {detail}")


def row(ok: bool, text: str, fix: str = "") -> bool:
    """Print one checklist row.

    Args:
        ok: Whether the check passed.
        text: What the check looked at.
        fix: What to run when it failed.

    Returns:
        The value of ``ok``, so callers can accumulate it.
    """
    print(f"  [{'OK ' if ok else 'FIX'}] {text}")
    if not ok and fix:
        print(f"         {fix}")
    return ok


async def status_tier1(state: dict[str, Any]) -> bool:
    """Check the live trunk against what PSTN calls need.

    Args:
        state: State to update in place with what the trunk looks like now.

    Returns:
        Whether every check passed.
    """
    print()
    if not os.getenv("TWILIO_ACCOUNT_SID"):
        print("Tier 1 — skipped, TWILIO_ACCOUNT_SID is not set (Tier 0 needs no Twilio).")
        return True

    hint = "run: uv run python examples/sip/harness.py provision --tier 1"
    tw = twilio_state(state)
    passed = True

    async with TwilioAPI.from_env() as api:
        trunks = await api.list_trunks()
        trunk = next((t for t in trunks if t.get("friendly_name") == TRUNK_NAME), None)
        print("Tier 1 — Twilio Elastic SIP trunk:")
        if trunk is None:
            print(f"  (not provisioned — {hint})")
            return True

        tw.setdefault("trunk", {}).update(sid=trunk["sid"], domain_name=trunk.get("domain_name"))
        passed &= row(True, f"trunk {TRUNK_NAME} exists at {trunk.get('domain_name')}")
        if not trunk.get("symmetric_rtp_enabled"):
            note(
                "Symmetric RTP off (Twilio's default; the REST API cannot set it)",
                f"turn on only for one-way audio: Console > Elastic SIP Trunking > "
                f"{TRUNK_NAME} > General > Symmetric RTP",
            )
        passed &= row(not trunk.get("secure"), "Secure off (the harness terminates over UDP)", hint)

        want = cached_clients(state).get(BOT_USER, {}).get("sip_uri")
        want_url = f"{sip_scheme(want)};transport=tls" if want else None
        urls = await api.list_origination_urls(trunk["sid"])
        mine = next((u for u in urls if u.get("friendly_name") == ORIGINATION_NAME), None)
        if mine:
            tw["origination"] = {"sid": mine["sid"], "sip_url": mine.get("sip_url")}
        passed &= row(
            bool(mine) and mine.get("sip_url") == want_url and bool(mine.get("enabled")),
            f"origination points at the bot: {(mine or {}).get('sip_url') or '(missing)'}",
            hint,
        )

        public_ip = await get_public_ip()
        acls = await api.list_ip_acls()
        acl = next((a for a in acls if a.get("friendly_name") == ACL_NAME), None)
        entries = await api.list_ip_acl_addresses(acl["sid"]) if acl else []
        authorized = [e.get("ip_address") for e in entries]
        if acl:
            tw["acl"] = {"sid": acl["sid"], "ip_address": public_ip}
        passed &= row(
            public_ip in authorized,
            f"IP ACL authorizes this machine ({public_ip})",
            f"{hint}  (your address moved; provision refreshes the entry)",
        )
        attached_acls = await api.list_trunk_ip_acls(trunk["sid"]) if acl else []
        passed &= row(
            bool(acl) and any(a.get("sid") == acl["sid"] for a in attached_acls),
            f"IP ACL {ACL_NAME} attached to the trunk",
            hint,
        )

        lists = await api.list_credential_lists()
        cred_list = next((c for c in lists if c.get("friendly_name") == CRED_LIST_NAME), None)
        creds = await api.list_credentials(cred_list["sid"]) if cred_list else []
        cred = next((c for c in creds if c.get("username") == CRED_USER), None)
        if cred_list and cred:
            tw.setdefault("credential", {}).update(
                list_sid=cred_list["sid"], sid=cred["sid"], username=CRED_USER
            )
        passed &= row(bool(cred), f"credential {CRED_USER} exists", hint)
        passed &= row(
            bool(tw.get("credential", {}).get("password")),
            f"credential {CRED_USER} password cached",
            f"{hint}  (resets a password Twilio won't return again)",
        )
        attached_lists = await api.list_trunk_credential_lists(trunk["sid"]) if cred_list else []
        passed &= row(
            bool(cred_list) and any(c.get("sid") == cred_list["sid"] for c in attached_lists),
            f"credential list {CRED_LIST_NAME} attached to the trunk",
            hint,
        )

        numbers = await api.list_trunk_phone_numbers(trunk["sid"])
        if numbers:
            tw["number"] = {
                "sid": numbers[0]["sid"],
                "phone_number": numbers[0].get("phone_number"),
            }
        else:
            tw.pop("number", None)
        passed &= row(
            bool(numbers),
            "number routed to the trunk: "
            + (", ".join(n.get("phone_number", "?") for n in numbers) or "(none)"),
            f"{hint} --number +1XXXXXXXXXX   (buying one costs money — see README.md)",
        )

    return passed


async def cmd_status(args: argparse.Namespace) -> int:
    """Check the live resources against what the harness needs.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code — 0 when every check passed.
    """
    state = read_state()
    provision_hint = "run: uv run python examples/sip/harness.py provision"
    passed = True

    async with aiohttp.ClientSession() as session:
        helper = make_helper(session)
        live = await live_harness_clients(helper)

    print("Tier 0 — Daily SIP clients:")
    for username in TIER0_USERS:
        client = live.get(username)
        passed &= row(
            client is not None,
            f"{username} exists" + (f" at {client.sip_uri}" if client else ""),
            provision_hint,
        )
        if client:
            cache_client(state, client)
            passed &= row(
                bool(cached_clients(state).get(username, {}).get("password")),
                f"{username} password cached",
                f"{provision_hint}  (rotates a password Daily won't return again)",
            )

    passed &= await status_tier1(state)

    write_state(state)
    print_tier0_credentials(state)
    print_tier1_summary(state)
    return 0 if passed else 1


# ------------------------------------------------------------------- teardown


async def teardown_tier0(
    helper: DailyRESTHelper | None, state: dict[str, Any], dry_run: bool
) -> int:
    """Delete the Tier 0 SIP clients, newest first.

    Args:
        helper: Daily REST helper, or ``None`` on a dry run.
        state: State to update in place.
        dry_run: Print the deletions instead of making them.

    Returns:
        Number of resources torn down (or that would be).
    """
    clients = cached_clients(state)
    count = 0
    for username in reversed(TIER0_USERS):
        if username not in clients:
            continue
        count += 1
        if dry_run:
            plan("DELETE", f"{DAILY_API_URL}/sip-clients/{username}")
            continue
        assert helper is not None
        await helper.delete_sip_client(username)
        print(f"[gone] SIP client {username} deleted")
        clients.pop(username, None)
        write_state(state)
    return count


async def teardown_tier1(api: TwilioAPI | None, state: dict[str, Any], args) -> int:
    """Delete the Twilio resources, newest first.

    The number is only detached from the trunk — releasing it is what ends the
    monthly charge, and that needs ``--release-number``.

    Args:
        api: Twilio client, or ``None`` on a dry run.
        state: State to update in place.
        args: Parsed arguments, for ``--dry-run`` and ``--release-number``.

    Returns:
        Number of resources torn down (or that would be).
    """
    tw = twilio_state(state)
    trunk = tw.get("trunk", {})
    trunk_sid = trunk.get("sid")

    if api is None:
        trunk_path = f"{TWILIO_TRUNKING}/Trunks/{trunk_sid or '<trunk>'}"

    if api is not None and not trunk_sid:
        # State can be deleted; the trunk is still findable by its name.
        live = await api.list_trunks()
        found = next((t for t in live if t.get("friendly_name") == TRUNK_NAME), None)
        trunk_sid = found["sid"] if found else None

    count = 0
    number = tw.get("number", {})
    if number.get("sid"):
        count += 1
        if api is None:
            plan("DELETE", f"{trunk_path}/PhoneNumbers/{number['sid']}")
            if args.release_number:
                plan("DELETE", f"{TWILIO_API}/IncomingPhoneNumbers/{number['sid']}.json")
        else:
            if trunk_sid:
                await api.detach_phone_number(trunk_sid, number["sid"])
            print(f"[gone] number {number.get('phone_number')} detached from the trunk")
            if args.release_number:
                await api.release_phone_number(number["sid"])
                print(f"[gone] number {number.get('phone_number')} released to Twilio")
            else:
                print(f"       kept {number.get('phone_number')}; --release-number gives it back")
            tw.pop("number", None)
            write_state(state)

    cred = tw.get("credential", {})
    if cred.get("list_sid"):
        count += 1
        if api is None:
            plan("DELETE", f"{trunk_path}/CredentialLists/{cred['list_sid']}")
            plan("DELETE", f"{TWILIO_API}/SIP/CredentialLists/{cred['list_sid']}.json")
        else:
            if trunk_sid:
                await api.detach_credential_list(trunk_sid, cred["list_sid"])
            await api.delete_credential_list(cred["list_sid"])
            print(f"[gone] credential list {CRED_LIST_NAME} deleted")
            tw.pop("credential", None)
            write_state(state)

    acl = tw.get("acl", {})
    if acl.get("sid"):
        count += 1
        if api is None:
            plan("DELETE", f"{trunk_path}/IpAccessControlLists/{acl['sid']}")
            plan("DELETE", f"{TWILIO_API}/SIP/IpAccessControlLists/{acl['sid']}.json")
        else:
            if trunk_sid:
                await api.detach_ip_acl(trunk_sid, acl["sid"])
            await api.delete_ip_acl(acl["sid"])
            print(f"[gone] IP ACL {ACL_NAME} deleted")
            tw.pop("acl", None)
            write_state(state)

    origination = tw.get("origination", {})
    if origination.get("sid"):
        count += 1
        if api is None:
            plan(
                "DELETE",
                f"{trunk_path}/OriginationUrls/{origination['sid']}",
            )
        else:
            if trunk_sid:
                await api.delete_origination_url(trunk_sid, origination["sid"])
            print("[gone] origination deleted")
            tw.pop("origination", None)
            write_state(state)

    if trunk_sid:
        count += 1
        if api is None:
            plan("DELETE", f"{TWILIO_TRUNKING}/Trunks/{trunk_sid}")
        else:
            await api.delete_trunk(trunk_sid)
            print(f"[gone] trunk {TRUNK_NAME} deleted")
            tw.pop("trunk", None)
            tw.pop("domain_suffix", None)
            write_state(state)

    if not tw:
        state.pop("twilio", None)
        if api is not None:
            write_state(state)
    return count


async def cmd_teardown(args: argparse.Namespace) -> int:
    """Delete what the harness provisioned, in reverse creation order.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    state = read_state()
    tiers = {"0": {"0"}, "1": {"1"}, "all": {"0", "1"}}[args.tier]

    if "0" in tiers and state.get("twilio"):
        print("note: Tier 1 origination points at the bot's SIP client; tear down Tier 1 too.")

    async def run(helper: DailyRESTHelper | None, api: TwilioAPI | None) -> int:
        count = 0
        # Tier 1 first: its origination points at the Tier 0 bot client.
        if "1" in tiers:
            count += await teardown_tier1(api, state, args)
        if "0" in tiers:
            count += await teardown_tier0(helper, state, args.dry_run)
        return count

    if args.dry_run:
        count = await run(None, None)
        print("nothing to tear down." if not count else f"--dry-run: {count} deletions planned.")
        return 0

    async with aiohttp.ClientSession() as session:
        api = TwilioAPI.from_env() if "1" in tiers and state.get("twilio") else None
        try:
            count = await run(make_helper(session), api)
        finally:
            if api is not None:
                await api.aclose()
    print("nothing to tear down." if not count else f"torn down {count} resources.")
    return 0


# ----------------------------------------------------------------------- call

DIALOUT_BODY_PATH = HERE / "dialout-body.json"

# Two seconds of silence for the bot's greeting, one sentence its STT should
# transcribe, then silence to answer into.
TEST_TWIML = (
    "<Response>"
    '<Pause length="2"/>'
    "<Say>This is the pipecat SIP harness test call. Please say something back.</Say>"
    '<Pause length="25"/>'
    "</Response>"
)

# Twilio statuses that mean the call is over.
FINAL_STATUSES = frozenset({"completed", "busy", "failed", "no-answer", "canceled"})

# The milestone the bot terminal reaches as each status arrives.
STATUS_HINTS = {
    "ringing": "watch the bot terminal for [2/6] INVITE — call ringing",
    "in-progress": "then [3/6] media flowing and [4/6] pipeline started",
}

# Dry runs print these where a provisioned value belongs.
PLACEHOLDER_BOT_URI = f"sip:{BOT_USER}@<your-daily-domain>.sip.daily.co"
PLACEHOLDER_DEV_URI = f"sip:{DEV_USER}@<your-daily-domain>.sip.daily.co"
PLACEHOLDER_NUMBER = "+1XXXXXXXXXX"
PLACEHOLDER_DOMAIN = f"{NAME_PREFIX}-<8 hex>.pstn.twilio.com"
PLACEHOLDER_PASSWORD = "<run provision --tier 1>"


def resolve(value: str | None, placeholder: str, missing: str, *, strict: bool) -> str:
    """One provisioned value, or the placeholder a dry run stands in with.

    Args:
        value: Cached value, when the resource is provisioned.
        placeholder: What a dry run prints instead.
        missing: What to tell the developer when the value is needed for real.
        strict: Raise instead of falling back to the placeholder.

    Returns:
        The value, or the placeholder.
    """
    if value:
        return value
    if strict:
        raise HarnessError(missing)
    return placeholder


def trunk_number(state: dict[str, Any]) -> str | None:
    """The number routed to the harness trunk.

    Args:
        state: State to read.

    Returns:
        Number in E.164, or None when none is attached.
    """
    return twilio_state(state).get("number", {}).get("phone_number")


def trunk_domain(state: dict[str, Any]) -> str | None:
    """The trunk's termination domain.

    Args:
        state: State to read.

    Returns:
        Domain name, or None when no trunk is provisioned.
    """
    return twilio_state(state).get("trunk", {}).get("domain_name")


def dialin_target(state: dict[str, Any], via: str, *, strict: bool) -> str:
    """Where the test call rings.

    Args:
        state: State to read.
        via: ``sip`` for the bot's Daily SIP URI, ``pstn`` for the trunk number.
        strict: Raise on a missing value instead of using a placeholder.

    Returns:
        Twilio ``To`` value.
    """
    if via == "pstn":
        return resolve(
            trunk_number(state),
            PLACEHOLDER_NUMBER,
            "no number is routed to the trunk, so the PSTN has nothing to ring. Run: "
            "uv run python examples/sip/harness.py provision --tier 1 --number +1XXXXXXXXXX",
            strict=strict,
        )
    uri = cached_clients(state).get(BOT_USER, {}).get("sip_uri")
    return resolve(
        sip_scheme(uri) if uri else None,
        PLACEHOLDER_BOT_URI,
        f"{BOT_USER} is not provisioned, so there is no SIP URI to dial. Run: "
        "uv run python examples/sip/harness.py provision",
        strict=strict,
    )


def dialin_caller_id(state: dict[str, Any], *, strict: bool) -> str:
    """Caller ID the test call comes from.

    Args:
        state: State to read.
        strict: Raise on a missing value instead of using a placeholder.

    Returns:
        Twilio ``From`` value.
    """
    return resolve(
        trunk_number(state) or os.getenv("TWILIO_CALLER_ID"),
        PLACEHOLDER_NUMBER,
        "no caller ID to call from. Either route a number to the trunk "
        "(provision --tier 1 --number +1XXXXXXXXXX), or set TWILIO_CALLER_ID in "
        "examples/sip/.env to another number this Twilio account owns.",
        strict=strict,
    )


async def place_test_call(api: TwilioAPI, to: str, from_: str, duration: int) -> int:
    """Call the bot, follow the status, and hang up after ``duration``.

    Args:
        api: Twilio client.
        to: Destination to ring.
        from_: Caller ID to call from.
        duration: Seconds to stay on the call.

    Returns:
        Process exit code — 0 when the call was answered and held 5 s.
    """
    call = await api.create_call(to=to, from_=from_, twiml=TEST_TWIML)
    sid = call["sid"]
    status = call.get("status") or "queued"
    chain = [status]
    print(f"[new]  call {sid}: {from_} -> {to}")
    print(f"  {status}")

    started = time.monotonic()
    answered_at: float | None = None
    while time.monotonic() - started < duration:
        await asyncio.sleep(1)
        current = (await api.fetch_call(sid)).get("status") or status
        if current == status:
            continue
        status = current
        chain.append(status)
        hint = STATUS_HINTS.get(status, "")
        print(f"  → {status}" + (f"   {hint}" if hint else ""))
        if status == "in-progress":
            answered_at = time.monotonic()
        if status in FINAL_STATUSES:
            break

    if status not in FINAL_STATUSES:
        print(f"  hanging up after {duration}s")
        await api.complete_call(sid)
        chain.append("completed")
    print(f"  {' → '.join(chain)}")

    if answered_at is None:
        print(f"error: the call never reached in-progress (ended {status}).", file=sys.stderr)
        return 1
    print("  the bot terminal ends with [5/6] call ended and [6/6] quality")
    held = time.monotonic() - answered_at
    if held < 5:
        print(f"error: the call was answered but held only {held:.1f}s.", file=sys.stderr)
        return 1
    print(f"answered and held {held:.0f}s.")
    return 0


async def cmd_call_dialin(args: argparse.Namespace) -> int:
    """Have Twilio call the running bot and read a sentence to it.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    state = read_state()
    to = dialin_target(state, args.via, strict=not args.dry_run)
    from_ = dialin_caller_id(state, strict=not args.dry_run)

    if args.dry_run:
        print(f"Dial-in test call via {args.via}, hanging up after {args.duration}s:")
        plan("POST", f"{TWILIO_API}/Calls.json", {"From": from_, "To": to, "Twiml": TEST_TWIML})
        plan("GET", f"{TWILIO_API}/Calls/<call>.json", when="every second, to follow the status")
        plan(
            "POST",
            f"{TWILIO_API}/Calls/<call>.json",
            {"Status": "completed"},
            when=f"after {args.duration}s, unless the call already ended",
        )
        print()
        print("--dry-run: no requests sent; unprovisioned values print as placeholders.")
        return 0

    if to == from_:
        print(f"warning: Twilio rejects a call whose From and To are both {to}.")
    async with TwilioAPI.from_env() as api:
        return await place_test_call(api, to, from_, args.duration)


def dialout_plan(state: dict[str, Any], to: str, *, strict: bool) -> tuple[dict, str]:
    """The runner body that dials ``to`` and the command that runs the bot with it.

    Args:
        state: State to read.
        to: ``softphone`` or a number in E.164.
        strict: Raise on a missing value instead of using a placeholder.

    Returns:
        Runner body, and the shell command that starts the bot with it.
    """
    body_file = f"examples/sip/{DIALOUT_BODY_PATH.name}"
    run = f"uv run python examples/sip/bot.py -t sip -d --runner-body {body_file}"

    if to == "softphone":
        # Both clients sit on the same Daily domain, so the bot's own
        # registration carries the call.
        uri = cached_clients(state).get(DEV_USER, {}).get("sip_uri")
        sip_uri = resolve(
            sip_scheme(uri) if uri else None,
            PLACEHOLDER_DEV_URI,
            f"{DEV_USER} is not provisioned, so the softphone has no SIP URI. Run: "
            "uv run python examples/sip/harness.py provision",
            strict=strict,
        )
        return {"dialout_settings": {"sipUri": sip_uri}}, run

    domain = resolve(
        trunk_domain(state),
        PLACEHOLDER_DOMAIN,
        "no Twilio trunk to dial through. Run: "
        "uv run python examples/sip/harness.py provision --tier 1",
        strict=strict,
    )
    caller = resolve(
        trunk_number(state) or os.getenv("TWILIO_CALLER_ID"),
        PLACEHOLDER_NUMBER,
        "no caller ID to present to the trunk. Either route a number to it "
        "(provision --tier 1 --number +1XXXXXXXXXX), or set TWILIO_CALLER_ID in "
        "examples/sip/.env.",
        strict=strict,
    )
    cred = twilio_state(state).get("credential", {})
    user = cred.get("username") or CRED_USER
    password = resolve(
        cred.get("password"),
        PLACEHOLDER_PASSWORD,
        "the trunk's SIP password is not cached. Run: "
        "uv run python examples/sip/harness.py provision --tier 1",
        strict=strict,
    )
    body = {
        "dialout_settings": {
            "sipUri": f"sip:{to}@{domain}",
            "headers": {"P-Asserted-Identity": f"<sip:{caller}@sip.twilio.com>"},
        }
    }
    # Twilio termination authenticates each INVITE against the trunk's
    # credential list, so the bot registers nowhere. A registration-less
    # account breaks medianat=stun (STUN lookup runs before any DNS fixes an
    # address family), so STUN is off; the trunk's Symmetric RTP setting
    # carries return media instead — see README.md troubleshooting.
    env = (
        f"SIP_USER={user} SIP_PASS={password} SIP_DOMAIN={domain} "
        f"SIP_REG_INTERVAL=0 SIP_AUTH_USER={user} SIP_STUN_SERVER=off"
    )
    return body, f"{env} {run}"


async def cmd_call_dialout(args: argparse.Namespace) -> int:
    """Write the runner body the bot dials out with, and print how to run it.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    if args.to != "softphone" and not (args.to.startswith("+") and args.to[1:].isdigit()):
        raise HarnessError(f"--to takes 'softphone' or a number in E.164 form, not {args.to!r}.")

    state = read_state()
    body, command = dialout_plan(state, args.to, strict=not args.dry_run)
    text = json.dumps(body, indent=2, sort_keys=True) + "\n"

    print(f"Runner body for dialing {args.to}:")
    print(text, end="")
    if args.dry_run:
        print(f"--dry-run: {DIALOUT_BODY_PATH.name} not written; values may be placeholders.")
    else:
        DIALOUT_BODY_PATH.write_text(text)
        print(f"written to {DIALOUT_BODY_PATH}")
    print()
    print("Start the bot with it — it dials as soon as it is ready:")
    print(f"  {command}")
    return 0


# ----------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        Parser covering every subcommand.
    """
    parser = argparse.ArgumentParser(
        prog="harness.py",
        description="Provision the accounts and trunks the SIP test harness calls through.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    provision = subparsers.add_parser("provision", help="create or refresh harness resources")
    provision.add_argument(
        "--tier",
        choices=("0", "1"),
        default="0",
        help="0 = Daily SIP clients only; 1 = also a Twilio trunk for real phones",
    )
    provision.add_argument(
        "--number",
        metavar="+1XXXXXXXXXX",
        help="route a number this Twilio account already owns to the trunk (tier 1)",
    )
    provision.add_argument(
        "--buy-number",
        action="store_true",
        help="buy a number and route it to the trunk — COSTS MONEY, asks first (tier 1)",
    )
    provision.add_argument(
        "--area-code", help="area code to search in when buying a number (tier 1)"
    )
    provision.add_argument(
        "--yes", action="store_true", help="skip the purchase confirmation for --buy-number"
    )
    provision.add_argument(
        "--dry-run", action="store_true", help="print the REST calls and change nothing"
    )
    provision.set_defaults(func=cmd_provision)

    status = subparsers.add_parser("status", help="check live resources against what's needed")
    status.set_defaults(func=cmd_status)

    teardown = subparsers.add_parser("teardown", help="delete harness resources")
    teardown.add_argument(
        "--tier",
        choices=("0", "1", "all"),
        default="0",
        help="which tier's resources to delete (default 0); 'all' deletes both",
    )
    teardown.add_argument(
        "--release-number",
        action="store_true",
        help="give the trunk's number back to Twilio instead of keeping it (tier 1)",
    )
    teardown.add_argument(
        "--dry-run", action="store_true", help="print the deletions and change nothing"
    )
    teardown.set_defaults(func=cmd_teardown)

    call = subparsers.add_parser("call", help="place a test call")
    directions = call.add_subparsers(dest="direction", required=True)

    dialin = directions.add_parser("dial-in", help="have Twilio call the running bot")
    dialin.add_argument(
        "--via",
        choices=("sip", "pstn"),
        default="sip",
        help="sip = ring the bot's Daily SIP URI (default); pstn = ring the trunk's "
        "number, best-effort — Twilio rejects a call whose From and To match",
    )
    dialin.add_argument(
        "--duration", type=int, default=30, help="seconds to stay on the call (default 30)"
    )
    dialin.add_argument(
        "--dry-run", action="store_true", help="print the request and change nothing"
    )
    dialin.set_defaults(func=cmd_call_dialin)

    dialout = directions.add_parser("dial-out", help="write the runner body the bot calls out with")
    dialout.add_argument(
        "--to",
        required=True,
        metavar="+1XXXXXXXXXX|softphone",
        help=f"a number to reach through the trunk, or 'softphone' for the {DEV_USER} client",
    )
    dialout.add_argument("--dry-run", action="store_true", help="print the body and write nothing")
    dialout.set_defaults(func=cmd_call_dialout)

    return parser


def main() -> int:
    """Parse arguments and run the chosen subcommand.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args()
    try:
        return asyncio.run(args.func(args))
    except HarnessError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
