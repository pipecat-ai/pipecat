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
import sys
from pathlib import Path
from typing import Any

import aiohttp
from dotenv import load_dotenv

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
        print()
        print(f"--dry-run: no requests sent, {STATE_PATH.name} not written.")
        return 0

    state = read_state()
    async with aiohttp.ClientSession() as session:
        helper = make_helper(session)
        await provision_tier0(helper, state)
    print_tier0_credentials(state)
    return 0


# --------------------------------------------------------------------- status


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

    write_state(state)
    print_tier0_credentials(state)
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

    async def run(helper: DailyRESTHelper | None) -> int:
        count = 0
        if "0" in tiers:
            count += await teardown_tier0(helper, state, args.dry_run)
        return count

    if args.dry_run:
        count = await run(None)
        print("nothing to tear down." if not count else f"--dry-run: {count} deletions planned.")
        return 0

    async with aiohttp.ClientSession() as session:
        count = await run(make_helper(session))
    print("nothing to tear down." if not count else f"torn down {count} resources.")
    return 0


# ----------------------------------------------------------------------- call


async def cmd_call(args: argparse.Namespace) -> int:
    """Place a test call through the provisioned resources.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    raise HarnessError(f"call {args.direction} needs a provisioned Twilio trunk.")


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
        choices=("0",),
        default="0",
        help="0 = Daily SIP clients only",
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
        "--dry-run", action="store_true", help="print the deletions and change nothing"
    )
    teardown.set_defaults(func=cmd_teardown)

    call = subparsers.add_parser("call", help="place a test call")
    call.add_argument(
        "direction",
        choices=("dial-in", "dial-out"),
        help="dial-in calls the bot; dial-out has the bot call out",
    )
    call.add_argument("--dry-run", action="store_true", help="print the request and change nothing")
    call.set_defaults(func=cmd_call)

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
