#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Async Twilio REST client covering what the SIP examples provision.

Thin wrapper over ``httpx`` — no Twilio SDK. Two API hosts are involved:
``trunking.twilio.com/v1`` owns Elastic SIP trunks, their origination URLs,
and the phone numbers routed to them; ``api.twilio.com/2010-04-01`` owns the
account-level SIP IP access control lists and credential lists a trunk points
at, plus the Calls resource the auto-caller drives.

Every call returns Twilio's parsed JSON and raises :class:`TwilioError` on a
non-2xx, with the response body in the message::

    async with TwilioAPI.from_env() as api:
        trunks = await api.list_trunks()
"""

import os
from typing import Any

import httpx

API_BASE = "https://api.twilio.com/2010-04-01"
TRUNKING_BASE = "https://trunking.twilio.com/v1"

# Page size for the list endpoints. _list_all follows paging links regardless.
PAGE_SIZE = 50


class TwilioError(Exception):
    """A Twilio request failed, or its credentials are missing."""


def _form(fields: dict[str, Any]) -> dict[str, str]:
    """Render request fields as Twilio's form encoding.

    Args:
        fields: Field names to values; ``None`` values are dropped.

    Returns:
        Form body with booleans as ``true``/``false`` and everything else
        stringified.
    """
    body: dict[str, str] = {}
    for key, value in fields.items():
        if value is None:
            continue
        body[key] = str(value).lower() if isinstance(value, bool) else str(value)
    return body


async def get_public_ip(client: httpx.AsyncClient | None = None) -> str:
    """Look up this machine's public IP.

    Twilio's IP ACL authorizes the address Twilio sees traffic arrive from, so
    provision.py refreshes the entry with this value on every provision.

    Args:
        client: Client to issue the request on; one is created when omitted.

    Returns:
        Dotted-quad public address.
    """
    own = client is None
    client = client or httpx.AsyncClient(timeout=10)
    try:
        response = await client.get("https://api.ipify.org")
        if response.status_code >= 300:
            raise TwilioError(f"public IP lookup failed: {response.status_code} {response.text}")
        return response.text.strip()
    finally:
        if own:
            await client.aclose()


class TwilioAPI:
    """Twilio REST calls the SIP examples need, under one authenticated client."""

    def __init__(self, account_sid: str, auth_token: str, timeout: float = 30.0):
        """Initialize the client.

        Args:
            account_sid: Twilio account SID, used in both auth and URL paths.
            auth_token: Twilio auth token.
            timeout: Per-request timeout in seconds.
        """
        self.account_sid = account_sid
        self._client = httpx.AsyncClient(auth=(account_sid, auth_token), timeout=timeout)

    @classmethod
    def from_env(cls, **kwargs) -> "TwilioAPI":
        """Build a client from ``TWILIO_ACCOUNT_SID`` and ``TWILIO_AUTH_TOKEN``.

        Args:
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            Authenticated client.
        """
        sid = os.getenv("TWILIO_ACCOUNT_SID")
        token = os.getenv("TWILIO_AUTH_TOKEN")
        if not sid or not token:
            raise TwilioError(
                "TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN must both be set. "
                "Copy examples/sip/env.example to examples/sip/.env and fill the Twilio block."
            )
        return cls(sid, token, **kwargs)

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> "TwilioAPI":
        """Enter the context manager.

        Returns:
            This client.
        """
        return self

    async def __aexit__(self, *_exc) -> None:
        """Close the client on exit."""
        await self.aclose()

    # ------------------------------------------------------------- plumbing

    def account_url(self, suffix: str) -> str:
        """Build an account-scoped ``api.twilio.com`` URL.

        Args:
            suffix: Path under the account, e.g. ``SIP/CredentialLists.json``.

        Returns:
            Full request URL.
        """
        return f"{API_BASE}/Accounts/{self.account_sid}/{suffix}"

    async def request(
        self,
        method: str,
        url: str,
        *,
        data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        ok_404: bool = False,
    ) -> dict[str, Any]:
        """Issue one Twilio request.

        Args:
            method: HTTP method.
            url: Full request URL.
            data: Fields to form-encode as the body.
            params: Query parameters.
            ok_404: Treat a 404 as success, returning an empty dict.

        Returns:
            Parsed JSON body, or an empty dict for a 204 or a tolerated 404.
        """
        response = await self._client.request(
            method,
            url,
            data=_form(data) if data else None,
            params=_form(params) if params else None,
        )
        if ok_404 and response.status_code == 404:
            return {}
        if response.status_code >= 300:
            raise TwilioError(f"{method} {url} -> {response.status_code} {response.text}")
        if response.status_code == 204 or not response.content:
            return {}
        return response.json()

    async def _list_all(
        self, url: str, key: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Collect every page of a list endpoint.

        Both API hosts page differently — ``api.twilio.com`` returns a relative
        ``next_page_uri``, trunking a full ``meta.next_page_url``.

        Args:
            url: First page's URL.
            key: Property holding the items in each page.
            params: Query parameters for the first page.

        Returns:
            Items from every page.
        """
        page_params = {"PageSize": PAGE_SIZE, **(params or {})}
        items: list[dict[str, Any]] = []
        next_url: str | None = url
        while next_url:
            page = await self.request("GET", next_url, params=page_params)
            items.extend(page.get(key) or [])
            relative = page.get("next_page_uri")
            next_url = (
                f"https://api.twilio.com{relative}"
                if relative
                else (page.get("meta") or {}).get("next_page_url")
            )
            page_params = None if next_url else page_params
        return items

    # ---------------------------------------------------------------- trunks

    async def list_trunks(self) -> list[dict[str, Any]]:
        """List the account's Elastic SIP trunks.

        Returns:
            Trunk resources.
        """
        return await self._list_all(f"{TRUNKING_BASE}/Trunks", "trunks")

    async def create_trunk(
        self,
        friendly_name: str,
        domain_name: str,
        *,
        secure: bool = False,
    ) -> dict[str, Any]:
        """Create an Elastic SIP trunk.

        The response's ``symmetric_rtp_enabled`` is read-only — Trunk
        create/update accept no field for it, so media latching is a Console
        setting.

        Args:
            friendly_name: Name provision.py finds the trunk by.
            domain_name: Termination domain, ``<name>.pstn.twilio.com``.
            secure: Require TLS and SRTP on the termination side.

        Returns:
            Created trunk.
        """
        return await self.request(
            "POST",
            f"{TRUNKING_BASE}/Trunks",
            data={
                "FriendlyName": friendly_name,
                "DomainName": domain_name,
                "Secure": secure,
            },
        )

    async def update_trunk(self, trunk_sid: str, **fields: Any) -> dict[str, Any]:
        """Update a trunk's fields.

        Args:
            trunk_sid: Trunk to update.
            **fields: Twilio field names, e.g. ``Secure``, ``TransferMode``.

        Returns:
            Updated trunk.
        """
        return await self.request("POST", f"{TRUNKING_BASE}/Trunks/{trunk_sid}", data=fields)

    async def delete_trunk(self, trunk_sid: str) -> None:
        """Delete a trunk, tolerating one that is already gone.

        Args:
            trunk_sid: Trunk to delete.
        """
        await self.request("DELETE", f"{TRUNKING_BASE}/Trunks/{trunk_sid}", ok_404=True)

    # ------------------------------------------------------ origination URLs

    async def list_origination_urls(self, trunk_sid: str) -> list[dict[str, Any]]:
        """List where a trunk sends inbound calls.

        Args:
            trunk_sid: Trunk to read.

        Returns:
            Origination URL resources.
        """
        return await self._list_all(
            f"{TRUNKING_BASE}/Trunks/{trunk_sid}/OriginationUrls", "origination_urls"
        )

    async def create_origination_url(
        self,
        trunk_sid: str,
        *,
        friendly_name: str,
        sip_url: str,
        priority: int = 10,
        weight: int = 10,
        enabled: bool = True,
    ) -> dict[str, Any]:
        """Point a trunk's inbound calls at a SIP URI.

        Args:
            trunk_sid: Trunk to add the URL to.
            friendly_name: Name provision.py finds the URL by.
            sip_url: Destination, e.g. ``sip:user@domain;transport=tls``.
            priority: Lower is tried first.
            weight: Share of traffic among equal priorities.
            enabled: Whether Twilio routes to it.

        Returns:
            Created origination URL.
        """
        return await self.request(
            "POST",
            f"{TRUNKING_BASE}/Trunks/{trunk_sid}/OriginationUrls",
            data={
                "FriendlyName": friendly_name,
                "SipUrl": sip_url,
                "Priority": priority,
                "Weight": weight,
                "Enabled": enabled,
            },
        )

    async def update_origination_url(
        self, trunk_sid: str, url_sid: str, **fields: Any
    ) -> dict[str, Any]:
        """Update an origination URL's fields.

        Args:
            trunk_sid: Trunk owning the URL.
            url_sid: URL to update.
            **fields: Twilio field names, e.g. ``SipUrl``, ``Enabled``.

        Returns:
            Updated origination URL.
        """
        return await self.request(
            "POST", f"{TRUNKING_BASE}/Trunks/{trunk_sid}/OriginationUrls/{url_sid}", data=fields
        )

    async def delete_origination_url(self, trunk_sid: str, url_sid: str) -> None:
        """Delete an origination URL, tolerating one that is already gone.

        Args:
            trunk_sid: Trunk owning the URL.
            url_sid: URL to delete.
        """
        await self.request(
            "DELETE",
            f"{TRUNKING_BASE}/Trunks/{trunk_sid}/OriginationUrls/{url_sid}",
            ok_404=True,
        )

    # --------------------------------------------------------------- IP ACLs

    async def list_ip_acls(self) -> list[dict[str, Any]]:
        """List the account's SIP IP access control lists.

        Returns:
            ACL resources.
        """
        return await self._list_all(
            self.account_url("SIP/IpAccessControlLists.json"), "ip_access_control_lists"
        )

    async def create_ip_acl(self, friendly_name: str) -> dict[str, Any]:
        """Create an IP access control list.

        Args:
            friendly_name: Name provision.py finds the list by.

        Returns:
            Created ACL.
        """
        return await self.request(
            "POST",
            self.account_url("SIP/IpAccessControlLists.json"),
            data={"FriendlyName": friendly_name},
        )

    async def delete_ip_acl(self, acl_sid: str) -> None:
        """Delete an IP access control list, tolerating one already gone.

        Args:
            acl_sid: ACL to delete.
        """
        await self.request(
            "DELETE", self.account_url(f"SIP/IpAccessControlLists/{acl_sid}.json"), ok_404=True
        )

    async def list_ip_acl_addresses(self, acl_sid: str) -> list[dict[str, Any]]:
        """List the addresses an ACL authorizes.

        Args:
            acl_sid: ACL to read.

        Returns:
            IP address resources.
        """
        return await self._list_all(
            self.account_url(f"SIP/IpAccessControlLists/{acl_sid}/IpAddresses.json"),
            "ip_addresses",
        )

    async def create_ip_acl_address(
        self,
        acl_sid: str,
        *,
        friendly_name: str,
        ip_address: str,
        cidr_prefix_length: int = 32,
    ) -> dict[str, Any]:
        """Authorize an address in an ACL.

        Args:
            acl_sid: ACL to add to.
            friendly_name: Name provision.py finds the entry by.
            ip_address: Address to authorize.
            cidr_prefix_length: Prefix length; 32 is a single address.

        Returns:
            Created IP address resource.
        """
        return await self.request(
            "POST",
            self.account_url(f"SIP/IpAccessControlLists/{acl_sid}/IpAddresses.json"),
            data={
                "FriendlyName": friendly_name,
                "IpAddress": ip_address,
                "CidrPrefixLength": cidr_prefix_length,
            },
        )

    async def update_ip_acl_address(
        self, acl_sid: str, address_sid: str, **fields: Any
    ) -> dict[str, Any]:
        """Update an authorized address.

        Args:
            acl_sid: ACL owning the entry.
            address_sid: Entry to update.
            **fields: Twilio field names, e.g. ``IpAddress``.

        Returns:
            Updated IP address resource.
        """
        return await self.request(
            "POST",
            self.account_url(f"SIP/IpAccessControlLists/{acl_sid}/IpAddresses/{address_sid}.json"),
            data=fields,
        )

    async def delete_ip_acl_address(self, acl_sid: str, address_sid: str) -> None:
        """Remove an authorized address, tolerating one already gone.

        Args:
            acl_sid: ACL owning the entry.
            address_sid: Entry to delete.
        """
        await self.request(
            "DELETE",
            self.account_url(f"SIP/IpAccessControlLists/{acl_sid}/IpAddresses/{address_sid}.json"),
            ok_404=True,
        )

    # ------------------------------------------------------ credential lists

    async def list_credential_lists(self) -> list[dict[str, Any]]:
        """List the account's SIP credential lists.

        Returns:
            Credential list resources.
        """
        return await self._list_all(
            self.account_url("SIP/CredentialLists.json"), "credential_lists"
        )

    async def create_credential_list(self, friendly_name: str) -> dict[str, Any]:
        """Create a SIP credential list.

        Args:
            friendly_name: Name provision.py finds the list by.

        Returns:
            Created credential list.
        """
        return await self.request(
            "POST",
            self.account_url("SIP/CredentialLists.json"),
            data={"FriendlyName": friendly_name},
        )

    async def delete_credential_list(self, list_sid: str) -> None:
        """Delete a credential list, tolerating one already gone.

        Args:
            list_sid: List to delete.
        """
        await self.request(
            "DELETE", self.account_url(f"SIP/CredentialLists/{list_sid}.json"), ok_404=True
        )

    async def list_credentials(self, list_sid: str) -> list[dict[str, Any]]:
        """List the usernames in a credential list.

        Args:
            list_sid: List to read.

        Returns:
            Credential resources (passwords are never returned).
        """
        return await self._list_all(
            self.account_url(f"SIP/CredentialLists/{list_sid}/Credentials.json"), "credentials"
        )

    async def create_credential(
        self, list_sid: str, username: str, password: str
    ) -> dict[str, Any]:
        """Add a credential to a list.

        Args:
            list_sid: List to add to.
            username: SIP username.
            password: SIP password; Twilio requires 12+ characters with a
                digit, a lower-case and an upper-case letter.

        Returns:
            Created credential.
        """
        return await self.request(
            "POST",
            self.account_url(f"SIP/CredentialLists/{list_sid}/Credentials.json"),
            data={"Username": username, "Password": password},
        )

    async def update_credential(
        self, list_sid: str, credential_sid: str, password: str
    ) -> dict[str, Any]:
        """Reset a credential's password.

        Args:
            list_sid: List owning the credential.
            credential_sid: Credential to update.
            password: New password.

        Returns:
            Updated credential.
        """
        return await self.request(
            "POST",
            self.account_url(f"SIP/CredentialLists/{list_sid}/Credentials/{credential_sid}.json"),
            data={"Password": password},
        )

    async def delete_credential(self, list_sid: str, credential_sid: str) -> None:
        """Delete a credential, tolerating one already gone.

        Args:
            list_sid: List owning the credential.
            credential_sid: Credential to delete.
        """
        await self.request(
            "DELETE",
            self.account_url(f"SIP/CredentialLists/{list_sid}/Credentials/{credential_sid}.json"),
            ok_404=True,
        )

    # ------------------------------------------------------- trunk auth sets

    async def list_trunk_ip_acls(self, trunk_sid: str) -> list[dict[str, Any]]:
        """List the IP ACLs a trunk authorizes termination from.

        Args:
            trunk_sid: Trunk to read.

        Returns:
            ACL resources attached to the trunk.
        """
        return await self._list_all(
            f"{TRUNKING_BASE}/Trunks/{trunk_sid}/IpAccessControlLists", "ip_access_control_lists"
        )

    async def attach_ip_acl(self, trunk_sid: str, acl_sid: str) -> dict[str, Any]:
        """Authorize a trunk's termination from an ACL's addresses.

        Args:
            trunk_sid: Trunk to attach to.
            acl_sid: ACL to attach.

        Returns:
            Attached ACL resource.
        """
        return await self.request(
            "POST",
            f"{TRUNKING_BASE}/Trunks/{trunk_sid}/IpAccessControlLists",
            data={"IpAccessControlListSid": acl_sid},
        )

    async def detach_ip_acl(self, trunk_sid: str, acl_sid: str) -> None:
        """Detach an ACL from a trunk, tolerating one already detached.

        Args:
            trunk_sid: Trunk to detach from.
            acl_sid: ACL to detach.
        """
        await self.request(
            "DELETE",
            f"{TRUNKING_BASE}/Trunks/{trunk_sid}/IpAccessControlLists/{acl_sid}",
            ok_404=True,
        )

    async def list_trunk_credential_lists(self, trunk_sid: str) -> list[dict[str, Any]]:
        """List the credential lists a trunk authorizes termination from.

        Args:
            trunk_sid: Trunk to read.

        Returns:
            Credential list resources attached to the trunk.
        """
        return await self._list_all(
            f"{TRUNKING_BASE}/Trunks/{trunk_sid}/CredentialLists", "credential_lists"
        )

    async def attach_credential_list(self, trunk_sid: str, list_sid: str) -> dict[str, Any]:
        """Authorize a trunk's termination from a credential list.

        Args:
            trunk_sid: Trunk to attach to.
            list_sid: Credential list to attach.

        Returns:
            Attached credential list resource.
        """
        return await self.request(
            "POST",
            f"{TRUNKING_BASE}/Trunks/{trunk_sid}/CredentialLists",
            data={"CredentialListSid": list_sid},
        )

    async def detach_credential_list(self, trunk_sid: str, list_sid: str) -> None:
        """Detach a credential list from a trunk, tolerating one already gone.

        Args:
            trunk_sid: Trunk to detach from.
            list_sid: Credential list to detach.
        """
        await self.request(
            "DELETE", f"{TRUNKING_BASE}/Trunks/{trunk_sid}/CredentialLists/{list_sid}", ok_404=True
        )

    # --------------------------------------------------------- phone numbers

    async def list_incoming_phone_numbers(
        self, phone_number: str | None = None
    ) -> list[dict[str, Any]]:
        """List the numbers the account owns.

        Args:
            phone_number: Filter to one number in E.164 form.

        Returns:
            Incoming phone number resources.
        """
        params = {"PhoneNumber": phone_number} if phone_number else None
        return await self._list_all(
            self.account_url("IncomingPhoneNumbers.json"), "incoming_phone_numbers", params
        )

    async def list_available_local_numbers(
        self, *, country: str = "US", area_code: str | None = None, limit: int = 5
    ) -> list[dict[str, Any]]:
        """Search Twilio's inventory for buyable local numbers.

        Args:
            country: ISO country code.
            area_code: Area code to search within.
            limit: How many results to return.

        Returns:
            Available phone number resources.
        """
        page = await self.request(
            "GET",
            self.account_url(f"AvailablePhoneNumbers/{country}/Local.json"),
            params={"AreaCode": area_code, "VoiceEnabled": True, "PageSize": limit},
        )
        return page.get("available_phone_numbers") or []

    async def buy_phone_number(self, phone_number: str, friendly_name: str) -> dict[str, Any]:
        """Purchase a number. Costs money — callers gate this on confirmation.

        Args:
            phone_number: Number to buy, in E.164 form.
            friendly_name: Name to label it with.

        Returns:
            Purchased incoming phone number resource.
        """
        return await self.request(
            "POST",
            self.account_url("IncomingPhoneNumbers.json"),
            data={"PhoneNumber": phone_number, "FriendlyName": friendly_name},
        )

    async def list_trunk_phone_numbers(self, trunk_sid: str) -> list[dict[str, Any]]:
        """List the numbers routed to a trunk.

        Args:
            trunk_sid: Trunk to read.

        Returns:
            Phone number resources attached to the trunk.
        """
        return await self._list_all(
            f"{TRUNKING_BASE}/Trunks/{trunk_sid}/PhoneNumbers", "phone_numbers"
        )

    async def attach_phone_number(self, trunk_sid: str, number_sid: str) -> dict[str, Any]:
        """Route a number the account owns to a trunk.

        Args:
            trunk_sid: Trunk to route to.
            number_sid: Number to attach.

        Returns:
            Attached phone number resource.
        """
        return await self.request(
            "POST",
            f"{TRUNKING_BASE}/Trunks/{trunk_sid}/PhoneNumbers",
            data={"PhoneNumberSid": number_sid},
        )

    async def detach_phone_number(self, trunk_sid: str, number_sid: str) -> None:
        """Stop routing a number to a trunk, keeping the number.

        Args:
            trunk_sid: Trunk to detach from.
            number_sid: Number to detach.
        """
        await self.request(
            "DELETE",
            f"{TRUNKING_BASE}/Trunks/{trunk_sid}/PhoneNumbers/{number_sid}",
            ok_404=True,
        )

    async def release_phone_number(self, number_sid: str) -> None:
        """Release a number back to Twilio, ending its monthly charge.

        Args:
            number_sid: Number to release.
        """
        await self.request(
            "DELETE", self.account_url(f"IncomingPhoneNumbers/{number_sid}.json"), ok_404=True
        )

    # ------------------------------------------------------------------ calls

    async def create_call(self, *, to: str, from_: str, twiml: str) -> dict[str, Any]:
        """Place a call that plays inline TwiML.

        Args:
            to: Destination — a phone number or a SIP URI.
            from_: Caller ID; a number the account owns or the trunk's number.
            twiml: TwiML the call executes once answered.

        Returns:
            Created call resource.
        """
        return await self.request(
            "POST",
            self.account_url("Calls.json"),
            data={"To": to, "From": from_, "Twiml": twiml},
        )

    async def fetch_call(self, call_sid: str) -> dict[str, Any]:
        """Read a call's current state.

        Args:
            call_sid: Call to read.

        Returns:
            Call resource.
        """
        return await self.request("GET", self.account_url(f"Calls/{call_sid}.json"))

    async def complete_call(self, call_sid: str) -> dict[str, Any]:
        """Hang up a call, tolerating one that already ended.

        Args:
            call_sid: Call to hang up.

        Returns:
            Call resource, or an empty dict when the call was already gone.
        """
        try:
            return await self.request(
                "POST", self.account_url(f"Calls/{call_sid}.json"), data={"Status": "completed"}
            )
        except TwilioError:
            return {}
