#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Guards against server-side request forgery via client-supplied URLs."""

import asyncio
import ipaddress
from collections.abc import Sequence
from typing import Literal
from urllib.parse import urlsplit

IPNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network

UrlReachability = Literal["public", "allowed", "blocked"]


async def classify_url_reachability(
    url: str, allowed_networks: Sequence[IPNetwork] = ()
) -> UrlReachability:
    """Classify who can safely reach `url`'s host: the public internet, this server, or no one.

    Resolution happens once, up front; it doesn't defend against a target
    that resolves differently between this check and the actual connection
    (DNS rebinding).

    Args:
        url: The URL to classify.
        allowed_networks: Private networks this server is trusted to reach
            directly (e.g. where a deployment's own file servers live),
            beyond the public internet.

    Returns:
        "public" if every resolved address is publicly routable — safe to
            hand to a third party (e.g. an LLM provider) to fetch itself.
        "allowed" if not publicly routable, but every address falls within
            `allowed_networks` — safe for this server to fetch directly, but
            not to hand off to a third party that can't reach it.
        "blocked" otherwise: loopback, link-local (including the cloud
            metadata range ``169.254.0.0/16``), private (RFC 1918), or
            unique-local (IPv6 ULA) addresses outside `allowed_networks`, or
            a host that fails to resolve.
    """
    hostname = urlsplit(url).hostname
    if not hostname:
        return "blocked"
    try:
        addr_infos = await asyncio.get_event_loop().getaddrinfo(hostname, None)
    except OSError:
        return "blocked"
    if not addr_infos:
        return "blocked"

    ips = [ipaddress.ip_address(info[4][0]) for info in addr_infos]
    if all(ip.is_global for ip in ips):
        return "public"
    if all(ip.is_global or any(ip in network for network in allowed_networks) for ip in ips):
        return "allowed"
    return "blocked"
