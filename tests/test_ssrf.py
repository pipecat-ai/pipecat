#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for pipecat.utils.security.ssrf.classify_url_reachability."""

import ipaddress
import socket
import unittest
from unittest.mock import AsyncMock, patch

from pipecat.utils.security.ssrf import classify_url_reachability


def _addr_info(*ips: str) -> list[tuple]:
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0)) for ip in ips]


class TestClassifyUrlReachability(unittest.IsolatedAsyncioTestCase):
    async def test_public_for_a_public_address(self):
        with patch(
            "asyncio.get_event_loop",
            return_value=AsyncMock(getaddrinfo=AsyncMock(return_value=_addr_info("8.8.8.8"))),
        ):
            self.assertEqual(
                await classify_url_reachability("https://example.com/doc.pdf"), "public"
            )

    async def test_blocked_for_loopback(self):
        with patch(
            "asyncio.get_event_loop",
            return_value=AsyncMock(getaddrinfo=AsyncMock(return_value=_addr_info("127.0.0.1"))),
        ):
            self.assertEqual(await classify_url_reachability("http://localhost/doc.pdf"), "blocked")

    async def test_blocked_for_cloud_metadata_link_local_address(self):
        with patch(
            "asyncio.get_event_loop",
            return_value=AsyncMock(
                getaddrinfo=AsyncMock(return_value=_addr_info("169.254.169.254"))
            ),
        ):
            self.assertEqual(
                await classify_url_reachability("http://169.254.169.254/latest/meta-data/"),
                "blocked",
            )

    async def test_blocked_for_private_rfc1918_address_with_no_allowed_networks(self):
        with patch(
            "asyncio.get_event_loop",
            return_value=AsyncMock(getaddrinfo=AsyncMock(return_value=_addr_info("10.0.0.5"))),
        ):
            self.assertEqual(await classify_url_reachability("http://10.0.0.5/internal"), "blocked")

    async def test_blocked_if_any_resolved_address_is_not_public(self):
        """A hostname resolving to a mix of public and private addresses is blocked."""
        with patch(
            "asyncio.get_event_loop",
            return_value=AsyncMock(
                getaddrinfo=AsyncMock(return_value=_addr_info("8.8.8.8", "127.0.0.1"))
            ),
        ):
            self.assertEqual(
                await classify_url_reachability("https://example.com/doc.pdf"), "blocked"
            )

    async def test_blocked_when_resolution_fails(self):
        with patch(
            "asyncio.get_event_loop",
            return_value=AsyncMock(getaddrinfo=AsyncMock(side_effect=OSError("no such host"))),
        ):
            self.assertEqual(
                await classify_url_reachability("https://does-not-resolve.invalid/doc.pdf"),
                "blocked",
            )

    async def test_blocked_for_url_without_a_host(self):
        self.assertEqual(await classify_url_reachability("not-a-url"), "blocked")

    async def test_allowed_for_private_address_within_an_allowed_network(self):
        with patch(
            "asyncio.get_event_loop",
            return_value=AsyncMock(getaddrinfo=AsyncMock(return_value=_addr_info("10.0.0.5"))),
        ):
            self.assertEqual(
                await classify_url_reachability(
                    "http://10.0.0.5/internal",
                    allowed_networks=[ipaddress.ip_network("10.0.0.0/8")],
                ),
                "allowed",
            )

    async def test_blocked_for_private_address_outside_the_allowed_networks(self):
        with patch(
            "asyncio.get_event_loop",
            return_value=AsyncMock(getaddrinfo=AsyncMock(return_value=_addr_info("192.168.1.5"))),
        ):
            self.assertEqual(
                await classify_url_reachability(
                    "http://192.168.1.5/internal",
                    allowed_networks=[ipaddress.ip_network("10.0.0.0/8")],
                ),
                "blocked",
            )

    async def test_allowed_networks_does_not_relax_loopback(self):
        """An allowed private network doesn't widen the loopback block."""
        with patch(
            "asyncio.get_event_loop",
            return_value=AsyncMock(getaddrinfo=AsyncMock(return_value=_addr_info("127.0.0.1"))),
        ):
            self.assertEqual(
                await classify_url_reachability(
                    "http://localhost/doc.pdf",
                    allowed_networks=[ipaddress.ip_network("10.0.0.0/8")],
                ),
                "blocked",
            )

    async def test_public_for_a_public_address_even_with_allowed_networks_set(self):
        """A public address still classifies as public, not merely allowed."""
        with patch(
            "asyncio.get_event_loop",
            return_value=AsyncMock(getaddrinfo=AsyncMock(return_value=_addr_info("8.8.8.8"))),
        ):
            self.assertEqual(
                await classify_url_reachability(
                    "https://example.com/doc.pdf",
                    allowed_networks=[ipaddress.ip_network("10.0.0.0/8")],
                ),
                "public",
            )
