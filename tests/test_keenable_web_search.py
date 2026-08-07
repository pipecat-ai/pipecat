#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Keenable web search service."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# MCP is an optional dependency (the `keenable` extra); skip the whole module if
# it isn't installed.
pytest.importorskip("mcp")

from pipecat.services.keenable.search import (  # noqa: E402
    _SERVER_FETCH_TOOL,
    _SERVER_SEARCH_TOOL,
    _SERVER_URL,
    KeenableWebSearch,
)


class _FakeStreamableHttpParameters:
    """Records the args KeenableWebSearch builds the MCP transport with."""

    def __init__(self, url, headers):
        self.url = url
        self.headers = headers


def _patch_mcp():
    """Patch the module-level MCPClient / StreamableHttpParameters.

    Returns a (patch_context, factory, client) triple. The mock client captures
    the constructor arguments it was built with and exposes async
    ``tools``/``start``/``close``.
    """
    client = AsyncMock()

    def make_client(*, server_params, **kwargs):
        client.server_params = server_params
        client.client_kwargs = kwargs
        return client

    factory = MagicMock(side_effect=make_client)
    ctx = patch.multiple(
        "pipecat.services.keenable.search",
        MCPClient=factory,
        StreamableHttpParameters=_FakeStreamableHttpParameters,
    )
    return ctx, factory, client


class TestKeenableWebSearchConfig(unittest.TestCase):
    """Construction-time config: MCPClient wiring, mode resolution, headers."""

    def _build(self, **kwargs):
        ctx, factory, client = _patch_mcp()
        with ctx:
            search = KeenableWebSearch(**kwargs)
        return search, client

    def test_wires_mcp_client_for_keenable_server(self):
        search, client = self._build()
        self.assertEqual(client.server_params.url, _SERVER_URL)
        self.assertEqual(
            client.client_kwargs["tools_filter"], [_SERVER_SEARCH_TOOL, _SERVER_FETCH_TOOL]
        )

    def test_mode_pinned_via_tools_arguments(self):
        search, client = self._build(api_key="k", mode="realtime")
        self.assertEqual(
            client.client_kwargs["tools_arguments"], {_SERVER_SEARCH_TOOL: {"mode": "realtime"}}
        )

    def test_default_mode_pro_when_keyless(self):
        search, client = self._build()
        self.assertEqual(
            client.client_kwargs["tools_arguments"], {_SERVER_SEARCH_TOOL: {"mode": "pro"}}
        )

    def test_default_mode_realtime_when_keyed(self):
        search, client = self._build(api_key="k")
        self.assertEqual(
            client.client_kwargs["tools_arguments"], {_SERVER_SEARCH_TOOL: {"mode": "realtime"}}
        )

    def test_explicit_mode_overrides_keyed_default(self):
        search, client = self._build(api_key="k", mode="pro")
        self.assertEqual(
            client.client_kwargs["tools_arguments"], {_SERVER_SEARCH_TOOL: {"mode": "pro"}}
        )

    def test_explicit_mode_honored_when_keyless(self):
        search, client = self._build(mode="realtime")
        self.assertEqual(
            client.client_kwargs["tools_arguments"], {_SERVER_SEARCH_TOOL: {"mode": "realtime"}}
        )

    def test_headers_keyless(self):
        search, client = self._build()
        self.assertNotIn("X-API-Key", client.server_params.headers)
        self.assertTrue(client.server_params.headers["User-Agent"].startswith("pipecat/"))

    def test_headers_with_key(self):
        search, client = self._build(api_key="secret")
        self.assertEqual(client.server_params.headers["X-API-Key"], "secret")


class TestKeenableWebSearchDelegation(unittest.IsolatedAsyncioTestCase):
    """tools()/start()/close() delegate to the underlying MCPClient."""

    async def test_tools_delegates(self):
        ctx, factory, client = _patch_mcp()
        with ctx:
            search = KeenableWebSearch()
        client.tools.return_value = "TOOLS_SCHEMA"
        result = await search.tools()
        client.tools.assert_awaited_once()
        self.assertEqual(result, "TOOLS_SCHEMA")

    async def test_start_and_close_delegate(self):
        ctx, factory, client = _patch_mcp()
        with ctx:
            search = KeenableWebSearch()
        await search.start()
        client.start.assert_awaited_once()
        await search.close()
        client.close.assert_awaited_once()

    async def test_async_context_manager_starts_and_closes(self):
        ctx, factory, client = _patch_mcp()
        with ctx:
            search = KeenableWebSearch()
        async with search:
            client.start.assert_awaited_once()
        client.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
