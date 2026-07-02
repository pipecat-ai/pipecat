#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Keenable web search service."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# MCP is an optional dependency (the `keenable` extra); skip the whole module if
# it isn't installed.
pytest.importorskip("mcp")

from pipecat.services.keenable.search import (  # noqa: E402
    _SERVER_FETCH_TOOL,
    _SERVER_SEARCH_TOOL,
    KeenableWebSearch,
)


class _FakeStreamableHttpParameters:
    """Records the args KeenableWebSearch builds the MCP transport with."""

    def __init__(self, url, headers):
        self.url = url
        self.headers = headers


def _patch_mcp():
    """Patch the module-level MCPClient / StreamableHttpParameters.

    Returns a (patch_context, client) pair. The mock client captures the
    ``server_params`` it was built with and exposes async
    ``start``/``close``/``call_tool``.
    """
    client = AsyncMock()
    client.call_tool.return_value = "RESULT"

    def make_client(*, server_params, **kwargs):
        client.server_params = server_params
        return client

    factory = MagicMock(side_effect=make_client)
    ctx = patch.multiple(
        "pipecat.services.keenable.search",
        MCPClient=factory,
        StreamableHttpParameters=_FakeStreamableHttpParameters,
    )
    return ctx, factory, client


class TestKeenableWebSearchConfig(unittest.TestCase):
    """Construction-time config: mode resolution and header building."""

    def test_default_mode_pro_when_keyless(self):
        self.assertEqual(KeenableWebSearch()._mode, "pro")

    def test_default_mode_realtime_when_keyed(self):
        self.assertEqual(KeenableWebSearch(api_key="k")._mode, "realtime")

    def test_explicit_mode_overrides_keyed_default(self):
        # An explicit mode wins over the key-based default.
        self.assertEqual(KeenableWebSearch(api_key="k", mode="pro")._mode, "pro")

    def test_explicit_mode_honored(self):
        self.assertEqual(KeenableWebSearch(mode="realtime")._mode, "realtime")

    def test_headers_keyless(self):
        headers = KeenableWebSearch()._build_headers()
        self.assertNotIn("X-API-Key", headers)
        self.assertTrue(headers["User-Agent"].startswith("pipecat/"))

    def test_headers_with_explicit_key(self):
        headers = KeenableWebSearch(api_key="secret")._build_headers()
        self.assertEqual(headers["X-API-Key"], "secret")


class TestKeenableWebSearchTools(unittest.TestCase):
    """The static tool schema (no connection required)."""

    def test_tools_schema_is_static_and_carries_handlers(self):
        search = KeenableWebSearch()
        tools = search.tools()
        by_name = {t.name: t for t in tools.standard_tools}
        # Two curated tools, built without any MCP connection.
        self.assertEqual(set(by_name), {"search_web", "fetch_page"})

        search_web = by_name["search_web"]
        self.assertEqual(search_web.required, ["query"])
        # Optional filters are advertised to the model.
        for key in (
            "site",
            "published_after",
            "published_before",
            "acquired_after",
            "acquired_before",
        ):
            self.assertIn(key, search_web.properties)
        # The embedded handler is what drives the LLM's auto-registration.
        self.assertEqual(search_web.handler, search._handle_search)

        fetch_page = by_name["fetch_page"]
        self.assertEqual(fetch_page.required, ["url"])
        self.assertIn("max_chars", fetch_page.properties)
        self.assertIn("live", fetch_page.properties)
        self.assertEqual(fetch_page.handler, search._handle_fetch)


class TestKeenableWebSearchConnection(unittest.IsolatedAsyncioTestCase):
    """Connection lifecycle and delegation to the underlying MCPClient."""

    async def test_start_builds_transport(self):
        ctx, factory, client = _patch_mcp()
        with ctx:
            search = KeenableWebSearch(api_key="k", mode="realtime")
            await search.start()
            # MCPClient built with the plain endpoint URL (mode goes per call).
            params = client.server_params
            self.assertEqual(params.url, "https://api.keenable.ai/mcp")
            self.assertEqual(params.headers["X-API-Key"], "k")
            client.start.assert_awaited_once()
            await search.close()
            client.close.assert_awaited_once()

    async def test_handle_search_injects_mode_and_returns_text(self):
        ctx, factory, client = _patch_mcp()
        with ctx:
            search = KeenableWebSearch(mode="realtime")
            result_callback = AsyncMock()
            params = SimpleNamespace(
                arguments={"query": "latest news"},
                result_callback=result_callback,
            )

            await search._handle_search(params)

            # Lazily connected on first use.
            client.start.assert_awaited_once()
            # Underlying server tool called with the mode injected.
            client.call_tool.assert_awaited_once_with(
                _SERVER_SEARCH_TOOL,
                {"query": "latest news", "mode": "realtime"},
            )
            # The tool's text result is sent back through the callback.
            result_callback.assert_awaited_once_with("RESULT")
            await search.close()

    async def test_handle_search_forwards_only_supplied_filters(self):
        ctx, factory, client = _patch_mcp()
        with ctx:
            search = KeenableWebSearch(mode="pro")
            params = SimpleNamespace(
                arguments={
                    "query": "ai news",
                    "site": "techcrunch.com",
                    "published_after": "2026-01-01",
                    # Unsupplied filters and explicit None must not be forwarded.
                    "acquired_before": None,
                },
                result_callback=AsyncMock(),
            )

            await search._handle_search(params)

            client.call_tool.assert_awaited_once_with(
                _SERVER_SEARCH_TOOL,
                {
                    "query": "ai news",
                    "mode": "pro",
                    "site": "techcrunch.com",
                    "published_after": "2026-01-01",
                },
            )
            await search.close()

    async def test_handle_fetch_calls_fetch_tool_without_mode(self):
        ctx, factory, client = _patch_mcp()
        with ctx:
            search = KeenableWebSearch(mode="realtime")
            result_callback = AsyncMock()
            params = SimpleNamespace(
                arguments={"url": "https://example.com", "max_chars": 2000},
                result_callback=result_callback,
            )

            await search._handle_fetch(params)

            # Fetch takes no search mode; only supplied options are forwarded.
            client.call_tool.assert_awaited_once_with(
                _SERVER_FETCH_TOOL,
                {"url": "https://example.com", "max_chars": 2000},
            )
            result_callback.assert_awaited_once_with("RESULT")
            await search.close()

    async def test_start_is_idempotent(self):
        ctx, factory, client = _patch_mcp()
        with ctx:
            search = KeenableWebSearch()
            await search.start()
            await search.start()  # second call is a no-op
            factory.assert_called_once()
            await search.close()

    async def test_close_resets_connection(self):
        ctx, factory, client = _patch_mcp()
        with ctx:
            search = KeenableWebSearch()
            await search.start()
            await search.close()
            client.close.assert_awaited_once()
            # A subsequent start reconnects (builds a new client).
            await search.start()
            self.assertEqual(factory.call_count, 2)
            await search.close()

    async def test_start_failure_resets_state(self):
        ctx, factory, client = _patch_mcp()
        client.start.side_effect = RuntimeError("boom")
        with ctx:
            search = KeenableWebSearch()
            with self.assertRaises(RuntimeError):
                await search.start()
            # State is reset so close() is a safe no-op and a retry can reconnect.
            self.assertIsNone(search._session_task)
            await search.close()


if __name__ == "__main__":
    unittest.main()
