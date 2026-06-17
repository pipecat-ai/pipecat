#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Keenable web search service wrapper."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from pipecat.services.keenable.search import (
    _MCP_URL,
    KeenableWebSearch,
)


class _FakeStreamableHttpParameters:
    """Records the args KeenableWebSearch builds the MCP transport with."""

    def __init__(self, url, headers):
        self.url = url
        self.headers = headers


class _FakeLLM:
    """Captures register_function(name, handler) calls."""

    def __init__(self):
        self.registered = {}

    def register_function(self, name, handler):
        self.registered[name] = handler


def _fake_tools_schema():
    return SimpleNamespace(
        standard_tools=[
            SimpleNamespace(name="search_web_pages"),
            SimpleNamespace(name="fetch_page_content"),
        ]
    )


def _fake_mcp_deps():
    """Return a (MCPClient factory, client, StreamableHttpParameters) triple.

    The mock client captures the ``server_params`` it was built with and
    exposes async ``start``/``close``/``get_tools_schema``/``_tool_wrapper``.
    """
    client = AsyncMock()
    client.get_tools_schema.return_value = _fake_tools_schema()

    def make_client(*, server_params, **kwargs):
        client.server_params = server_params
        return client

    factory = MagicMock(side_effect=make_client)
    return factory, client, _FakeStreamableHttpParameters


class TestKeenableWebSearchConfig(unittest.TestCase):
    """Construction-time config: mode resolution and header building."""

    @patch.dict(os.environ, {}, clear=True)
    def test_default_mode_pro_when_keyless(self):
        self.assertEqual(KeenableWebSearch()._mode, "pro")

    @patch.dict(os.environ, {}, clear=True)
    def test_default_mode_realtime_when_keyed(self):
        self.assertEqual(KeenableWebSearch(api_key="k")._mode, "realtime")

    @patch.dict(os.environ, {}, clear=True)
    def test_explicit_mode_overrides_keyed_default(self):
        # An explicit mode wins over the key-based default.
        self.assertEqual(KeenableWebSearch(api_key="k", mode="pro")._mode, "pro")

    @patch.dict(os.environ, {}, clear=True)
    def test_explicit_mode_honored(self):
        self.assertEqual(KeenableWebSearch(mode="realtime")._mode, "realtime")

    @patch.dict(os.environ, {"KEENABLE_SEARCH_MODE": "realtime"}, clear=True)
    def test_env_overrides_mode(self):
        # Even with mode="pro" passed, the env var wins.
        self.assertEqual(KeenableWebSearch(mode="pro")._mode, "realtime")

    @patch.dict(os.environ, {"KEENABLE_SEARCH_MODE": "bogus"}, clear=True)
    def test_invalid_env_mode_ignored(self):
        # Invalid env mode is ignored; the passed mode is kept.
        self.assertEqual(KeenableWebSearch(mode="pro")._mode, "pro")

    @patch.dict(os.environ, {}, clear=True)
    def test_headers_keyless(self):
        headers = KeenableWebSearch()._build_headers()
        self.assertNotIn("X-API-Key", headers)
        self.assertTrue(headers["User-Agent"].startswith("pipecat/"))

    @patch.dict(os.environ, {}, clear=True)
    def test_headers_with_explicit_key(self):
        headers = KeenableWebSearch(api_key="secret")._build_headers()
        self.assertEqual(headers["X-API-Key"], "secret")

    @patch.dict(os.environ, {"KEENABLE_API_KEY": "from-env"}, clear=True)
    def test_api_key_falls_back_to_env(self):
        headers = KeenableWebSearch()._build_headers()
        self.assertEqual(headers["X-API-Key"], "from-env")

    @patch.dict(os.environ, {"KEENABLE_API_KEY": "explicit-wins"}, clear=True)
    def test_explicit_key_overrides_env(self):
        headers = KeenableWebSearch(api_key="explicit")._build_headers()
        self.assertEqual(headers["X-API-Key"], "explicit")


class TestKeenableWebSearchConnection(unittest.IsolatedAsyncioTestCase):
    """Connection lifecycle and delegation to the underlying MCPClient."""

    @patch.dict(os.environ, {}, clear=True)
    async def test_methods_raise_before_connect(self):
        search = KeenableWebSearch()
        with self.assertRaises(RuntimeError):
            await search.register_tools(object())
        with self.assertRaises(RuntimeError):
            await search.get_tools_schema()

    @patch.dict(os.environ, {}, clear=True)
    async def test_start_builds_transport_and_registers(self):
        factory, client, params_cls = _fake_mcp_deps()
        llm = _FakeLLM()
        with patch(
            "pipecat.services.keenable.search._import_mcp_deps",
            return_value=(factory, params_cls),
        ):
            async with KeenableWebSearch(api_key="k", mode="realtime") as search:
                # MCPClient built with the plain endpoint URL (mode goes per call).
                params = client.server_params
                self.assertEqual(params.url, _MCP_URL)
                self.assertEqual(params.headers["X-API-Key"], "k")
                client.start.assert_awaited_once()

                await search.register_tools(llm)
                self.assertEqual(set(llm.registered), {"search_web_pages", "fetch_page_content"})
                # Search tool gets the mode-injecting handler; others delegate raw.
                self.assertEqual(llm.registered["search_web_pages"], search._mode_injecting_handler)
                self.assertEqual(llm.registered["fetch_page_content"], client._tool_wrapper)

        # Context manager exit closes the connection.
        client.close.assert_awaited_once()

    @patch.dict(os.environ, {}, clear=True)
    async def test_mode_injected_into_search_call(self):
        factory, client, params_cls = _fake_mcp_deps()
        with patch(
            "pipecat.services.keenable.search._import_mcp_deps",
            return_value=(factory, params_cls),
        ):
            async with KeenableWebSearch(mode="realtime") as search:
                # Existing args preserved, mode added.
                params = SimpleNamespace(arguments={"query": "x"})
                await search._mode_injecting_handler(params)
                self.assertEqual(params.arguments, {"query": "x", "mode": "realtime"})
                client._tool_wrapper.assert_awaited_once_with(params)

                # Works when the LLM supplies no arguments.
                params2 = SimpleNamespace(arguments=None)
                await search._mode_injecting_handler(params2)
                self.assertEqual(params2.arguments, {"mode": "realtime"})

    @patch.dict(os.environ, {}, clear=True)
    async def test_start_is_idempotent(self):
        factory, client, params_cls = _fake_mcp_deps()
        with patch(
            "pipecat.services.keenable.search._import_mcp_deps",
            return_value=(factory, params_cls),
        ):
            search = KeenableWebSearch()
            await search.start()
            await search.start()  # second call is a no-op
            factory.assert_called_once()
            await search.close()


if __name__ == "__main__":
    unittest.main()
