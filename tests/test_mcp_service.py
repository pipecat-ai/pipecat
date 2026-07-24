#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the MCP client service."""

import asyncio
import io
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from loguru import logger

# MCP is an optional dependency (the `mcp` extra); skip the whole module if it
# isn't installed.
pytest.importorskip("mcp")

from mcp.client.session_group import StreamableHttpParameters  # noqa: E402

from pipecat.services.llm_service import LLMService  # noqa: E402
from pipecat.services.mcp_service import MCPClient  # noqa: E402


def _tool(name, properties=None, required=None, description="A tool."):
    """Build a fake MCP server tool as returned by ``session.list_tools()``."""
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema={"properties": properties or {}, "required": required or []},
    )


class _FakeTransport:
    """Fake streamablehttp_client context manager; records enter/exit tasks."""

    def __init__(self, record):
        self._record = record

    async def __aenter__(self):
        self._record["enters"] = self._record.get("enters", 0) + 1
        self._record["enter_task"] = asyncio.current_task()
        return (MagicMock(), MagicMock(), MagicMock())

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._record["exits"] = self._record.get("exits", 0) + 1
        self._record["exit_task"] = asyncio.current_task()
        return False


class _FakeSession:
    """Fake mcp ClientSession with canned tools and call results."""

    def __init__(self, tools, record, fail_initializes=0):
        self._tools = tools
        self._record = record
        self._fail_initializes = fail_initializes
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    async def initialize(self):
        if self._fail_initializes > 0:
            self._fail_initializes -= 1
            raise RuntimeError("connect failed")
        self._record["initializes"] = self._record.get("initializes", 0) + 1

    async def list_tools(self):
        return SimpleNamespace(tools=self._tools)

    async def call_tool(self, name, arguments=None):
        self.calls.append((name, arguments))
        return SimpleNamespace(content=[SimpleNamespace(text=f"{name}-RESULT")])


class MCPClientTestBase(unittest.IsolatedAsyncioTestCase):
    """Builds MCPClients against a fake transport/session pair."""

    def _make_client(self, tools, fail_initializes=0, **client_kwargs):
        record = {}
        session = _FakeSession(tools, record, fail_initializes)
        ctx = patch.multiple(
            "pipecat.services.mcp_service",
            streamablehttp_client=lambda **kwargs: _FakeTransport(record),
            ClientSession=lambda read, write: session,
        )
        ctx.start()
        self.addCleanup(ctx.stop)
        client = MCPClient(
            server_params=StreamableHttpParameters(url="http://test/mcp"),
            **client_kwargs,
        )
        self.addAsyncCleanup(client.close)
        return client, session, record


class TestTools(MCPClientTestBase):
    """tools(): JIT start, handler attachment, filtering."""

    async def test_tools_starts_connection_and_attaches_handlers(self):
        client, session, record = self._make_client([_tool("tool_a"), _tool("tool_b")])
        tools_schema = await client.tools()
        self.assertEqual(record["initializes"], 1)
        self.assertEqual({s.name for s in tools_schema.standard_tools}, {"tool_a", "tool_b"})
        for schema in tools_schema.standard_tools:
            self.assertIsNotNone(schema.handler)
        await client.close()

    async def test_tools_is_idempotent_on_connection(self):
        client, session, record = self._make_client([_tool("tool_a")])
        await client.tools()
        await client.tools()
        self.assertEqual(record["enters"], 1)
        self.assertEqual(record["initializes"], 1)
        await client.close()

    async def test_tools_respects_tools_filter(self):
        client, session, record = self._make_client(
            [_tool("tool_a"), _tool("tool_b")], tools_filter=["tool_b"]
        )
        tools_schema = await client.tools()
        self.assertEqual({s.name for s in tools_schema.standard_tools}, {"tool_b"})
        await client.close()

    async def test_handler_calls_tool_and_delivers_result(self):
        client, session, record = self._make_client(
            [_tool("tool_a", properties={"x": {"type": "string"}})]
        )
        tools_schema = await client.tools()
        schema = tools_schema.standard_tools[0]
        params = SimpleNamespace(
            function_name="tool_a",
            tool_call_id="call-1",
            arguments={"x": "hello"},
            result_callback=AsyncMock(),
        )
        await schema.handler(params)
        self.assertEqual(session.calls, [("tool_a", {"x": "hello"})])
        params.result_callback.assert_awaited_once_with("tool_a-RESULT")
        await client.close()


class TestToolsArguments(MCPClientTestBase):
    """tools_arguments: schema stripping and call-time injection."""

    def _search_tools(self):
        return [
            _tool(
                "search",
                properties={"query": {"type": "string"}, "mode": {"type": "string"}},
                required=["query", "mode"],
            ),
            _tool("other", properties={"x": {"type": "string"}}, required=["x"]),
        ]

    async def test_fixed_arguments_hidden_from_advertised_schema(self):
        client, session, record = self._make_client(
            self._search_tools(), tools_arguments={"search": {"mode": "realtime"}}
        )
        tools_schema = await client.tools()
        by_name = {s.name: s for s in tools_schema.standard_tools}
        self.assertEqual(set(by_name["search"].properties), {"query"})
        self.assertEqual(by_name["search"].required, ["query"])
        # Other tools are untouched.
        self.assertEqual(set(by_name["other"].properties), {"x"})
        self.assertEqual(by_name["other"].required, ["x"])
        await client.close()

    async def test_fixed_arguments_injected_and_win_over_model_arguments(self):
        client, session, record = self._make_client(
            self._search_tools(), tools_arguments={"search": {"mode": "realtime"}}
        )
        await client.start()
        await client.call_tool("search", {"query": "news", "mode": "model-supplied"})
        self.assertEqual(session.calls, [("search", {"query": "news", "mode": "realtime"})])
        await client.close()

    async def test_fixed_arguments_injected_when_no_model_arguments(self):
        client, session, record = self._make_client(
            self._search_tools(), tools_arguments={"search": {"mode": "realtime"}}
        )
        await client.start()
        await client.call_tool("search", None)
        self.assertEqual(session.calls, [("search", {"mode": "realtime"})])
        await client.close()

    async def test_fixed_argument_absent_from_server_schema_still_injected(self):
        client, session, record = self._make_client(
            self._search_tools(), tools_arguments={"other": {"hidden": 1}}
        )
        tools_schema = await client.tools()
        by_name = {s.name: s for s in tools_schema.standard_tools}
        # Stripping a name the schema doesn't have is a no-op...
        self.assertEqual(set(by_name["other"].properties), {"x"})
        # ...but the argument is still injected at call time.
        await client.call_tool("other", {"x": "y"})
        self.assertEqual(session.calls, [("other", {"x": "y", "hidden": 1})])
        await client.close()


class TestLifecycle(MCPClientTestBase):
    """start/close: task safety, retry, idempotency."""

    async def test_close_from_other_task_exits_transport_in_owner_task(self):
        client, session, record = self._make_client([_tool("tool_a")])
        await client.tools()
        # Close from a different task than the one that started the client (as
        # an on_client_disconnected handler would).
        await asyncio.create_task(client.close())
        self.assertEqual(record["exits"], 1)
        # The anyio-based transport requires enter/exit in the same task.
        self.assertIs(record["enter_task"], record["exit_task"])
        self.assertIsNot(record["enter_task"], asyncio.current_task())

    async def test_failed_start_raises_and_allows_retry(self):
        client, session, record = self._make_client([_tool("tool_a")], fail_initializes=1)
        with self.assertRaises(RuntimeError):
            await client.start()
        # The partially opened transport was cleaned up.
        self.assertEqual(record.get("exits"), 1)
        # A later call retries cleanly.
        await client.start()
        self.assertEqual(record["initializes"], 1)
        await client.close()

    async def test_close_before_start_and_double_close_are_safe(self):
        client, session, record = self._make_client([_tool("tool_a")])
        await client.close()
        await client.start()
        await client.close()
        await client.close()
        self.assertEqual(record["exits"], 1)

    async def test_call_tool_after_close_raises(self):
        client, session, record = self._make_client([_tool("tool_a")])
        await client.start()
        await client.close()
        with self.assertRaises(RuntimeError):
            await client.call_tool("tool_a", {})

    async def test_tools_after_close_reconnects(self):
        client, session, record = self._make_client([_tool("tool_a")])
        await client.tools()
        await client.close()
        tools_schema = await client.tools()
        self.assertEqual(record["enters"], 2)
        self.assertEqual({s.name for s in tools_schema.standard_tools}, {"tool_a"})
        await client.close()

    async def test_concurrent_starts_open_one_connection(self):
        client, session, record = self._make_client([_tool("tool_a")])
        await asyncio.gather(client.start(), client.start(), client.start())
        self.assertEqual(record["enters"], 1)
        await client.close()

    async def test_async_with_still_works(self):
        client, session, record = self._make_client([_tool("tool_a")])
        async with client as mcp:
            tools_schema = await mcp.tools()
            self.assertEqual({s.name for s in tools_schema.standard_tools}, {"tool_a"})
        self.assertEqual(record["exits"], 1)


class TestCallTool(MCPClientTestBase):
    """Public call_tool(): text results and output filters."""

    async def test_call_tool_returns_text(self):
        client, session, record = self._make_client([_tool("tool_a")])
        await client.start()
        result = await client.call_tool("tool_a", {"x": "y"})
        self.assertEqual(result, "tool_a-RESULT")
        await client.close()

    async def test_call_tool_applies_output_filter(self):
        client, session, record = self._make_client(
            [_tool("tool_a")], tools_output_filters={"tool_a": lambda text: text.upper()}
        )
        await client.start()
        result = await client.call_tool("tool_a", {})
        self.assertEqual(result, "TOOL_A-RESULT")
        await client.close()

    async def test_call_tool_requires_connection(self):
        client, session, record = self._make_client([_tool("tool_a")])
        with self.assertRaises(RuntimeError):
            await client.call_tool("tool_a", {})


class TestDeprecatedRegistrationApi(MCPClientTestBase):
    """register_tools/register_tools_schema/get_tools_schema are deprecated but work."""

    async def test_get_tools_schema_warns_and_returns_handlerless_schemas(self):
        client, session, record = self._make_client([_tool("tool_a")])
        await client.start()
        with self.assertWarns(DeprecationWarning):
            tools_schema = await client.get_tools_schema()
        self.assertEqual({s.name for s in tools_schema.standard_tools}, {"tool_a"})
        for schema in tools_schema.standard_tools:
            self.assertIsNone(schema.handler)
        await client.close()

    async def test_register_tools_warns_and_registers(self):
        client, session, record = self._make_client([_tool("tool_a"), _tool("tool_b")])
        await client.start()
        llm = MagicMock()
        with self.assertWarns(DeprecationWarning):
            tools_schema = await client.register_tools(llm)
        self.assertEqual(llm.register_function.call_count, 2)
        registered = {call.args[0] for call in llm.register_function.call_args_list}
        self.assertEqual(registered, {"tool_a", "tool_b"})
        for schema in tools_schema.standard_tools:
            self.assertIsNone(schema.handler)
        await client.close()

    async def test_register_tools_schema_warns_and_registers(self):
        client, session, record = self._make_client([_tool("tool_a")])
        await client.start()
        tools_schema = await client.tools()
        llm = MagicMock()
        with self.assertWarns(DeprecationWarning):
            await client.register_tools_schema(tools_schema, llm)
        llm.register_function.assert_called_once()
        await client.close()


class TestAutoCloseOnCleanup(MCPClientTestBase):
    """LLMService.cleanup() closes clients whose handlers were registered."""

    async def test_cleanup_closes_registered_client(self):
        client, session, record = self._make_client([_tool("tool_a")])
        tools_schema = await client.tools()
        service = LLMService()
        service._sync_registered_tool_handlers(tools_schema)
        await service.cleanup()
        self.assertEqual(record.get("exits"), 1)

    async def test_cleanup_twice_is_safe(self):
        client, session, record = self._make_client([_tool("tool_a")])
        service = LLMService()
        service._sync_registered_tool_handlers(await client.tools())
        await service.cleanup()
        await service.cleanup()
        self.assertEqual(record.get("exits"), 1)

    async def test_two_services_sharing_client_close_idempotently(self):
        # e.g. two LLMs behind a switcher advertising the same context tools.
        client, session, record = self._make_client([_tool("tool_a")])
        tools_schema = await client.tools()
        service_a, service_b = LLMService(), LLMService()
        service_a._sync_registered_tool_handlers(tools_schema)
        service_b._sync_registered_tool_handlers(tools_schema)
        await service_a.cleanup()
        await service_b.cleanup()
        self.assertEqual(record.get("exits"), 1)

    async def test_no_close_when_handlers_never_registered(self):
        # Known gap: a connected client the LLM service never learned about
        # (no inference ran) is not auto-closed.
        client, session, record = self._make_client([_tool("tool_a")])
        await client.tools()
        service = LLMService()
        await service.cleanup()
        self.assertIsNone(record.get("exits"))
        await client.close()
        self.assertEqual(record.get("exits"), 1)

    async def test_deprecated_register_tools_path_also_auto_closes(self):
        client, session, record = self._make_client([_tool("tool_a")])
        await client.start()
        service = LLMService()
        with self.assertWarns(DeprecationWarning):
            await client.register_tools(service)
        await service.cleanup()
        self.assertEqual(record.get("exits"), 1)

    async def test_client_survives_tool_pruning_until_cleanup(self):
        # De-advertising a tool prunes its handler but must not close the
        # session mid-conversation; the close happens at teardown.
        client, session, record = self._make_client([_tool("tool_a")])
        service = LLMService()
        service._sync_registered_tool_handlers(await client.tools())
        service._sync_registered_tool_handlers([])  # tool set replaced
        self.assertIsNone(record.get("exits"))
        await service.cleanup()
        self.assertEqual(record.get("exits"), 1)


class TestLLMAutoRegistration(MCPClientTestBase):
    """End-to-end with a real LLMService: tools() auto-registers, old path doesn't warn."""

    async def test_tools_auto_register_with_llm_service(self):
        client, session, record = self._make_client([_tool("tool_a")])
        tools_schema = await client.tools()
        service = LLMService()
        service._sync_registered_tool_handlers(tools_schema)
        self.assertTrue(service.has_function("tool_a"))
        self.assertTrue(service._functions["tool_a"].auto_registered)
        await client.close()

    async def test_deprecated_register_tools_path_does_not_warn_redundant(self):
        client, session, record = self._make_client([_tool("tool_a")])
        await client.start()
        service = LLMService()
        with self.assertWarns(DeprecationWarning):
            tools_schema = await client.register_tools(service)
        # Advertising the handler-less schemas alongside the manual registration
        # must not trip the redundant-manual-registration advisory.
        sink = io.StringIO()
        handler_id = logger.add(sink, level="WARNING", format="{message}")
        try:
            service._sync_registered_tool_handlers(tools_schema)
        finally:
            logger.remove(handler_id)
        self.assertEqual(sink.getvalue(), "")
        self.assertTrue(service.has_function("tool_a"))
        await client.close()


if __name__ == "__main__":
    unittest.main()
