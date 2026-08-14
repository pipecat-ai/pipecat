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

    def __init__(self, record, exit_error=None, connect_delay=0):
        self._record = record
        self._exit_error = exit_error
        self._connect_delay = connect_delay

    async def __aenter__(self):
        self._record["enters"] = self._record.get("enters", 0) + 1
        self._record["enter_task"] = asyncio.current_task()
        if self._connect_delay:
            # A connect window wide enough to cancel the caller inside it.
            await asyncio.sleep(self._connect_delay)
        return (MagicMock(), MagicMock(), MagicMock())

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._record["exits"] = self._record.get("exits", 0) + 1
        self._record["exit_task"] = asyncio.current_task()
        if self._exit_error is not None:
            # A real transport reports a failed connection as its task group
            # unwinds, rather than at the connect site.
            raise self._exit_error
        return False


class _FakeSession:
    """Fake mcp ClientSession with canned tools and call results."""

    def __init__(self, tools, record, fail_initializes=0, cancel_initialize=False):
        self._tools = tools
        self._record = record
        self._fail_initializes = fail_initializes
        self._cancel_initialize = cancel_initialize
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    async def initialize(self):
        if self._cancel_initialize:
            # An anyio transport cancels the connecting task when its own
            # request fails, so initialize() ends in cancellation.
            raise asyncio.CancelledError("Cancelled via cancel scope")
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

    def _make_client(
        self,
        tools,
        fail_initializes=0,
        cancel_initialize=False,
        transport_exit_error=None,
        connect_delay=0,
        **client_kwargs,
    ):
        record = {}
        session = _FakeSession(tools, record, fail_initializes, cancel_initialize)
        ctx = patch.multiple(
            "pipecat.services.mcp_service",
            streamablehttp_client=lambda **kwargs: _FakeTransport(
                record, transport_exit_error, connect_delay
            ),
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

    async def _call_via_handler(self, tools_schema, name, arguments=None):
        """Invoke a tool the way the LLM service does, through its handler.

        Returns:
            The result callback, so callers can assert on what was delivered.
        """
        schema = next(s for s in tools_schema.standard_tools if s.name == name)
        params = SimpleNamespace(
            function_name=name,
            tool_call_id="call-1",
            arguments=arguments,
            result_callback=AsyncMock(),
        )
        await schema.handler(params)
        return params.result_callback


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

    async def test_handler_applies_output_filter(self):
        client, session, record = self._make_client(
            [_tool("tool_a")], tools_output_filters={"tool_a": lambda text: text.upper()}
        )
        tools_schema = await client.tools()
        result_callback = await self._call_via_handler(tools_schema, "tool_a", {"x": "y"})
        result_callback.assert_awaited_once_with("TOOL_A-RESULT")
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
        tools_schema = await client.tools()
        await self._call_via_handler(
            tools_schema, "search", {"query": "news", "mode": "model-supplied"}
        )
        self.assertEqual(session.calls, [("search", {"query": "news", "mode": "realtime"})])
        await client.close()

    async def test_fixed_arguments_injected_when_no_model_arguments(self):
        client, session, record = self._make_client(
            self._search_tools(), tools_arguments={"search": {"mode": "realtime"}}
        )
        tools_schema = await client.tools()
        await self._call_via_handler(tools_schema, "search")
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
        await self._call_via_handler(tools_schema, "other", {"x": "y"})
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

    async def test_cancelling_transport_raises_its_underlying_cause(self):
        # A transport whose request fails cancels the connecting task and reports
        # the cause only as it unwinds; tools() must raise that cause rather than
        # wait on a connection that will never arrive.
        cause = RuntimeError("Client error '401 Unauthorized'")
        client, session, record = self._make_client(
            [_tool("tool_a")],
            cancel_initialize=True,
            transport_exit_error=ExceptionGroup("unhandled errors in a TaskGroup", [cause]),
        )
        with self.assertRaises(RuntimeError) as ctx:
            await asyncio.wait_for(client.tools(), timeout=5)
        self.assertIs(ctx.exception, cause)
        self.assertEqual(record.get("exits"), 1)

    async def test_cancelling_transport_without_a_cause_still_raises(self):
        # Nothing surfaced on unwind, so there's no cause to report — but the
        # caller still gets an error instead of waiting forever.
        client, session, record = self._make_client([_tool("tool_a")], cancel_initialize=True)
        with self.assertRaises(ConnectionError):
            await asyncio.wait_for(client.tools(), timeout=5)

    async def test_cancelling_the_caller_stops_a_connect_in_flight(self):
        # Cancelling the task awaiting start() cancels the future it waits on. A
        # session still inside the connect can't see that, so start() stops it —
        # an unresponsive server would otherwise hold the transport (a spawned
        # server process) with nothing left able to reach it.
        client, session, record = self._make_client([_tool("tool_a")], connect_delay=3600)
        caller = asyncio.create_task(client.tools())
        await asyncio.sleep(0.05)  # inside the connect window
        session_task = client._session_task
        caller.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await caller

        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(session_task, timeout=5)
        self.assertIsNone(client._active_session)
        self.assertIsNone(client._session_task)
        # Nothing is left for close() to do, and it says so quietly.
        await client.close()

    async def test_connection_nobody_awaits_is_released(self):
        # The connect can still land in the window between the caller's
        # cancellation and start() reacting to it, leaving a session with no one
        # to hand it to. It has to release itself rather than sit open.
        client, session, record = self._make_client([_tool("tool_a")])
        ready = asyncio.get_running_loop().create_future()
        ready.cancel()

        await client._run_session(ready, asyncio.Event())

        self.assertEqual(record["exits"], 1)
        self.assertIsNone(client._active_session)

    async def test_failed_connect_nobody_awaits_settles_quietly(self):
        # Same window, but the connect fails: with no one waiting, the failure has
        # nowhere to go and must not become an error of its own.
        client, session, record = self._make_client([_tool("tool_a")], fail_initializes=1)
        ready = asyncio.get_running_loop().create_future()
        ready.cancel()

        await client._run_session(ready, asyncio.Event())

        self.assertEqual(record["exits"], 1)
        self.assertIsNone(client._active_session)

    async def test_close_before_start_and_double_close_are_safe(self):
        client, session, record = self._make_client([_tool("tool_a")])
        await client.close()
        await client.start()
        await client.close()
        await client.close()
        self.assertEqual(record["exits"], 1)

    async def test_calling_a_tool_after_close_raises(self):
        client, session, record = self._make_client([_tool("tool_a")])
        tools_schema = await client.tools()
        await client.close()
        with self.assertRaises(RuntimeError):
            await self._call_via_handler(tools_schema, "tool_a", {})

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
