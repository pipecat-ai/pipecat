#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Free, low-latency web search for Pipecat agents.

Exposes ``search_web`` and ``fetch_page`` function tools that voice agents can
call to look up current information and read specific pages. Drop them into an
:class:`~pipecat.processors.aggregators.llm_context.LLMContext` and the LLM
auto-registers their handlers:

Example::

    from pipecat.services.keenable.search import KeenableWebSearch

    search = KeenableWebSearch()
    context = LLMContext(messages=[...], tools=search.tools())

Search runs against a hosted MCP server powered by `Keenable AI
<https://keenable.ai>`_, which this service connects to lazily on the first
call. No API key is required — keyless requests use ``pro`` mode. Pass
``api_key=`` for higher rate limits and access to the lower-latency
``realtime`` mode (which requires an enabled account); select it with ``mode=``.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Literal

from loguru import logger

from pipecat import version as pipecat_version
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.utils.base_object import BaseObject

if TYPE_CHECKING:
    from pipecat.services.llm_service import FunctionCallParams

try:
    from mcp.client.session_group import StreamableHttpParameters

    from pipecat.services.mcp_service import MCPClient
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        'In order to use Keenable web search, you need to `uv add "pipecat-ai[keenable]"`.'
    )
    raise ImportError(f"Missing module: {e}") from e

#: Search mode: ``"pro"`` (higher quality) or ``"realtime"`` (lower latency).
SearchMode = Literal["pro", "realtime"]

# The MCP tools on Keenable's server that the curated tools map to.
_SERVER_SEARCH_TOOL = "search_web_pages"
_SERVER_FETCH_TOOL = "fetch_page_content"

# Optional ``search_web`` filters, forwarded to the server only when the model
# supplies them.
_SEARCH_FILTER_PARAMS = (
    "site",
    "published_after",
    "published_before",
    "acquired_after",
    "acquired_before",
)


class KeenableWebSearch(BaseObject):
    """Zero-config web search tool for voice agents.

    Provides curated ``search_web`` and ``fetch_page`` function tools backed by a
    hosted MCP server powered by Keenable AI. The tool schemas are local and
    static, so :meth:`tools` is synchronous and can be used to build an
    ``LLMContext`` without any setup — the MCP connection is opened lazily on the
    first call.

    Each schema carries its own handler, so the LLM registers them automatically
    when the context's tools are advertised.

    Call :meth:`start` to pre-open the connection and :meth:`close` to release it
    (e.g. on client disconnect).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        mode: SearchMode | None = None,
        **kwargs,
    ) -> None:
        """Initialize the web search tool.

        Args:
            api_key: API key for higher rate limits and ``realtime`` access.
                When unset, the keyless free tier is used.
            mode: Search mode — ``"pro"`` (higher quality) or ``"realtime"``
                (lower latency, good for voice; requires an enabled account).
                When unset, defaults to ``"realtime"`` if an API key is present,
                else ``"pro"``.
            **kwargs: Additional arguments passed to ``BaseObject``.
        """
        super().__init__(**kwargs)

        self._api_key = api_key
        # realtime requires an enabled key, so default to it only when keyed;
        # keyless falls back to pro.
        self._mode: SearchMode = mode or ("realtime" if api_key else "pro")

        # The MCP session is anyio-task-bound: it must be opened and closed in the
        # same task. Tool calls and close() arrive on different tasks, so a single
        # owner task (``_run_session``) holds the session open and tears it down on
        # signal; everyone else just talks to ``self._mcp``.
        self._mcp: MCPClient | None = None
        self._session_task: asyncio.Task | None = None
        self._closing: asyncio.Event | None = None
        self._start_lock = asyncio.Lock()

    def tools(self) -> ToolsSchema:
        """Return the ``search_web`` and ``fetch_page`` tool schemas.

        Each schema carries its handler, so the LLM auto-registers both when the
        context advertises its tools.

        Returns:
            A ToolsSchema containing the ``search_web`` and ``fetch_page`` tools.
        """
        search_web = FunctionSchema(
            name="search_web",
            description=(
                "Search the web for current, real-time information — news, facts, and "
                "anything beyond the model's training data."
            ),
            properties={
                "query": {
                    "type": "string",
                    "description": (
                        "A semantically rich description of the ideal page, not just keywords."
                    ),
                },
                "site": {
                    "type": "string",
                    "description": 'Restrict results to a specific site (e.g. "techcrunch.com").',
                },
                "published_after": {
                    "type": "string",
                    "description": "Only include pages published after this date (YYYY-MM-DD).",
                },
                "published_before": {
                    "type": "string",
                    "description": "Only include pages published before this date (YYYY-MM-DD).",
                },
                "acquired_after": {
                    "type": "string",
                    "description": "Only include pages indexed after this date (YYYY-MM-DD).",
                },
                "acquired_before": {
                    "type": "string",
                    "description": "Only include pages indexed before this date (YYYY-MM-DD).",
                },
            },
            required=["query"],
            handler=self._handle_search,
        )
        fetch_page = FunctionSchema(
            name="fetch_page",
            description=(
                "Fetch the readable text of a specific web page by URL — useful for reading "
                "a result returned by search_web."
            ),
            properties={
                "url": {
                    "type": "string",
                    "description": 'The URL to fetch. Example: "https://example.com".',
                },
                "max_chars": {
                    "type": "integer",
                    "description": (
                        "Maximum number of characters to return; longer content is "
                        "truncated. Defaults to 50000."
                    ),
                },
                "live": {
                    "type": "boolean",
                    "description": "Fetch live content instead of the indexed copy. Defaults to false.",
                },
            },
            required=["url"],
            handler=self._handle_fetch,
        )
        return ToolsSchema(standard_tools=[search_web, fetch_page])

    async def _handle_search(self, params: FunctionCallParams) -> None:
        """Run a ``search_web`` call against Keenable's MCP server.

        Opens the MCP connection on first use, calls the underlying search tool
        with the configured ``pro``/``realtime`` mode injected (and any optional
        filters the model supplied), and returns the result to the model via the
        function-call result callback.
        """
        mcp = await self.start()
        arguments = params.arguments or {}
        # ``mode`` selects pro/realtime per call; the server reads it per request.
        call_args = {"query": arguments.get("query", ""), "mode": self._mode}
        # Forward optional filters only when the model supplied them.
        for key in _SEARCH_FILTER_PARAMS:
            if arguments.get(key) is not None:
                call_args[key] = arguments[key]
        result = await mcp.call_tool(_SERVER_SEARCH_TOOL, call_args)
        await params.result_callback(result)

    async def _handle_fetch(self, params: FunctionCallParams) -> None:
        """Run a ``fetch_page`` call against Keenable's MCP server.

        Opens the MCP connection on first use, fetches the page text (passing the
        optional ``max_chars`` / ``live`` arguments when supplied), and returns
        the result to the model via the function-call result callback.
        """
        mcp = await self.start()
        arguments = params.arguments or {}
        call_args = {"url": arguments.get("url", "")}
        for key in ("max_chars", "live"):
            if arguments.get(key) is not None:
                call_args[key] = arguments[key]
        result = await mcp.call_tool(_SERVER_FETCH_TOOL, call_args)
        await params.result_callback(result)

    async def start(self) -> MCPClient:
        """Open the MCP connection to Keenable's server.

        Idempotent and called automatically on the first tool call; call it
        directly only to connect ahead of time.

        Returns:
            The connected MCP client.
        """
        if self._mcp is not None:
            return self._mcp

        async with self._start_lock:
            if self._mcp is not None:
                return self._mcp
            self._closing = asyncio.Event()
            ready: asyncio.Future = asyncio.get_running_loop().create_future()
            self._session_task = asyncio.create_task(
                self._run_session(ready, self._closing), name="keenable-mcp-session"
            )
            try:
                await ready  # resolves once connected; re-raises a connect failure
            except Exception:
                self._session_task = None  # let a later call retry cleanly
                raise
            assert self._mcp is not None
            return self._mcp

    async def _run_session(self, ready: asyncio.Future, closing: asyncio.Event) -> None:
        """Own the MCP session for its whole lifetime, in a single task.

        Opens the connection, signals ``ready``, then holds it open until
        :meth:`close` sets the closing event — so open and close happen in the
        same task, as the anyio-based MCP session requires.
        """
        mcp = MCPClient(
            server_params=StreamableHttpParameters(
                url="https://api.keenable.ai/mcp",
                headers=self._build_headers(),
            ),
        )
        try:
            await mcp.start()
        except Exception as e:
            ready.set_exception(e)
            return

        self._mcp = mcp
        logger.info(f"KeenableWebSearch: connected (mode={self._mode})")
        ready.set_result(None)
        try:
            await closing.wait()
        finally:
            await mcp.close()
            self._mcp = None

    async def close(self) -> None:
        """Close the MCP connection. Safe to call multiple times and from any task."""
        if self._session_task is None:
            return
        task, self._session_task = self._session_task, None
        if self._closing is not None:
            self._closing.set()
        await task

    def _build_headers(self) -> dict[str, str]:
        headers = {"User-Agent": f"pipecat/{pipecat_version()}"}
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        return headers
