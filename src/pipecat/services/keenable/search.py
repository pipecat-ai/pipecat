#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Free, low-latency web search for pipecat agents via remote MCP.

Thin wrapper around :class:`~pipecat.services.mcp_service.MCPClient` that
connects to a hosted MCP server powered by `Keenable AI <https://keenable.ai>`_.
Tool schemas and implementations live server-side, so updates ship instantly
without a pipecat release.

No API key is required — keyless requests use ``pro`` mode. Set
``KEENABLE_API_KEY`` (or pass ``api_key=``) for higher rate limits and the
lower-latency ``realtime`` mode (which requires an enabled account).

Example::

    from pipecat.services.keenable.search import KeenableWebSearch

    async with KeenableWebSearch() as search:
        tools = await search.register_tools(llm)
        context = LLMContext(tools=tools, ...)
"""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from pipecat.adapters.schemas.tools_schema import ToolsSchema
    from pipecat.pipeline.llm_switcher import LLMSwitcher
    from pipecat.services.llm_service import FunctionCallParams, LLMService

_MCP_URL = "https://api.keenable.ai/mcp"
API_KEY_ENV_VAR = "KEENABLE_API_KEY"
SEARCH_MODE_ENV_VAR = "KEENABLE_SEARCH_MODE"
VALID_SEARCH_MODES = ("pro", "realtime")

# Tools that accept a search ``mode`` argument. The server selects pro/realtime
# per call, so the configured mode is injected into these calls' arguments.
MODE_AWARE_TOOLS = ("search_web_pages",)


def _pipecat_version() -> str:
    """Best-effort pipecat version for the User-Agent header."""
    try:
        return version("pipecat-ai")
    except PackageNotFoundError:
        return "unknown"


def _import_mcp_deps():
    """Lazy-import MCP dependencies. Raises with a clear message if missing."""
    try:
        from mcp.client.session_group import StreamableHttpParameters

        from pipecat.services.mcp_service import MCPClient
    except ModuleNotFoundError as e:
        raise ImportError(
            "Web search requires the 'mcp' package. Install it with: pip install 'mcp>=1.11.0'"
        ) from e
    return MCPClient, StreamableHttpParameters


class KeenableWebSearch:
    """Zero-config web search for voice agents.

    Connects to a hosted MCP server powered by Keenable AI (Streamable HTTP
    transport) and registers search tools on any pipecat ``LLMService``. Tool
    schemas are served dynamically, so API updates propagate automatically.

    Can be used as an async context manager::

        async with KeenableWebSearch() as search:
            tools = await search.register_tools(llm)

    Or managed manually::

        search = KeenableWebSearch()
        await search.start()
        tools = await search.register_tools(llm)
        ...
        await search.close()
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        mode: str | None = None,
    ) -> None:
        """Initialize the web search wrapper.

        Args:
            api_key: API key for higher rate limits and ``realtime`` access.
                Falls back to the ``KEENABLE_API_KEY`` env var. When unset, the
                keyless free tier is used.
            mode: Search mode — ``"pro"`` (higher quality) or ``"realtime"``
                (lower latency, good for voice; requires an enabled account).
                When unset, defaults to ``"realtime"`` if an API key is present,
                else ``"pro"``. The ``KEENABLE_SEARCH_MODE`` env var overrides
                this.
        """
        if api_key is None:
            api_key = (os.environ.get(API_KEY_ENV_VAR) or "").strip() or None
        self._api_key = api_key

        # realtime requires an enabled key, so default to it only when keyed;
        # keyless falls back to pro.
        if mode is None:
            mode = "realtime" if api_key else "pro"

        env_mode = (os.environ.get(SEARCH_MODE_ENV_VAR) or "").strip().lower()
        if env_mode:
            if env_mode not in VALID_SEARCH_MODES:
                logger.warning(
                    f"KeenableWebSearch: ignoring invalid {SEARCH_MODE_ENV_VAR}={env_mode!r}, "
                    f"expected one of {VALID_SEARCH_MODES}. Using {mode!r}."
                )
            else:
                mode = env_mode

        self._mode = mode
        self._mcp: Any = None  # MCPClient instance (lazy import)

    def _build_headers(self) -> dict[str, str]:
        headers = {"User-Agent": f"pipecat/{_pipecat_version()}"}
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        return headers

    async def start(self) -> None:
        """Open the MCP connection to Keenable's server.

        Raises:
            ImportError: If ``pipecat-ai[keenable]`` is not installed.
        """
        if self._mcp is not None:
            return

        MCPClient, StreamableHttpParameters = _import_mcp_deps()

        self._mcp = MCPClient(
            server_params=StreamableHttpParameters(
                url=_MCP_URL,
                headers=self._build_headers(),
            ),
        )
        await self._mcp.start()
        logger.info(f"KeenableWebSearch: connected to {_MCP_URL} (mode={self._mode})")

    async def close(self) -> None:
        """Close the MCP connection. Safe to call multiple times."""
        if self._mcp is not None:
            await self._mcp.close()
            self._mcp = None

    async def __aenter__(self) -> KeenableWebSearch:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def register_tools(self, llm: LLMService | LLMSwitcher) -> ToolsSchema:
        """Discover tools from the MCP server and register them on the LLM.

        Search tools are wrapped so the configured search ``mode`` is sent as a
        call argument — the server picks ``pro``/``realtime`` per call, not per
        connection.

        Args:
            llm: The pipecat LLM service to register tools with.

        Returns:
            A ToolsSchema containing the registered tools.
        """
        self._ensure_connected()
        tools_schema = await self._mcp.get_tools_schema()
        for function_schema in tools_schema.standard_tools:
            if function_schema.name in MODE_AWARE_TOOLS:
                llm.register_function(function_schema.name, self._mode_injecting_handler)
            else:
                llm.register_function(function_schema.name, self._mcp._tool_wrapper)
        return tools_schema

    async def _mode_injecting_handler(self, params: FunctionCallParams) -> None:
        """Inject the configured search mode into the call, then delegate to MCP."""
        params.arguments = {**(params.arguments or {}), "mode": self._mode}
        await self._mcp._tool_wrapper(params)

    async def get_tools_schema(self) -> ToolsSchema:
        """Get tool schemas from the MCP server without registering them.

        Returns:
            A ToolsSchema containing the available tools.
        """
        self._ensure_connected()
        return await self._mcp.get_tools_schema()

    def _ensure_connected(self) -> None:
        if self._mcp is None:
            raise RuntimeError(
                "KeenableWebSearch is not connected. Use 'async with KeenableWebSearch()' "
                "or call 'await search.start()' first."
            )
