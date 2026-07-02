#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Free, low-latency web search for Pipecat agents.

Exposes the tools of Keenable's hosted MCP server — ``search_web_pages`` and
``fetch_page_content`` — so voice agents can look up current information and
read specific pages. Drop them into an
:class:`~pipecat.processors.aggregators.llm_context.LLMContext` and the LLM
auto-registers their handlers:

Example::

    from pipecat.services.keenable.search import KeenableWebSearch

    search = KeenableWebSearch()
    context = LLMContext(messages=[...], tools=await search.tools())

Search runs against a hosted MCP server powered by `Keenable AI
<https://keenable.ai>`_. No API key is required — keyless requests use ``pro``
mode. Pass ``api_key=`` for higher rate limits and access to the lower-latency
``realtime`` mode (which requires an enabled account); select it with ``mode=``.
"""

from typing import Literal

from loguru import logger

from pipecat import version as pipecat_version
from pipecat.adapters.schemas.tools_schema import ToolsSchema

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

_SERVER_URL = "https://api.keenable.ai/mcp"

# The tools on Keenable's MCP server that this wrapper exposes.
_SERVER_SEARCH_TOOL = "search_web_pages"
_SERVER_FETCH_TOOL = "fetch_page_content"


class KeenableWebSearch:
    """Zero-config web search tool for voice agents.

    A thin configuration wrapper around :class:`~pipecat.services.mcp_service.MCPClient`
    for Keenable's hosted MCP server. :meth:`tools` connects to the server and
    returns its ``search_web_pages`` and ``fetch_page_content`` tool schemas with
    handlers attached, so the LLM registers them automatically when the context's
    tools are advertised.

    The configured ``pro``/``realtime`` mode is pinned per call and hidden from
    the model. The connection is closed automatically at pipeline teardown;
    call :meth:`close` only to release it earlier (e.g. on client disconnect).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        mode: SearchMode | None = None,
    ) -> None:
        """Initialize the web search tool.

        Args:
            api_key: API key for higher rate limits and ``realtime`` access.
                When unset, the keyless free tier is used.
            mode: Search mode — ``"pro"`` (higher quality) or ``"realtime"``
                (lower latency, good for voice; requires an enabled account).
                When unset, defaults to ``"realtime"`` if an API key is present,
                else ``"pro"``.
        """
        self._api_key = api_key
        # realtime requires an enabled key, so default to it only when keyed;
        # keyless falls back to pro.
        self._mode: SearchMode = mode or ("realtime" if api_key else "pro")
        self._mcp = MCPClient(
            server_params=StreamableHttpParameters(
                url=_SERVER_URL,
                headers=self._build_headers(),
            ),
            tools_filter=[_SERVER_SEARCH_TOOL, _SERVER_FETCH_TOOL],
            tools_arguments={_SERVER_SEARCH_TOOL: {"mode": self._mode}},
        )

    async def tools(self) -> ToolsSchema:
        """Get the search and page-fetch tool schemas, handlers attached.

        Connects to Keenable's server on first use. Pass the result to an
        ``LLMContext`` and the LLM auto-registers the handlers.

        Returns:
            A ToolsSchema containing the server's ``search_web_pages`` and
            ``fetch_page_content`` tools.
        """
        return await self._mcp.tools()

    async def start(self) -> None:
        """Open the connection to Keenable's server.

        Idempotent, and called automatically by :meth:`tools`; call it directly
        only to connect ahead of time.
        """
        await self._mcp.start()

    async def close(self) -> None:
        """Close the connection.

        Called automatically at pipeline teardown; call it directly only to
        release the connection earlier. Safe to call multiple times and from
        any task.
        """
        await self._mcp.close()

    async def __aenter__(self) -> "KeenableWebSearch":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    def _build_headers(self) -> dict[str, str]:
        headers = {"User-Agent": f"pipecat/{pipecat_version()}"}
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        return headers
