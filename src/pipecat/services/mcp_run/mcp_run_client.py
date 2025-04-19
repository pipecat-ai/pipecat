import os
from typing import Any, Dict, Optional

from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.mcp_client_service import MCPClient

## not needed with SSE, but we might expose other ways to connect in future
# try:
#     os.getenv("MCP_RUN_SESSION_ID") != None
# except ModuleNotFoundError as e:
#     logger.error(f"Exception: {e}")
#     logger.error(
#         "In order to use mcp.run, you need to set MCP_RUN_SESSION_ID - `npx --yes -p @dylibso/mcpx gen-session`."
#     )
#     raise Exception(f"Missing module: {e}")


class MCPRunClient(MCPClient):
    def __init__(
        self,
        api_key: str = os.getenv("MCP_RUN_SESSION_ID"),  # unnec with mcp.run see endpoint
        # https://docs.mcp.run/integrating/tutorials/mcp-run-sse-openai-agents/
        sse_server_url: Optional[str] = "https://www.mcp.run/api/mcp/sse?...",
        **kwargs,
    ):
        super().__init__(server_url=sse_server_url, **kwargs)
        self._mcp_run_session_id = api_key
