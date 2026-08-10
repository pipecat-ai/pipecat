#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A Pipecat service that drives a coding agent over the Agent Client Protocol.

The service is a protocol bridge and nothing more. It turns
:class:`~pipecat.services.acp.frames.ACPPromptFrame` into ``session/prompt``,
turns everything the agent streams back into ACP frames, and hands the agent's
callbacks to whichever processor is willing to answer them. It emits no text a
TTS service would speak; converting the agent's output into something worth
saying is a downstream concern.
"""

import asyncio
from collections.abc import Callable
from typing import Any

from loguru import logger

from pipecat.frames.frames import CancelFrame, EndFrame, Frame, InterruptionFrame, StartFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.acp.client import ACPClient, ACPError
from pipecat.services.acp.frames import (
    ACPAgentMessageFrame,
    ACPAgentThoughtFrame,
    ACPCancelTurnFrame,
    ACPClientRequestFrame,
    ACPClientResponseFrame,
    ACPCommandsUpdatedFrame,
    ACPModeUpdatedFrame,
    ACPPermissionRequestFrame,
    ACPPlanFrame,
    ACPPromptFrame,
    ACPReadFileRequestFrame,
    ACPSessionEndedFrame,
    ACPSessionStartedFrame,
    ACPSetModeFrame,
    ACPTerminalRequestFrame,
    ACPToolCallFrame,
    ACPToolCallUpdateFrame,
    ACPTurnEndedFrame,
    ACPTurnStartedFrame,
    ACPUserMessageFrame,
    ACPWriteFileRequestFrame,
)
from pipecat.services.acp.types import (
    AvailableCommand,
    ClientCapabilities,
    ContentBlock,
    MCPServer,
    PlanEntry,
    ReadTextFileParams,
    RequestPermissionParams,
    StopReason,
    TerminalParams,
    ToolCall,
    ToolCallUpdate,
    WriteTextFileParams,
)
from pipecat.services.ai_service import AIService
from pipecat.services.settings import ServiceSettings


def _content_frame(frame_cls):
    """Build a frame carrying a single content block."""
    return lambda service, session_id, params: frame_cls(
        session_id=session_id, content=ContentBlock.model_validate(params["content"])
    )


# Each ``sessionUpdate`` variant and the frame it becomes. Builders take the
# service so the tool-call ones can read and update its merged view.
SESSION_UPDATE_FRAMES: dict[str, Callable[["ACPService", str, dict], Frame]] = {
    "agent_message_chunk": _content_frame(ACPAgentMessageFrame),
    "agent_thought_chunk": _content_frame(ACPAgentThoughtFrame),
    "user_message_chunk": _content_frame(ACPUserMessageFrame),
    "tool_call": lambda service, session_id, params: service._build_tool_call_frame(
        session_id, params
    ),
    "tool_call_update": lambda service, session_id, params: service._build_tool_call_update_frame(
        session_id, params
    ),
    "plan": lambda service, session_id, params: ACPPlanFrame(
        session_id=session_id,
        entries=[PlanEntry.model_validate(e) for e in params.get("entries", [])],
    ),
    "current_mode_update": lambda service, session_id, params: ACPModeUpdatedFrame(
        session_id=session_id, current_mode_id=params.get("currentModeId")
    ),
    "available_commands_update": lambda service, session_id, params: ACPCommandsUpdatedFrame(
        session_id=session_id,
        commands=[AvailableCommand.model_validate(c) for c in params.get("availableCommands", [])],
    ),
}

# Each agent-initiated method and the frame that carries it into the pipeline.
CLIENT_REQUEST_FRAMES: dict[str, tuple[type[ACPClientRequestFrame], type]] = {
    "session/request_permission": (ACPPermissionRequestFrame, RequestPermissionParams),
    "fs/read_text_file": (ACPReadFileRequestFrame, ReadTextFileParams),
    "fs/write_text_file": (ACPWriteFileRequestFrame, WriteTextFileParams),
    "terminal/create": (ACPTerminalRequestFrame, TerminalParams),
    "terminal/output": (ACPTerminalRequestFrame, TerminalParams),
    "terminal/wait_for_exit": (ACPTerminalRequestFrame, TerminalParams),
    "terminal/kill": (ACPTerminalRequestFrame, TerminalParams),
    "terminal/release": (ACPTerminalRequestFrame, TerminalParams),
}


class ACPService(AIService):
    """Runs a coding agent as a subprocess and bridges it to the pipeline.

    The agent owns the conversation history, so the service holds no context and
    sends only the new user turn on each prompt. What it does hold is the state
    the protocol requires a client to keep: the session id, the in-flight turn,
    a merged view of every tool call so partial updates can be rendered, and the
    futures for agent callbacks awaiting an answer.

    Prompts are serialized. ACP allows one turn at a time per session, so a
    prompt that arrives while the agent is working is queued rather than
    dropped.

    Event handlers available:

    - on_session_started: Called with the ``ACPSessionStartedFrame``
    - on_turn_started: Called with the session id when a prompt is sent
    - on_turn_ended: Called with the session id and the turn's ``StopReason``
    - on_agent_exited: Called with the agent's return code when it dies

    Example::

        acp = ACPService(
            command=["npx", "@zed-industries/claude-code-acp"],
            cwd="/path/to/repo",
        )
    """

    def __init__(
        self,
        *,
        command: list[str],
        cwd: str,
        env: dict[str, str] | None = None,
        mcp_servers: list[MCPServer] | None = None,
        client_capabilities: ClientCapabilities | None = None,
        cancel_turn_on_interruption: bool = False,
        request_timeout: float = 120.0,
        **kwargs,
    ):
        """Initialize the service.

        Args:
            command: The agent executable and its arguments.
            cwd: Working directory the agent should operate in.
            env: Extra environment variables for the agent process.
            mcp_servers: MCP servers the agent should connect to.
            client_capabilities: What this client will do on the agent's behalf.
                Defaults to nothing, which makes agents fall back to touching
                the filesystem and spawning processes themselves.
            cancel_turn_on_interruption: Whether a user interruption cancels the
                agent's turn. Off by default: barge-in during a multi-file edit
                would leave the working tree half-changed.
            request_timeout: Seconds to wait for a processor to answer an agent
                callback before returning an error to the agent.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(settings=ServiceSettings(model=None), **kwargs)

        self._command = command
        self._cwd = cwd
        self._env = env
        self._mcp_servers = mcp_servers or []
        self._client_capabilities = client_capabilities or ClientCapabilities()
        self._cancel_turn_on_interruption = cancel_turn_on_interruption
        self._request_timeout = request_timeout

        self._client = ACPClient(task_factory=self.create_task)
        self._client.on_session_update = self._on_session_update
        self._client.on_client_request = self._on_client_request
        self._client.on_closed = self._on_agent_exited

        self._session_id: str | None = None
        self._session_started_frame: ACPSessionStartedFrame | None = None
        self._tool_calls: dict[str, ToolCall] = {}
        self._pending_requests: dict[str, asyncio.Future] = {}
        self._prompts: asyncio.Queue[list[ContentBlock]] = asyncio.Queue()
        self._turn_task: asyncio.Task | None = None
        self._session_closed = False

        self._register_event_handler("on_session_started")
        self._register_event_handler("on_turn_started")
        self._register_event_handler("on_turn_ended")
        self._register_event_handler("on_agent_exited")

    @property
    def session_id(self) -> str | None:
        """The current ACP session, or None before the session opens."""
        return self._session_id

    #
    # Lifecycle
    #

    async def start(self, frame: StartFrame):
        """Spawn the agent, negotiate capabilities, and open a session.

        Args:
            frame: The start frame.
        """
        await super().start(frame)

        try:
            await self._client.start(self._command, self._cwd, self._env)
            init = await self._client.initialize(self._client_capabilities)
            session = await self._client.new_session(self._cwd, self._mcp_servers)
        except Exception as e:
            # AIService._start only logs what start() raises, which would leave
            # the pipeline running against an agent that never came up.
            await self.push_error(f"Could not start ACP agent {self._command}", e, fatal=True)
            return

        self._session_id = session.session_id

        modes = session.modes
        # Held rather than pushed: start() runs before this processor forwards
        # the StartFrame, and nothing may precede a StartFrame downstream.
        self._session_started_frame = ACPSessionStartedFrame(
            session_id=session.session_id,
            agent_capabilities=init.agent_capabilities,
            current_mode_id=modes.current_mode_id if modes else None,
            available_modes=modes.available_modes if modes else [],
        )
        self._turn_task = self.create_task(self._turn_loop(), "acp-turns")

    async def stop(self, frame: EndFrame):
        """Close the session and terminate the agent.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._shutdown("ended")

    async def cancel(self, frame: CancelFrame):
        """Terminate the agent immediately.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._shutdown("cancelled")

    async def _shutdown(self, reason: str):
        if self._turn_task:
            await self.cancel_task(self._turn_task)
            self._turn_task = None
        await self._close_session(reason)
        await self._client.stop()

    async def _close_session(self, reason: str):
        """Announce the session's end, at most once."""
        if self._session_closed or not self._session_id:
            return
        self._session_closed = True
        await self.push_frame(ACPSessionEndedFrame(session_id=self._session_id, reason=reason))
        self._session_id = None

    async def _on_agent_exited(self, returncode: int | None):
        """Handle the agent process going away on its own.

        Reached when the agent crashes or exits mid-session. The client has
        already failed everything waiting on it, so the remaining job is to tell
        the pipeline the session is over and that it cannot recover.

        An agent that dies before the session opens is reported by ``start()``
        instead, which has the failing call to describe.
        """
        if not self._session_id:
            return

        logger.error(f"{self}: ACP agent exited with code {returncode}")
        await self._close_session(f"agent exited ({returncode})")
        await self._call_event_handler("on_agent_exited", returncode)
        await self.push_error(f"ACP agent exited with code {returncode}", fatal=True)

    #
    # Frame processing
    #

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            if self._cancel_turn_on_interruption:
                await self._cancel_turn()
            await self.push_frame(frame, direction)
        elif isinstance(frame, ACPPromptFrame):
            await self._prompts.put(frame.blocks)
        elif isinstance(frame, ACPClientResponseFrame):
            self._resolve_request(frame)
        elif isinstance(frame, ACPCancelTurnFrame):
            await self._cancel_turn()
        elif isinstance(frame, ACPSetModeFrame):
            if self._session_id:
                await self._client.set_mode(self._session_id, frame.mode_id)
        else:
            await self.push_frame(frame, direction)

        if isinstance(frame, StartFrame) and self._session_started_frame:
            started = self._session_started_frame
            self._session_started_frame = None
            await self.push_frame(started)
            await self._call_event_handler("on_session_started", started)

    async def _cancel_turn(self):
        if self._session_id:
            await self._client.cancel(self._session_id)

    #
    # Turns
    #

    async def _turn_loop(self):
        """Send queued prompts one at a time, since a session runs one turn."""
        while True:
            blocks = await self._prompts.get()
            # Captured up front: the agent can die mid-turn, which clears
            # _session_id before the turn is done being reported.
            session_id = self._session_id
            if not session_id:
                continue
            await self.push_frame(ACPTurnStartedFrame(session_id=session_id))
            await self._call_event_handler("on_turn_started", session_id)
            try:
                stop_reason = await self._client.prompt(session_id, blocks)
            except ACPError as e:
                await self.push_error(f"ACP prompt failed: {e.message}", e)
                stop_reason = StopReason.REFUSAL
            await self.push_frame(ACPTurnEndedFrame(session_id=session_id, stop_reason=stop_reason))
            await self._call_event_handler("on_turn_ended", session_id, stop_reason)

    #
    # Agent to pipeline
    #

    async def _on_session_update(self, params: dict[str, Any]):
        """Turn a ``session/update`` notification into a frame."""
        session_id = params.get("sessionId", self._session_id or "")
        kind = params.get("sessionUpdate")

        builder = SESSION_UPDATE_FRAMES.get(kind)
        if not builder:
            logger.debug(f"{self}: ignoring session update {kind}")
            return

        await self.push_frame(builder(self, session_id, params))

    def _build_tool_call_frame(self, session_id: str, params: dict) -> ACPToolCallFrame:
        tool_call = ToolCall.model_validate(params)
        self._tool_calls[tool_call.tool_call_id] = tool_call
        return ACPToolCallFrame(session_id=session_id, tool_call=tool_call)

    def _build_tool_call_update_frame(
        self, session_id: str, params: dict
    ) -> ACPToolCallUpdateFrame:
        update = ToolCallUpdate.model_validate(params)
        return ACPToolCallUpdateFrame(
            session_id=session_id, update=update, tool_call=self._merge_tool_call(update)
        )

    def _merge_tool_call(self, update: ToolCallUpdate) -> ToolCall:
        """Apply an update to the tracked call and return the merged result.

        Updates carry only what changed, so the title announced with the
        original call is the only place a renderer can learn what finished.
        """
        call = self._tool_calls.get(update.tool_call_id) or ToolCall(
            tool_call_id=update.tool_call_id
        )
        changes = update.model_dump(exclude_none=True, exclude={"tool_call_id", "content"})
        merged = call.model_copy(update=changes)
        if update.content:
            merged.content = call.content + update.content
        self._tool_calls[update.tool_call_id] = merged
        return merged

    #
    # Agent callbacks
    #

    async def _on_client_request(
        self, request_id: str, method: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Ask the pipeline to answer an agent-initiated request."""
        entry = CLIENT_REQUEST_FRAMES.get(method)
        if not entry:
            raise ACPError(-32601, f"Method not supported: {method}")
        frame_cls, params_cls = entry

        future: asyncio.Future = self.get_event_loop().create_future()
        self._pending_requests[request_id] = future

        # Broadcast: an answering processor may sit on either side of this
        # service. A permission answer derived from user speech, for instance,
        # arrives from upstream.
        await self.broadcast_frame_instance(
            frame_cls(
                request_id=request_id,
                method=method,
                params=params_cls.model_validate(params),
            )
        )

        try:
            response: ACPClientResponseFrame = await asyncio.wait_for(
                future, timeout=self._request_timeout
            )
        except TimeoutError:
            raise ACPError(-32000, f"No processor answered {method} in time")
        finally:
            self._pending_requests.pop(request_id, None)

        if response.error:
            raise ACPError(response.error.code, response.error.message, response.error.data)
        return response.result or {}

    def _resolve_request(self, frame: ACPClientResponseFrame):
        """Unblock the agent callback this response answers."""
        future = self._pending_requests.get(frame.request_id)
        if not future:
            logger.debug(f"{self}: no request pending for {frame.request_id}")
        elif not future.done():
            # The request was broadcast, so a response may arrive twice; the
            # second one lands here after the future is already resolved.
            future.set_result(frame)
