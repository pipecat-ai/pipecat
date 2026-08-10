#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A JSON-RPC client for the Agent Client Protocol.

This module has no Pipecat dependencies so it can be driven from a plain script
while iterating on protocol behavior.

ACP is bidirectional: the client calls the agent to start sessions and send
prompts, and the agent calls back into the client to request permission, read
and write files, and run terminal commands. Both directions run over
newline-delimited JSON on the agent subprocess's stdin and stdout.
"""

import asyncio
import json
import os
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any

from loguru import logger

from pipecat.services.acp.types import (
    PROTOCOL_VERSION,
    ClientCapabilities,
    ContentBlock,
    InitializeResult,
    MCPServer,
    NewSessionResult,
    StopReason,
)

TaskFactory = Callable[[Coroutine, str], asyncio.Task]
"""Creates a tracked task. Defaults to :func:`asyncio.create_task`."""

SessionUpdateHandler = Callable[[dict[str, Any]], Awaitable[None]]
"""Receives the params of a ``session/update`` notification."""

ClientRequestHandler = Callable[[str, str, dict[str, Any]], Awaitable[dict[str, Any]]]
"""Answers an agent-initiated request, given its id, method, and params."""

ClosedHandler = Callable[[int | None], Awaitable[None]]
"""Notified when the agent exits on its own, with its return code."""


class ACPError(Exception):
    """A JSON-RPC error returned by the agent, or raised to become one.

    Parameters:
        code: JSON-RPC error code.
        message: Human-readable description.
        data: Additional error detail.
    """

    def __init__(self, code: int, message: str, data: Any | None = None):
        """Initialize the error.

        Args:
            code: JSON-RPC error code.
            message: Human-readable description.
            data: Additional error detail.
        """
        super().__init__(f"ACP error {code}: {message}")
        self.code = code
        self.message = message
        self.data = data


class ACPClient:
    """Speaks ACP to an agent subprocess.

    The client owns the subprocess, correlates requests with responses, and
    dispatches everything the agent initiates to the handlers assigned to
    :attr:`on_session_update` and :attr:`on_client_request`.

    Example::

        client = ACPClient()
        client.on_session_update = handle_update
        await client.start(["npx", "@zed-industries/claude-code-acp"], cwd="/repo")
        await client.initialize(ClientCapabilities())
        session_id = await client.new_session("/repo")
        await client.prompt(session_id, [text_block("what does worker.py do?")])
    """

    def __init__(
        self,
        *,
        request_timeout: float = 30.0,
        task_factory: TaskFactory | None = None,
    ):
        """Initialize the client.

        Args:
            request_timeout: Seconds to wait for the agent to answer a request.
                Does not apply to ``session/prompt``, which is unbounded.
            task_factory: Creates the read loop and per-request tasks. Pass a
                processor's ``create_task`` so the pipeline's task manager
                tracks them.
        """
        self._request_timeout = request_timeout
        self._task_factory: TaskFactory = task_factory or (
            lambda coro, name: asyncio.create_task(coro, name=name)
        )

        self._proc: asyncio.subprocess.Process | None = None
        self._read_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._write_lock = asyncio.Lock()
        self._next_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._stopping = False

        self.on_session_update: SessionUpdateHandler | None = None
        self.on_client_request: ClientRequestHandler | None = None
        self.on_closed: ClosedHandler | None = None

    #
    # Lifecycle
    #

    async def start(self, command: list[str], cwd: str, env: dict[str, str] | None = None):
        """Spawn the agent and begin reading its output.

        Args:
            command: The agent executable and its arguments.
            cwd: Working directory for the agent process.
            env: Extra environment variables, merged over the current
                environment.
        """
        self._stopping = False
        self._proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            env={**os.environ, **(env or {})},
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._read_task = self._task_factory(self._read_loop(), "acp-read")
        self._stderr_task = self._task_factory(self._stderr_loop(), "acp-stderr")
        logger.debug(f"ACP agent started: {' '.join(command)} (pid {self._proc.pid})")

    async def stop(self):
        """Terminate the agent and fail any in-flight requests."""
        self._stopping = True
        self._fail_pending("ACP client stopped")

        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except TimeoutError:
                self._proc.kill()
        self._proc = None

    #
    # Client to agent
    #

    async def initialize(self, capabilities: ClientCapabilities) -> InitializeResult:
        """Negotiate the protocol version and exchange capabilities.

        Args:
            capabilities: What this client is willing to do for the agent.

        Returns:
            What the agent supports.
        """
        result = await self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "clientCapabilities": capabilities.to_wire(),
            },
        )
        return InitializeResult.model_validate(result)

    async def authenticate(self, method_id: str):
        """Run one of the agent's authentication methods.

        Args:
            method_id: The method to run, from ``InitializeResult.auth_methods``.
        """
        await self._request("authenticate", {"methodId": method_id})

    async def new_session(
        self, cwd: str, mcp_servers: list[MCPServer] | None = None
    ) -> NewSessionResult:
        """Start a conversation.

        Args:
            cwd: Working directory the agent should operate in.
            mcp_servers: MCP servers the agent should connect to.

        Returns:
            The new session's identifier and initial mode state.
        """
        result = await self._request(
            "session/new",
            {
                "cwd": cwd,
                "mcpServers": [s.to_wire() for s in (mcp_servers or [])],
            },
        )
        return NewSessionResult.model_validate(result)

    async def load_session(
        self, session_id: str, cwd: str, mcp_servers: list[MCPServer] | None = None
    ):
        """Resume a prior session.

        The agent replays the session's history as ``session/update``
        notifications before this call returns.

        Args:
            session_id: The session to resume.
            cwd: Working directory the agent should operate in.
            mcp_servers: MCP servers the agent should connect to.
        """
        await self._request(
            "session/load",
            {
                "sessionId": session_id,
                "cwd": cwd,
                "mcpServers": [s.to_wire() for s in (mcp_servers or [])],
            },
        )

    async def prompt(self, session_id: str, blocks: list[ContentBlock]) -> StopReason:
        """Send a user turn and wait for the agent to finish acting on it.

        A turn can run for minutes and can call back into the client several
        times before it resolves, so this call has no timeout.

        Args:
            session_id: The session to prompt.
            blocks: The user turn's content.

        Returns:
            Why the turn ended.
        """
        result = await self._request(
            "session/prompt",
            {
                "sessionId": session_id,
                "prompt": [b.to_wire() for b in blocks],
            },
            timeout=None,
        )
        return StopReason(result.get("stopReason", StopReason.END_TURN))

    async def cancel(self, session_id: str):
        """Interrupt the in-flight turn.

        The pending :meth:`prompt` still resolves, with
        :attr:`~pipecat.services.acp.types.StopReason.CANCELLED`.

        Args:
            session_id: The session to interrupt.
        """
        await self._notify("session/cancel", {"sessionId": session_id})

    async def set_mode(self, session_id: str, mode_id: str):
        """Switch the agent's operating mode.

        Args:
            session_id: The session to change.
            mode_id: The mode to switch to.
        """
        await self._request("session/set_mode", {"sessionId": session_id, "modeId": mode_id})

    #
    # JSON-RPC plumbing
    #

    async def _request(
        self, method: str, params: dict[str, Any], timeout: float | None = -1.0
    ) -> dict[str, Any]:
        """Send a request and wait for its response."""
        if not self._proc:
            raise ACPError(-32000, "ACP agent is not running")

        self._next_id += 1
        request_id = self._next_id
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future

        await self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})

        # -1.0 means "use the configured default"; None means wait forever.
        wait_for = self._request_timeout if timeout == -1.0 else timeout
        try:
            if wait_for is None:
                return await future
            return await asyncio.wait_for(future, timeout=wait_for)
        finally:
            self._pending.pop(request_id, None)

    async def _notify(self, method: str, params: dict[str, Any]):
        """Send a notification, which expects no response."""
        if not self._proc:
            return
        await self._write({"jsonrpc": "2.0", "method": method, "params": params})

    async def _write(self, message: dict[str, Any]):
        """Write one newline-delimited JSON message to the agent's stdin."""
        if not self._proc or not self._proc.stdin:
            raise ACPError(-32000, "ACP agent is not running")
        data = (json.dumps(message) + "\n").encode()
        async with self._write_lock:
            self._proc.stdin.write(data)
            await self._proc.stdin.drain()

    async def _read_loop(self):
        """Read messages from the agent until its stdout closes."""
        assert self._proc and self._proc.stdout
        while line := await self._proc.stdout.readline():
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"ACP: skipping unparseable line: {line[:200]!r}")
                continue
            try:
                await self._dispatch(message)
            except Exception as e:
                logger.exception(f"ACP: error dispatching message: {e}")

        # stdout closed: the agent is gone. Anything still waiting on it never
        # gets an answer, so fail it here rather than letting each caller sit
        # until its own timeout.
        returncode = await self._proc.wait() if self._proc else None
        self._fail_pending(f"ACP agent exited ({returncode})")
        if not self._stopping and self.on_closed:
            await self.on_closed(returncode)

    def _fail_pending(self, reason: str):
        """Fail every request still waiting on the agent."""
        for future in self._pending.values():
            if not future.done():
                future.set_exception(ACPError(-32000, reason))
        self._pending.clear()

    async def _stderr_loop(self):
        """Surface the agent's stderr, which is where agents log."""
        assert self._proc and self._proc.stderr
        while line := await self._proc.stderr.readline():
            logger.debug(f"ACP agent: {line.decode(errors='replace').rstrip()}")

    async def _dispatch(self, message: dict[str, Any]):
        """Route one inbound message to a pending future or a handler."""
        if "method" not in message:
            self._resolve(message)
        elif "id" in message:
            # An agent-initiated request. Handled in its own task: a permission
            # request blocks until a human answers, and the read loop has to
            # keep running so the agent's other traffic still arrives.
            self._task_factory(self._serve(message), f"acp-serve-{message['id']}")
        elif self.on_session_update and message["method"] == "session/update":
            await self.on_session_update(message.get("params", {}))
        else:
            logger.debug(f"ACP: ignoring notification {message['method']}")

    def _resolve(self, message: dict[str, Any]):
        """Complete the future waiting on this response."""
        future = self._pending.get(message.get("id"))
        if not future or future.done():
            return
        if error := message.get("error"):
            future.set_exception(
                ACPError(error.get("code", -32603), error.get("message", ""), error.get("data"))
            )
        else:
            future.set_result(message.get("result") or {})

    async def _serve(self, message: dict[str, Any]):
        """Answer an agent-initiated request."""
        request_id = message["id"]
        method = message["method"]
        try:
            if not self.on_client_request:
                raise ACPError(-32601, f"Method not supported: {method}")
            result = await self.on_client_request(
                str(request_id), method, message.get("params", {})
            )
            await self._write({"jsonrpc": "2.0", "id": request_id, "result": result})
        except ACPError as e:
            await self._write(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": e.code, "message": e.message, "data": e.data},
                }
            )
        except Exception as e:
            logger.exception(f"ACP: error serving {method}: {e}")
            await self._write(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32603, "message": str(e)},
                }
            )
