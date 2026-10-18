#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A websocket client for the OpenClaw Gateway.

The Gateway is the surface an OpenClaw agent publishes for other programs to
drive it: ``chat.send`` starts a run, the ``chat`` event stream follows it,
``sessions.steer`` redirects it, and ``chat.abort`` stops it. A NemoClaw sandbox
republishes that socket on the host, which is how a bot outside the sandbox
talks to an agent inside one.

This module deals in runs and events. Turning those into frames is
:class:`~pipecat.services.openclaw.gateway.OpenClawGatewayService`'s job.

The client targets the Gateway as OpenClaw v2026.6.1 speaks it. Where the
Gateway does something surprising, the surprise is written down at the method
that has to cope with it rather than summarized here.
"""

import asyncio
import json
import sys
import uuid
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal

from loguru import logger
from websockets.protocol import State

from pipecat.frames.frames import ErrorFrame
from pipecat.services.websocket_service import WebsocketService
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.utils.base_object import BaseObject

PROTOCOL_VERSION = 4
"""The Gateway protocol version this client negotiates."""

DEFAULT_GATEWAY_URL = "ws://127.0.0.1:18789"
"""Where OpenClaw's Gateway listens.

A NemoClaw sandbox republishes it on the host at 18790, so a client reaching an
agent inside a sandbox goes through that published port instead.
"""

GATEWAY_CLIENT_ID = "gateway-client"
"""What the Gateway calls a program driving an agent from outside.

The id comes from a closed set the Gateway validates the handshake against, so
it names a kind of client rather than this one; ``pipecat`` rides along as the
display name.

Paired with the ``backend`` mode, this is also what earns a token-only client
its operator scopes: only a backend client may skip device pairing on a
loopback connection. Any other identity connects and is then refused
``chat.send`` for want of ``operator.write``.

The Gateway puts this identity in the message envelope it shows the agent, and
marks it untrusted. An agent will remark on that in its answer unless the
caller's own framing tells it not to describe how the message arrived.
"""

DEFAULT_SESSION_KEY = "agent:main:main"
"""OpenClaw's default session.

Sessions are durable and hold conversation history, so two runs against the
same key share context. Tests and parallel callers want distinct keys.
"""

DEFAULT_CONNECT_TIMEOUT = 15.0
"""Seconds to wait for the handshake.

Sits below a pipeline's setup budget, so a service reports an unreachable
Gateway itself rather than timing out the whole setup.
"""

DEFAULT_REQUEST_TIMEOUT = 30.0
"""Seconds to wait for a Gateway method to answer."""

DEFAULT_RUN_TIMEOUT = 300.0
"""Seconds the agent is given to finish a run."""

DEFAULT_ROLE = "operator"
"""Handshake role."""

DEFAULT_MAX_MESSAGE_SIZE = 25 * 1024 * 1024
"""Largest websocket frame to accept. Agent output can be large."""

MAX_BUFFERED_FRAMES = 100
"""How many unrouted chat frames to hold for replay. See :meth:`_route`."""

EventKind = Literal["text_delta", "completed", "cancelled", "failed"]
"""What a run's event reports."""

RunStatus = Literal["completed", "cancelled", "failed"]
"""How a run ended."""
"""What a run's event stream can carry."""


class OpenClawError(Exception):
    """An error returned by the Gateway, or raised while talking to it."""

    def __init__(self, message: str, code: str | int | None = None):
        """Initialize the error.

        Args:
            message: Human-readable description.
            code: The Gateway's error code, when it sent one.
        """
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class OpenClawEvent:
    """One thing that happened during a run.

    Parameters:
        kind: What happened.
        text: The chunk of the answer, or the reason the run ended.
    """

    kind: EventKind
    text: str = ""


@dataclass(frozen=True)
class OpenClawResult:
    """A finished run, folded into the single answer it produced.

    Parameters:
        summary: The agent's final answer, or why there isn't one.
        status: How the run ended, in the same words as
            :class:`~pipecat.services.openclaw.frames.OpenClawEndFrame`.
    """

    summary: str
    status: RunStatus = "completed"


class OpenClawRun:
    """A run in flight, and the events arriving for it.

    Mutable on purpose. Steering does not merge a follow-up into the running
    turn, it interrupts that run and starts a new one, so :attr:`run_id` moves
    to the replacement and the stream follows it. See
    :meth:`OpenClawGatewayClient.steer`.
    """

    def __init__(self, run_id: str, session_key: str):
        """Initialize the run.

        Args:
            run_id: The id the run is addressed by.
            session_key: The session it runs in. Steering and aborting address
                the session, so a run started somewhere other than the client's
                default has to carry where that was.
        """
        self.run_id = run_id
        self.session_key = session_key
        self.ids: set[str] = {run_id}
        self.queue: asyncio.Queue[OpenClawEvent] = asyncio.Queue()
        self.done = False

    def __str__(self):
        """Return a readable identifier for this run."""
        return f"OpenClawRun({self.run_id})"


async def collect_result(events: AsyncIterator[OpenClawEvent]) -> OpenClawResult:
    """Fold a run's event stream into one answer.

    Args:
        events: The stream from :meth:`OpenClawGatewayClient.events`.

    Returns:
        The agent's answer and how the run ended.
    """
    parts: list[str] = []
    async for event in events:
        if event.kind == "text_delta" and event.text:
            parts.append(event.text)
        elif event.kind == "completed":
            return OpenClawResult(event.text or "".join(parts).strip())
        elif event.kind == "cancelled":
            return OpenClawResult(event.text or "The agent run was cancelled.", "cancelled")
        elif event.kind == "failed":
            return OpenClawResult(event.text or "The agent run failed.", "failed")
    return OpenClawResult("The agent run ended without a final response.")


class OpenClawGatewayClient(BaseObject, WebsocketService):
    """Drives an OpenClaw agent over its Gateway websocket.

    One connection serves every run. That is not an optimization: aborting a run
    has to work after its event stream has ended, and a request written to a
    socket whose reader has stopped never gets a reply.

    A dropped socket cannot be carried back into a running turn, because the
    Gateway has no way to resume a run's event stream. Runs in flight therefore
    end as ``failed``, while the socket underneath them reconnects for the next
    one.

    A session runs one turn at a time. Starting a run while another is live is
    the caller's business; :meth:`steer` is the supported way to redirect work
    already in flight.

    Event handlers available:

    - on_connected: Called once the socket is open.
    - on_disconnected: Called once it is closed.
    - on_connection_error: Called with a message and whether the failure will
      keep recurring, when the connection fails.

    Example::

        client = OpenClawGatewayClient(token=os.getenv("OPENCLAW_TOKEN"))
        run = await client.start("What changed in the parser this week?")
        result = await collect_result(client.events(run))
        await client.disconnect()
    """

    def __init__(
        self,
        *,
        url: str = DEFAULT_GATEWAY_URL,
        token: str | None = None,
        password: str | None = None,
        session_key: str = DEFAULT_SESSION_KEY,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        run_timeout: float = DEFAULT_RUN_TIMEOUT,
        scopes: list[str] | None = None,
        role: str = DEFAULT_ROLE,
        max_message_size: int = DEFAULT_MAX_MESSAGE_SIZE,
        reconnect_on_error: bool = True,
        **kwargs,
    ):
        """Initialize the client.

        Args:
            url: The Gateway websocket, or the port a NemoClaw sandbox
                republishes it on.
            token: The Gateway's shared token, from ``gateway.auth.token`` in
                ``~/.openclaw/openclaw.json``; a NemoClaw sandbox prints its
                own with ``nemoclaw <sandbox> gateway-token --quiet``. Required
                even on loopback: without a shared secret the Gateway asks a
                backend client for a paired device identity instead, which this
                client does not carry, and refuses the handshake with
                ``NOT_PAIRED``.
            password: Gateway password, if the deployment uses one instead.
            session_key: Which OpenClaw session to run in. Runs against the
                same key share conversation history.
            connect_timeout: Seconds to wait for the handshake. The service
                connects while the pipeline sets up, which the pipeline bounds
                by its own ``setup_timeout_secs``, so a value above that budget
                never gets to report an unreachable Gateway itself.
            request_timeout: Seconds to wait for a Gateway method to answer.
                Does not apply to a run, which streams for as long as it takes.
            run_timeout: Seconds the agent is given to finish a run, sent to
                the Gateway as ``timeoutMs``. Enforced by the agent, not here.
            scopes: Handshake scopes. ``operator.write`` covers everything this
                client does — starting a run, steering it, and aborting it. The
                Gateway offers wider ones (``operator.admin``) and narrower
                (``operator.read``, which cannot start a run).
            role: Handshake role.
            max_message_size: Largest websocket frame to accept. Agent output
                can be large.
            reconnect_on_error: Whether to reconnect after the socket fails.
            **kwargs: Additional arguments passed to :class:`BaseObject`.
        """
        BaseObject.__init__(self, **kwargs)
        WebsocketService.__init__(self, reconnect_on_error=reconnect_on_error)

        self._url = url
        self._token = token
        self._password = password
        self._session_key = session_key
        self._connect_timeout = connect_timeout
        self._request_timeout = request_timeout
        self._run_timeout = run_timeout
        self._scopes = scopes if scopes is not None else ["operator.write"]
        self._role = role
        self._client_id = GATEWAY_CLIENT_ID
        self._client_mode = "backend"
        self._client_display_name = "pipecat"
        self._max_message_size = max_message_size

        self._receive_task: asyncio.Task | None = None
        self._pending: dict[str, asyncio.Future] = {}
        self._hello: asyncio.Future | None = None

        self._runs: dict[str, OpenClawRun] = {}
        self._unrouted: deque[tuple[str, dict[str, Any]]] = deque(maxlen=MAX_BUFFERED_FRAMES)

        self._register_event_handler("on_connected")
        self._register_event_handler("on_disconnected")
        self._register_event_handler("on_connection_error")

    @property
    def session_key(self) -> str:
        """The session this client runs in."""
        return self._session_key

    #
    # Lifecycle
    #

    async def connect(self):
        """Open the Gateway connection and complete the handshake.

        Called automatically by :meth:`start`. Call it directly to fail fast on
        a bad token or an unreachable sandbox rather than on the first run.
        """
        if self._websocket is not None:
            await self._wait_ready()
            return
        await self._connect()

    async def disconnect(self):
        """Close the connection and fail any run still streaming."""
        await self._disconnect()

    async def cleanup(self):
        """Release the connection at teardown."""
        await self.disconnect()
        await super().cleanup()

    #
    # Runs
    #

    async def start(self, message: str, *, session_key: str | None = None) -> OpenClawRun:
        """Start a run and return a handle to its event stream.

        Args:
            message: What to send the agent, verbatim. Any framing the agent
                needs belongs in here; this client adds none.
            session_key: Run in this session instead of the client's default.

        Returns:
            The run, ready to pass to :meth:`events`.
        """
        await self.connect()

        # Registered before the request is sent, so frames that arrive while it
        # is still in flight have somewhere to go.
        run = OpenClawRun(uuid.uuid4().hex, session_key or self._session_key)
        self._runs[run.run_id] = run

        try:
            payload = await self._request(
                "chat.send",
                {
                    "sessionKey": run.session_key,
                    "message": message,
                    "timeoutMs": int(self._run_timeout * 1000),
                    "idempotencyKey": run.run_id,
                },
            )
        except BaseException:
            self._forget(run)
            raise

        self._adopt_id(run, payload)
        logger.debug(f"{self} started run {run.run_id}")
        return run

    async def events(self, run: OpenClawRun) -> AsyncIterator[OpenClawEvent]:
        """Stream a run's events until it reaches a terminal state.

        The stream ends after exactly one of ``completed``, ``cancelled``, or
        ``failed``. A dropped connection arrives as ``failed`` rather than
        hanging.

        Args:
            run: The run to follow.

        Yields:
            Each event as it arrives.
        """
        try:
            while True:
                event = await run.queue.get()
                yield event
                if event.kind != "text_delta":
                    return
        finally:
            self._forget(run)

    async def steer(self, run: OpenClawRun, message: str) -> None:
        """Redirect the session onto a follow-up, and follow it.

        ``sessions.steer`` does not inject into the running turn. It answers
        ``{"status": "started", "interruptedActiveRun": true}``: the active run
        is aborted and a *new* run carries the follow-up. Its frames arrive on
        the same connection, so moving the run's id onto the replacement is
        enough for :meth:`events` to keep streaming.

        The id moves before the request is sent, not after. The old run's
        ``aborted`` frame can arrive first, and if the run still answered to it
        the stream would end there: the caller would be told the task was
        cancelled while the steered run continued unwatched.

        Args:
            run: The run to redirect.
            message: The follow-up, verbatim.
        """
        await self.connect()

        previous = run.run_id
        previous_ids = set(run.ids)
        self._rekey(run, uuid.uuid4().hex)
        try:
            payload = await self._request(
                "sessions.steer",
                {
                    "key": run.session_key,
                    "message": message,
                    "idempotencyKey": run.run_id,
                },
            )
        except BaseException:
            # The steer never reached the Gateway, so the run the caller has is
            # still the one running there. It has to answer to its own ids
            # again or its terminal event would arrive for nobody.
            self._restore(run, previous, previous_ids)
            raise

        self._adopt_id(run, payload)
        interrupted = payload.get("interruptedActiveRun") if isinstance(payload, dict) else None
        logger.debug(
            f"{self} steered onto run {run.run_id} (was {previous}, interrupted={interrupted})"
        )

    async def abort(self, run: OpenClawRun, reason: str | None = None) -> bool:
        """Stop a run.

        The Gateway answers a live run with ``{"ok": true, "aborted": true,
        "runIds": [id]}``, and answers both a finished run and an unknown one
        with ``aborted: false``. So ``False`` means there was nothing to stop,
        which is the routine race when a caller aborts a moment after the agent
        finished, and not a failure. A real Gateway error arrives as
        ``ok: false`` and is raised.

        Args:
            run: The run to stop.
            reason: Logged, for working out later why something was aborted.

        Returns:
            Whether a run was actually stopped.
        """
        await self.connect()

        logger.debug(f"{self} aborting run {run.run_id}: {reason}")
        payload = await self._request(
            "chat.abort",
            {"sessionKey": run.session_key, "runId": run.run_id},
        )
        if not isinstance(payload, dict):
            raise OpenClawError(f"Unexpected chat.abort response: {payload!r}")
        return payload.get("aborted") is True

    #
    # Connection
    #

    async def _connect(self):
        """Open the socket, start reading, and settle the handshake."""
        await super()._connect()
        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._read_until_closed(), name=f"{self}::read")
        try:
            await self._wait_ready()
        except BaseException:
            await self._disconnect()
            raise

    async def _disconnect(self):
        """Stop reading and close the socket."""
        await super()._disconnect()
        if self._receive_task:
            await self.cancel_task(self._receive_task, timeout=1.0)
            self._receive_task = None
        await self._disconnect_websocket()
        self._runs.clear()
        self._unrouted.clear()

    async def _connect_websocket(self):
        """Open the socket and arm the handshake.

        The Gateway challenges rather than waiting to be asked, so the
        handshake completes on the receive task. Callers wait for it through
        :meth:`_wait_ready`, which also covers the socket a reconnect
        establishes.
        """
        if self._websocket is not None and self._websocket.state is State.OPEN:
            return

        # A client driven from a script has no pipeline to inherit a task
        # manager from.
        if self._task_manager is None:
            self._task_manager = TaskManager()

        logger.debug(
            f"{self} connecting to {self._url} "
            f"(session {self._session_key}, token {'set' if self._token else 'unset'})"
        )
        self._hello = asyncio.get_running_loop().create_future()
        # A socket that dies before anyone asks about the handshake still
        # settles it, and an unretrieved failure is reported at collection.
        self._hello.add_done_callback(lambda hello: hello.cancelled() or hello.exception())
        self._websocket = await self._websocket_connect(self._url, max_size=self._max_message_size)
        await self._call_event_handler("on_connected")

    async def _disconnect_websocket(self):
        """Close the socket and end everything that was waiting on it."""
        self._fail_live_runs(
            "The connection to the OpenClaw Gateway closed before the run finished"
        )
        self._fail_pending(OpenClawError("The OpenClaw Gateway connection closed"))
        try:
            if self._websocket:
                await self._websocket.close()
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    async def _receive_messages(self):
        """Read frames until the socket closes."""
        if self._websocket is None:
            return
        async for raw in self._websocket:
            self._dispatch(json.loads(raw))

    async def _read_until_closed(self):
        """Read frames until the receive loop ends, then settle what waited on it.

        The base class's loop has exit paths that never reach
        :meth:`_disconnect_websocket`: reconnection disabled, or a handshake
        that keeps succeeding onto a socket that closes immediately. However it
        ends, nothing is reading the socket any more, so the socket is closed
        and runs in flight are failed here rather than left waiting for events
        that cannot arrive. An intentional disconnect does this itself.
        """
        try:
            await self._receive_task_handler(self._report_error)
        finally:
            self._receive_task = None

        if not self._disconnecting:
            await self._disconnect_websocket()

    async def _report_error(self, error: ErrorFrame, force_treat_as_permanent: bool = False):
        """Hand a connection failure to whoever is driving the client.

        Args:
            error: The failure to report.
            force_treat_as_permanent: Whether the failure will keep recurring,
                which a driver reports onwards so the pipeline can stop sending
                work this way.
        """
        await self._call_event_handler("on_connection_error", error.error, force_treat_as_permanent)

    async def _wait_ready(self):
        """Wait for the handshake to settle."""
        if self._hello is None:
            raise OpenClawError("The OpenClaw Gateway is not connected")
        # Shielded so a timeout here leaves the handshake standing for whoever
        # asks next.
        await asyncio.wait_for(asyncio.shield(self._hello), timeout=self._connect_timeout)

    async def _request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Send a request and wait for its response."""
        if self._websocket is None:
            raise OpenClawError("The OpenClaw Gateway is not connected")
        request_id = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._websocket.send(
                json.dumps(
                    {"type": "req", "id": request_id, "method": method, "params": params},
                    separators=(",", ":"),
                )
            )
            return await asyncio.wait_for(future, timeout=self._request_timeout)
        finally:
            self._pending.pop(request_id, None)

    def _dispatch(self, frame: dict[str, Any]):
        """Route one inbound frame to a handler or a pending request."""
        frame_type = frame.get("type")
        if frame_type == "event":
            if frame.get("event") == "connect.challenge":
                self.create_task(self._send_connect(), name="openclaw-connect")
            elif frame.get("event") == "chat":
                payload = frame.get("payload")
                if isinstance(payload, dict):
                    self._route(payload)
            return

        if frame_type != "res":
            return

        future = self._pending.pop(str(frame.get("id")), None)
        if frame.get("ok"):
            # Some deployments answer the handshake on a response this client
            # is not waiting on, so hello is resolved by payload as well as by
            # the connect request completing.
            if self._hello and not self._hello.done() and _is_hello_ok(frame.get("payload")):
                self._hello.set_result(frame.get("payload"))
            if future and not future.done():
                future.set_result(frame.get("payload"))
        elif future and not future.done():
            error = frame.get("error") or {}
            future.set_exception(
                OpenClawError(error.get("message") or str(error), error.get("code"))
            )

    def _fail_pending(self, error: Exception):
        """Fail everything still waiting on the Gateway."""
        if self._hello and not self._hello.done():
            self._hello.set_exception(error)
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)
        self._pending.clear()

    def _client_info(self) -> dict[str, Any]:
        """Describe this client to the Gateway.

        The identity decides how the agent is told about the message. Override
        to present a different one.
        """
        return {
            "id": self._client_id,
            "displayName": self._client_display_name,
            "version": "0.1.0",
            "platform": sys.platform,
            "mode": self._client_mode,
        }

    async def _send_connect(self):
        """Answer the connect challenge and settle the handshake."""
        auth: dict[str, str] = {}
        if self._token:
            auth["token"] = self._token
        if self._password:
            auth["password"] = self._password

        params: dict[str, Any] = {
            "minProtocol": PROTOCOL_VERSION,
            "maxProtocol": PROTOCOL_VERSION,
            "client": self._client_info(),
            "caps": [],
            "role": self._role,
            "scopes": self._scopes,
        }
        if auth:
            params["auth"] = auth

        hello = self._hello
        try:
            payload = await self._request("connect", params)
        except Exception as e:
            if hello and not hello.done():
                hello.set_exception(e)
            return
        if hello and not hello.done():
            hello.set_result(payload)

    #
    # Routing
    #

    def _route(self, payload: dict[str, Any]):
        """Deliver one ``chat`` payload to the run it belongs to.

        A frame for an unknown run is held rather than dropped. The Gateway can
        answer ``chat.send`` with a run id of its own choosing, and frames
        carrying that id can arrive before the response that names it, so the
        buffer is replayed whenever a run learns a new id.
        """
        run_id = str(payload.get("runId") or "")
        run = self._runs.get(run_id)
        if run is None:
            self._unrouted.append((run_id, payload))
            return

        state = payload.get("state")
        text = _extract_text(payload.get("message"))
        if state == "delta":
            # A delta frame carries both the piece the agent just produced and
            # the whole answer so far. Only the piece is new: the message
            # restates everything already sent, so joining those would repeat
            # the answer.
            delta = payload.get("deltaText")
            if isinstance(delta, str) and delta:
                run.queue.put_nowait(OpenClawEvent("text_delta", text=delta))
        elif state == "final":
            self._finish(run, OpenClawEvent("completed", text=text))
        elif state == "aborted":
            self._finish(run, OpenClawEvent("cancelled", text=text))
        elif state == "error":
            self._finish(
                run, OpenClawEvent("failed", text=str(payload.get("errorMessage") or text))
            )
        else:
            logger.trace(f"{self} ignoring chat frame in state {state!r}")

    def _finish(self, run: OpenClawRun, event: OpenClawEvent):
        """Deliver a run's terminal event, at most once."""
        if run.done:
            return
        run.done = True
        run.queue.put_nowait(event)

    def _adopt_id(self, run: OpenClawRun, payload: Any):
        """Answer to the run id the Gateway chose, as well as ours."""
        if not isinstance(payload, dict):
            return
        run_id = payload.get("runId")
        if not run_id or str(run_id) in run.ids:
            return
        run.run_id = str(run_id)
        run.ids.add(run.run_id)
        self._runs[run.run_id] = run
        self._replay()

    def _rekey(self, run: OpenClawRun, run_id: str):
        """Move a run onto a new id and stop answering to the old ones."""
        for old in run.ids:
            self._runs.pop(old, None)
        run.ids = {run_id}
        run.run_id = run_id
        self._runs[run_id] = run

    def _restore(self, run: OpenClawRun, run_id: str, ids: set[str]):
        """Put a run back on the ids it answered to before a failed rekey."""
        self._runs.pop(run.run_id, None)
        run.ids = set(ids)
        run.run_id = run_id
        for old in run.ids:
            self._runs[old] = run

    def _replay(self):
        """Re-route buffered frames now that a run has a new id.

        Frames that still match nothing are re-buffered by :meth:`_route`, so a
        replay never discards one.
        """
        buffered, self._unrouted = list(self._unrouted), deque(maxlen=MAX_BUFFERED_FRAMES)
        for _, payload in buffered:
            self._route(payload)

    def _forget(self, run: OpenClawRun):
        """Stop routing to a run whose stream has ended."""
        for run_id in run.ids:
            self._runs.pop(run_id, None)

    def _fail_live_runs(self, reason: str):
        """End every run still streaming, so nothing waits on a dead socket."""
        for run in set(self._runs.values()):
            self._finish(run, OpenClawEvent("failed", text=reason))


def _is_hello_ok(payload: Any) -> bool:
    """Whether a response payload is the Gateway's handshake acknowledgement."""
    return isinstance(payload, dict) and payload.get("type") == "hello-ok"


def _extract_text(value: Any) -> str:
    """Pull speakable text out of a chat frame's message.

    A heuristic, not a schema. The Gateway's message shape is not documented
    well enough to model, so this tries the keys it has been observed to use
    and falls back to the raw JSON rather than silently returning nothing.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "content", "message", "output", "summary"):
            text = value.get(key)
            if isinstance(text, str):
                return text
        content = value.get("content")
        if isinstance(content, list):
            return "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            )
    return json.dumps(value, ensure_ascii=False)
