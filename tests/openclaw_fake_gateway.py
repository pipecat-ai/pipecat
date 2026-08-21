#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A minimal OpenClaw Gateway, for testing the client against a real socket.

Speaks the Gateway's framing: it challenges on connect, answers ``req`` frames
with ``res`` frames, and pushes ``chat`` events on demand. Tests drive it
directly, so the timing that matters (a frame arriving before the response that
names its run, an abort landing after a run finished) can be reproduced exactly.

Use it as an async context manager::

    async with FakeGateway() as gateway:
        client = OpenClawGatewayClient(url=gateway.url)
        run = await client.start("hello")
        await gateway.chat(run.run_id, "final", "hi")
"""

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from websockets.asyncio.server import serve


def _assistant_message(text: str) -> dict[str, Any]:
    """Wrap text the way the Gateway wraps a chat message."""
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


class FakeGateway:
    """A scriptable stand-in for the OpenClaw Gateway.

    Attributes:
        requests: Every request frame received, in order.
        run_id: What ``chat.send`` reports as the run id. None echoes the
            client's idempotency key, which is what OpenClaw does.
        aborted: What ``chat.abort`` reports.
        errors: Methods that should answer with an error instead of a payload.
    """

    def __init__(self):
        """Initialize the gateway."""
        self.requests: list[dict[str, Any]] = []
        self.run_id: str | None = None
        self.aborted: bool = True
        self.errors: dict[str, dict[str, Any]] = {}
        self.on_request: Callable[[dict[str, Any]], Awaitable[None]] | None = None
        self.url: str = ""

        self._streamed: dict[str, str] = {}
        self._server = None
        self._ws: Any = None
        self._connected = asyncio.Event()
        self._arrived = asyncio.Event()

    async def __aenter__(self):
        """Start serving on an ephemeral loopback port."""
        self._server = await serve(self._handle, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        self.url = f"ws://127.0.0.1:{port}"
        return self

    async def __aexit__(self, *args):
        """Stop serving."""
        self._server.close()
        await self._server.wait_closed()

    #
    # Driving the client
    #

    async def chat(
        self,
        run_id: str,
        state: str,
        message: Any = None,
        *,
        error_message: str | None = None,
    ):
        """Push one ``chat`` event.

        A ``delta`` carries the piece just produced and, like the real Gateway,
        the whole answer so far alongside it. Pass the piece; the rest is
        accumulated here.
        """
        payload: dict[str, Any] = {"runId": run_id, "state": state}
        if state == "delta" and isinstance(message, str):
            payload["deltaText"] = message
            self._streamed[run_id] = self._streamed.get(run_id, "") + message
            payload["message"] = _assistant_message(self._streamed[run_id])
        elif message is not None:
            payload["message"] = message
        if error_message is not None:
            payload["errorMessage"] = error_message
        await self._send({"type": "event", "event": "chat", "payload": payload})

    async def drop(self):
        """Close the socket, as a sandbox restart or a network blip would."""
        if self._ws:
            await self._ws.close()

    async def wait_for(self, method: str, timeout: float = 2.0) -> dict[str, Any]:
        """Wait until a request for ``method`` has arrived, and return it."""

        async def _wait():
            while True:
                for request in self.requests:
                    if request.get("method") == method:
                        return request
                self._arrived.clear()
                await self._arrived.wait()

        return await asyncio.wait_for(_wait(), timeout=timeout)

    def params(self, method: str) -> dict[str, Any]:
        """The params of the last request for ``method``."""
        for request in reversed(self.requests):
            if request.get("method") == method:
                return request.get("params") or {}
        raise AssertionError(f"no {method} request was received")

    def count(self, method: str) -> int:
        """How many requests for ``method`` arrived."""
        return sum(1 for r in self.requests if r.get("method") == method)

    #
    # Serving
    #

    async def _handle(self, websocket):
        """Serve one client connection."""
        self._ws = websocket
        self._connected.set()
        await self._send({"type": "event", "event": "connect.challenge"})
        try:
            async for raw in websocket:
                await self._serve(json.loads(raw))
        except Exception:
            pass

    async def _serve(self, frame: dict[str, Any]):
        """Answer one request frame."""
        if frame.get("type") != "req":
            return
        self.requests.append(frame)
        self._arrived.set()

        method = frame.get("method")
        if self.on_request:
            # Before the response, so a test can reproduce a chat frame that
            # arrives ahead of the response naming its run.
            await self.on_request(frame)

        if error := self.errors.get(method):
            await self._send({"type": "res", "id": frame["id"], "ok": False, "error": error})
            return

        await self._send(
            {
                "type": "res",
                "id": frame["id"],
                "ok": True,
                "payload": self._payload(method, frame.get("params") or {}),
            }
        )

    def _payload(self, method: str, params: dict[str, Any]) -> Any:
        """What this gateway answers each method with."""
        if method == "connect":
            return {"type": "hello-ok", "protocol": params.get("maxProtocol")}
        if method == "chat.send":
            return {"runId": self.run_id or params.get("idempotencyKey")}
        if method == "sessions.steer":
            return {
                "status": "started",
                "interruptedActiveRun": True,
                "runId": self.run_id or params.get("idempotencyKey"),
            }
        if method == "chat.abort":
            return {
                "ok": True,
                "aborted": self.aborted,
                "runIds": [params.get("runId")] if self.aborted else [],
            }
        return {}

    async def _send(self, frame: dict[str, Any]):
        """Write one frame to the connected client."""
        if self._ws:
            await self._ws.send(json.dumps(frame))
