#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A local stand-in for the Responses API's WebSocket mode, with ``stream_id`` lanes.

It follows the documented lane semantics: a request with ``stream_id`` runs on
that lane, lanes are first-in, first-out and never overlap, different lanes run
concurrently, and every event for a named lane carries ``stream_id`` (events for
a request without one carry none). Each lane keeps its latest completed response
id, and a ``previous_response_id`` that is not the lane's latest is answered with
``previous_response_not_found``. Up to 16 responses may be in flight and up to 32
lanes named per connection. A response runs to completion whatever the client
sends after ``response.create``, as the real endpoint was measured to do.

Every response sends ``response.created`` at once, then after
``first_token_delay`` one ``response.output_text.delta`` per word of its answer,
``word_period`` apart, then ``response.completed``. The first response served
can be given an extra ``first_response_gap`` after ``response.created``, like a
long reasoning stretch.
"""

import asyncio
import json

import websockets

MAX_IN_FLIGHT = 16
MAX_NAMED_LANES = 32


class LaneServer:
    """The server's state and connection handler.

    Args:
        answers: The text of the n-th response served, by n; unnamed responses
            get generated filler.
        first_token_delay: Seconds from ``response.created`` to the first delta.
        word_period: Seconds between deltas.
        words: Words of filler for responses without an answer.
        first_response_gap: Extra seconds after ``response.created`` on the first
            response served.
        tag_events: Whether events for named lanes carry ``stream_id``.
    """

    def __init__(
        self,
        *,
        answers: dict[int, str] | None = None,
        first_token_delay: float = 0.1,
        word_period: float = 0.01,
        words: int = 20,
        first_response_gap: float = 0.0,
        tag_events: bool = True,
    ):
        self.answers = answers or {}
        self.first_token_delay = first_token_delay
        self.word_period = word_period
        self.words = words
        self.first_response_gap = first_response_gap
        self.tag_events = tag_events
        self.requests: list[dict] = []  # every response.create, in arrival order
        self.connections = 0
        self.errors: list[dict] = []  # every error event sent

    async def handler(self, ws):
        """Serve one connection."""
        self.connections += 1
        lanes: dict = {}
        workers: list[asyncio.Task] = []
        latest: dict = {}
        in_flight = {"n": 0}
        named: set = set()

        async def send(msg: dict, lane):
            if lane is not None and self.tag_events:
                msg = {**msg, "stream_id": lane}
            if msg.get("type") == "error":
                self.errors.append(msg)
            await ws.send(json.dumps(msg))

        async def lane_worker(lane, queue: asyncio.Queue):
            while True:
                msg = await queue.get()
                n = len(self.requests)
                rid = f"resp_{n}"
                prev = msg.get("previous_response_id")
                if prev is not None and latest.get(lane) != prev:
                    await send(
                        {
                            "type": "error",
                            "error": {
                                "code": "previous_response_not_found",
                                "message": f"{prev} not found",
                            },
                        },
                        lane,
                    )
                    continue
                in_flight["n"] += 1
                await send({"type": "response.created", "response": {"id": rid}}, lane)
                if n == 1 and self.first_response_gap:
                    await asyncio.sleep(self.first_response_gap)
                await asyncio.sleep(self.first_token_delay)
                text = self.answers.get(n) or " ".join(f"w{n}.{i}" for i in range(self.words))
                for word in text.split(" "):
                    await send({"type": "response.output_text.delta", "delta": word + " "}, lane)
                    await asyncio.sleep(self.word_period)
                await send(
                    {
                        "type": "response.completed",
                        "response": {
                            "id": rid,
                            "model": "mock",
                            "output": [],
                            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                        },
                    },
                    lane,
                )
                latest[lane] = rid
                in_flight["n"] -= 1

        try:
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") != "response.create":
                    continue
                lane = msg.get("stream_id")
                self.requests.append(msg)
                if lane is not None:
                    named.add(lane)
                    if len(named) > MAX_NAMED_LANES:
                        await send(
                            {
                                "type": "error",
                                "error": {
                                    "code": "websocket_stream_limit_reached",
                                    "message": "too many named lanes",
                                },
                            },
                            lane,
                        )
                        continue
                if in_flight["n"] >= MAX_IN_FLIGHT:
                    await send({"type": "response.queued"}, lane)
                if lane not in lanes:
                    lanes[lane] = asyncio.Queue()
                    workers.append(asyncio.create_task(lane_worker(lane, lanes[lane])))
                await lanes[lane].put(msg)
        except websockets.ConnectionClosed:
            pass
        finally:
            for worker in workers:
                worker.cancel()
