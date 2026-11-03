#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Local Prism protocol peer used with the published Rime SDK."""

import asyncio
import base64
import json
from contextlib import asynccontextmanager

from rimelabs_sdk import Rime
from websockets.asyncio.server import serve

from pipecat.frames.frames import (
    EndFrame,
    LLMRunFrame,
    LLMServiceMetadataFrame,
    StartFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.rime.realtime.llm import RimeRealtimeLLMService
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.workers.runner import WorkerRunner


class Peer:
    """Accept SDK requests and let each test control response events and failures."""

    def __init__(self, *, unversioned=False):
        self.unversioned = unversioned
        self.socket = None
        self.events = []
        self.requests = asyncio.Queue()
        self.receipts = asyncio.Queue()
        self.history = []
        self.settings = {}
        self.revision = 0
        self.serial = 0
        self.gates = {}
        self.refuse = {}
        self.history_failure_at = None
        self.history_mutations = 0
        self.history_serial = 0

    async def emit(self, kind, **body):
        await self.socket.send(json.dumps({"type": kind, "event_id": "peer", **body}))

    async def handler(self, socket):
        self.socket = socket
        await self.emit("session.created", session={"id": "session-1"})
        async with asyncio.TaskGroup() as tasks:
            async for raw in socket:
                event = json.loads(raw)
                self.events.append(event)
                tasks.create_task(self.handle_request(event))

    async def handle_request(self, event):
        kind = event["type"]
        if kind in self.gates:
            await self.gates[kind].wait()
        if kind in ("conversation.item.create", "conversation.item.delete"):
            self.history_mutations += 1
        refusal = self.refuse.pop(kind, None)
        if refusal or self.history_failure_at == self.history_mutations:
            self.history_failure_at = None
            await self.emit(
                "error",
                error={
                    "code": refusal or "invalid_request",
                    "message": "Peer refused request",
                    "scope": "event",
                    "owner": {"kind": "event", "event_id": event["event_id"]},
                },
            )
            return
        if kind == "session.update":
            self.settings.update(event["session"])
            self.revision += 1
            await self.emit(
                "session.updated",
                prism_request_event_id=event["event_id"],
                session={
                    **self.settings,
                    "id": "session-1",
                    "voice": "test",
                    "request_id": "actual-peer-request",
                    **({"prism_settings_revision": self.revision} if not self.unversioned else {}),
                    "prism_effective_interrupt_response": self.settings.get(
                        "turn_detection", {}
                    ).get("interrupt_response", True),
                },
            )
            await self.emit("prism.typed_input.ready")
        elif kind == "conversation.item.create":
            self.history_serial += 1
            item = {"id": f"history-{self.history_serial}", **event["item"]}
            anchor = event.get("previous_item_id")
            if anchor == "root":
                self.history.insert(0, item)
            elif anchor is not None:
                index = next(i for i, value in enumerate(self.history) if value["id"] == anchor)
                self.history.insert(index + 1, item)
            elif item["type"] == "function_call_output":
                index = next(
                    (
                        i
                        for i, value in enumerate(self.history)
                        if value.get("call_id") == item["call_id"]
                    ),
                    None,
                )
                if index is None:
                    self.history.append(item)
                else:
                    self.history.insert(index + 1, item)
            else:
                self.history.append(item)
            await self.emit(
                "conversation.item.created", prism_request_event_id=event["event_id"], item=item
            )
        elif kind == "conversation.item.delete":
            item = next((item for item in self.history if item["id"] == event["item_id"]), None)
            self.history = [
                value
                for value in self.history
                if value["id"] != event["item_id"]
                and not (item and item.get("call_id") and value.get("call_id") == item["call_id"])
            ]
            await self.emit(
                "conversation.item.deleted",
                prism_request_event_id=event["event_id"],
                item_id=event["item_id"],
            )
        elif kind == "prism.conversation.get":
            await self.emit(
                "prism.conversation",
                prism_request_event_id=event["event_id"],
                items=self.history,
            )
        elif kind == "prism.playback.drained":
            self.receipts.put_nowait(event)
            await self.emit("prism.typed_input.ready")
        elif kind == "response.cancel":
            await self.end(event["response_id"], status="cancelled")
        elif kind == "conversation.item.truncate":
            await self.emit(
                "conversation.item.truncated",
                item_id=event["item_id"],
                content_index=event["content_index"],
                audio_end_ms=event["audio_end_ms"],
            )
        elif kind == "input_audio_buffer.clear":
            await self.emit("input_audio_buffer.cleared")
        else:
            self.requests.put_nowait(event)

    async def next(self, kind="response.create"):
        async with asyncio.timeout(3):
            while True:
                event = await self.requests.get()
                if event["type"] == kind:
                    return event

    async def start(self, request=None, *, input_ids=None, response_id=None):
        self.serial += 1
        ref = response_id or f"response-{self.serial}"
        response = {
            "id": ref,
            **({"prism_settings_revision": self.revision} if not self.unversioned else {}),
            "metadata": request["response"]["metadata"]
            if request
            else {"prism_cause": "user_audio"},
        }
        if input_ids is not None:
            response["prism_input_item_ids"] = input_ids
        if request and "prism_input_text" in request["response"]:
            response["prism_input_item_ids"] = [f"typed-{self.serial}"]
        await self.emit(
            "response.created",
            response=response,
            **({"prism_request_event_id": request["event_id"]} if request else {}),
        )
        return ref

    async def message(self, ref, *, item_id=None, text="Hello", samples=480, done=True):
        item_id = item_id or f"message-{ref}"
        await self.emit(
            "response.output_item.added",
            response_id=ref,
            output_index=0,
            item={"id": item_id, "type": "message"},
        )
        common = dict(response_id=ref, output_index=0, item_id=item_id, content_index=0)
        await self.emit("response.text.delta", **common, delta=text)
        await self.emit("response.text.done", **common, text=text)
        if samples:
            await self.emit(
                "response.audio.delta",
                **common,
                delta=base64.b64encode(b"\x10\x00" * samples).decode(),
            )
        if done:
            await self.emit("response.audio.done", **common)
        return common

    async def end(self, ref, status="completed", *, ready=True):
        await self.emit(
            "response.done",
            response={
                "id": ref,
                "status": status,
                "metadata": {"prism_drain_token": f"drain-{ref}"},
            },
        )
        # The server admits another turn after its playback estimate expires.
        if ready:
            await self.emit("prism.typed_input.ready")


class Capture(FrameProcessor):
    def __init__(self, frames, direction):
        super().__init__()
        self.frames = frames
        self.direction = direction

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if direction == self.direction:
            self.frames.append(frame)
        await self.push_frame(frame, direction)


class Output(BaseOutputTransport):
    """Capture audio written through the unchanged output transport."""

    def __init__(self, *, enabled=True, channels=1):
        super().__init__(
            TransportParams(
                audio_out_enabled=enabled, audio_out_sample_rate=24000, audio_out_channels=channels
            )
        )
        self.writes = []

    async def start(self, frame: StartFrame):
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame):
        self.writes.append(frame)
        return True


async def until(predicate, timeout=3):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


@asynccontextmanager
async def conversation(*, context=None, enabled=True, unversioned=False, **options):
    peer = Peer(unversioned=unversioned)
    context = context or LLMContext()
    upstream, downstream = [], []
    async with serve(peer.handler, "127.0.0.1", 0) as server:
        endpoint = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1/realtime"
        async with Rime(api_key="local-peer") as client:
            service = RimeRealtimeLLMService(
                client=client, endpoint=endpoint, shutdown_timeout=0.4, **options
            )
            pair = service.create_context_aggregator(context)
            output = Output(enabled=enabled)
            pipeline = Pipeline(
                [
                    Capture(upstream, FrameDirection.UPSTREAM),
                    pair.user(),
                    service,
                    Capture(upstream, FrameDirection.UPSTREAM),
                    output,
                    Capture(downstream, FrameDirection.DOWNSTREAM),
                    pair.assistant(),
                ]
            )
            worker = PipelineWorker(pipeline, enable_rtvi=False, cancel_on_idle_timeout=False)
            runner = WorkerRunner(handle_sigint=False)
            started = asyncio.Event()

            @worker.event_handler("on_pipeline_started")
            async def on_started(*args):
                started.set()

            await runner.add_workers(worker)
            async with asyncio.TaskGroup() as tasks:
                running = tasks.create_task(runner.run())
                await asyncio.wait_for(started.wait(), 3)
                await until(lambda: any(isinstance(f, LLMServiceMetadataFrame) for f in upstream))
                await worker.queue_frame(LLMRunFrame())
                initial = await peer.next()
                response = await peer.start(initial)
                await peer.end(response)
                await until(
                    lambda: any(type(f).__name__ == "LLMFullResponseEndFrame" for f in downstream)
                )
                try:
                    yield peer, service, worker, context, output, upstream, downstream
                finally:
                    for gate in peer.gates.values():
                        gate.set()
                    await worker.queue_frame(EndFrame())
                    await asyncio.wait_for(running, 3)
