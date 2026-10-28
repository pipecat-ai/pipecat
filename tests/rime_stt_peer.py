#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Loopback gRPC peer for the published Rime SDK and real Pipecat pipelines."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import grpc
from rime_api import speech_to_text_pb2 as proto

from pipecat.frames.frames import StartFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.workers.runner import WorkerRunner
from tests.rime_stt_helpers import Capture


@dataclass
class Failure:
    code: grpc.StatusCode
    detail: str = "Peer refused recognition"


@dataclass
class Call:
    request_id: str
    request: object
    language: str
    audio: list[bytes] = field(default_factory=list)
    received_audio: asyncio.Event = field(default_factory=asyncio.Event)
    eof: asyncio.Event = field(default_factory=asyncio.Event)
    closed: asyncio.Event = field(default_factory=asyncio.Event)
    responses: asyncio.Queue = field(default_factory=asyncio.Queue)
    text: str = ""
    revision: int = 0

    def update(self, text: str, *, append: bool = False):
        if append:
            self.text += text
            message = proto.StreamingTranscriptionResponse(
                delta=proto.TranscriptionDelta(text=text)
            )
        else:
            self.text = text
            self.revision += 1
            message = proto.StreamingTranscriptionResponse(
                hypothesis=proto.TranscriptionHypothesis(text=text, revision=self.revision)
            )
        self.responses.put_nowait(message)

    def finish(self, *, text: str | None = None):
        self.responses.put_nowait(
            proto.StreamingTranscriptionResponse(
                done=proto.TranscriptionDone(
                    text=self.text if text is None else text,
                    revision=self.revision,
                    language=language(self.language),
                )
            )
        )
        self.responses.put_nowait(None)


def language(tag: str):
    return proto.ResolvedLanguage(tag=tag, source=proto.LANGUAGE_SOURCE_SELECTED)


class Peer:
    """Expose canonical STT RPCs with test-controlled responses and completion."""

    def __init__(self):
        self.calls: list[Call] = []
        self.requests: asyncio.Queue[Call] = asyncio.Queue()

    async def _begin(self, request, tag, context):
        call = Call(f"peer-{len(self.calls)}", request, tag)
        self.calls.append(call)
        context.add_done_callback(lambda _: call.closed.set())
        await context.send_initial_metadata((("x-request-id", call.request_id),))
        self.requests.put_nowait(call)
        return call

    async def transcribe(self, request, context):
        call = await self._begin(request, request.options.language, context)
        call.audio.append(request.audio)
        call.received_audio.set()
        call.eof.set()
        response = await call.responses.get()
        if isinstance(response, Failure):
            await context.abort(response.code, response.detail)
        return response

    async def stream(self, requests, context):
        request = await anext(requests)
        if request.WhichOneof("payload") != "config":
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Configuration must arrive first")
        call = await self._begin(request.config, request.config.language, context)
        yield proto.StreamingTranscriptionResponse(
            accepted=proto.StreamingAccepted(
                output_contract=request.config.output_contract,
                language=language(call.language),
            )
        )

        async def read_audio():
            async for item in requests:
                if item.WhichOneof("payload") != "audio":
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Expected PCM audio")
                call.audio.append(item.audio)
                call.received_audio.set()
            call.eof.set()

        async with asyncio.TaskGroup() as tasks:
            reader = tasks.create_task(read_audio())
            try:
                while True:
                    response = await call.responses.get()
                    if isinstance(response, Failure):
                        await context.abort(response.code, response.detail)
                    if response is None:
                        return
                    yield response
            finally:
                reader.cancel()

    async def next_call(self) -> Call:
        async with asyncio.timeout(3):
            return await self.requests.get()


@asynccontextmanager
async def peer_server(monkeypatch):
    """Redirect only the gRPC connection; SDK and protobuf processing stay real.

    The loopback channel is plaintext. These tests do not exercise TLS or the
    public Rime endpoint. No provider credential is used.
    """
    peer = Peer()
    server = grpc.aio.server()
    server.add_generic_rpc_handlers(
        (
            grpc.method_handlers_generic_handler(
                "rime.SpeechToText",
                {
                    "Transcribe": grpc.unary_unary_rpc_method_handler(
                        peer.transcribe,
                        request_deserializer=proto.TranscriptionRequest.FromString,
                        response_serializer=proto.TranscriptionResponse.SerializeToString,
                    ),
                    "TranscribeStreaming": grpc.stream_stream_rpc_method_handler(
                        peer.stream,
                        request_deserializer=proto.StreamingTranscriptionRequest.FromString,
                        response_serializer=proto.StreamingTranscriptionResponse.SerializeToString,
                    ),
                },
            ),
        )
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channels = []

    def connect(target, credentials, **kwargs):
        channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}", **kwargs)
        channels.append(channel)
        return channel

    with monkeypatch.context() as patch:
        patch.setattr(grpc.aio, "secure_channel", connect)
        try:
            yield peer
        finally:
            for channel in channels:
                await channel.close()
            await server.stop(None)


@asynccontextmanager
async def pipeline(service):
    """Drive frames through a worker with ordinary queued processing."""
    before, after = Capture(), Capture()
    worker = PipelineWorker(
        Pipeline([before, service, after]),
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        enable_rtvi=False,
        cancel_on_idle_timeout=False,
    )
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    async with asyncio.TaskGroup() as tasks:
        running = tasks.create_task(runner.run())
        await after.wait_for(StartFrame)
        try:
            yield worker, after, before
        finally:
            if not running.done():
                await worker.cancel()
            await asyncio.wait_for(running, 3)
