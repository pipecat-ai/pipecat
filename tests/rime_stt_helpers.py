#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Scriptable public SDK boundary and a frame-processing harness for Rime STT."""

import asyncio
from contextlib import asynccontextmanager

from rimelabs_sdk import Transcription, TranscriptionFinal

from pipecat.frames.frames import InputAudioRawFrame, StartFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup


class FakeStream:
    def __init__(self, audio, options, index, *, automatic=False, consume=True):
        self.audio = audio
        self.options = options
        self.request_id = f"request-{index}"
        self.chunks = []
        self.updates = asyncio.Queue()
        self.uploaded = asyncio.Event()
        self.exhausted = asyncio.Event()
        self.closed = asyncio.Event()
        self.allow_upload = asyncio.Event()
        if consume:
            self.allow_upload.set()
        self.task = None
        self.automatic = automatic

    async def _upload(self):
        await self.allow_upload.wait()
        async for chunk in self.audio:
            self.chunks.append(chunk)
            self.uploaded.set()
        self.exhausted.set()
        if self.automatic:
            await self.finish(f"segment {self.request_id}")

    async def __aenter__(self):
        self.task = asyncio.create_task(self._upload())
        return self

    async def __aexit__(self, *args):
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        self.closed.set()

    def __aiter__(self):
        return self

    async def __anext__(self):
        update = await self.updates.get()
        if isinstance(update, Exception):
            raise update
        if update is None:
            raise StopAsyncIteration
        return update

    async def finish(self, text, language="en"):
        await self.updates.put(TranscriptionFinal(text=text, language=language))
        await self.updates.put(None)


class FakeRime:
    def __init__(self, *, automatic=False, consume=True):
        self.stt = self
        self.streams = []
        self.calls = []
        self.new_stream = asyncio.Queue()
        self.new_call = asyncio.Queue()
        self.closed = False
        self.automatic = automatic
        self.consume = consume

    def stream(self, audio, **options):
        stream = FakeStream(
            audio, options, len(self.streams), automatic=self.automatic, consume=self.consume
        )
        self.streams.append(stream)
        self.new_stream.put_nowait(stream)
        return stream

    async def transcribe(self, audio, **options):
        result = asyncio.get_running_loop().create_future()
        call = (audio, options, result)
        self.calls.append(call)
        self.new_call.put_nowait(call)
        if self.automatic:
            result.set_result(
                Transcription(
                    text=f"segment {len(self.calls)}",
                    language="en",
                    request_id=f"unary-{len(self.calls)}",
                )
            )
        return await result

    async def close(self):
        self.closed = True


class Capture(FrameProcessor):
    def __init__(self):
        super().__init__(enable_direct_mode=True)
        self.frames = []
        self.changed = asyncio.Event()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        self.frames.append(frame)
        self.changed.set()
        await self.push_frame(frame, direction)

    async def wait_for(self, cls, count=1):
        async with asyncio.timeout(2):
            while len([f for f in self.frames if isinstance(f, cls)]) < count:
                self.changed.clear()
                await self.changed.wait()
        return [f for f in self.frames if isinstance(f, cls)]

    @property
    def transcripts(self):
        return [f for f in self.frames if isinstance(f, TranscriptionFrame)]


@asynccontextmanager
async def harness(service, downstream=None):
    manager = TaskManager()
    setup = frame_processor_setup(manager, enable_metrics=True, enable_usage_metrics=True)
    before, after = Capture(), Capture()
    before.link(service)
    service.link(after)
    processors = [before, service, after]
    if downstream:
        after.link(downstream)
        processors.append(downstream)
    for processor in processors:
        await processor.setup(setup)
    await before.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
    # StartFrame has a priority path even when normal processing is direct.
    async with asyncio.timeout(2):
        while service._recognition.client is None or not after.frames:
            await asyncio.sleep(0)
    try:
        yield after, before
    finally:
        for processor in processors:
            await processor.cleanup()
        for task in list(manager.current_tasks()):
            await manager.cancel_task(task)


async def send(service, *frames, direction=FrameDirection.DOWNSTREAM):
    for frame in frames:
        await service.process_frame(frame, direction)


def audio(data=b"\x01\x00" * 160, rate=16000, channels=1):
    return InputAudioRawFrame(audio=data, sample_rate=rate, num_channels=channels)


async def next_stream(client):
    async with asyncio.timeout(2):
        return await client.new_stream.get()


async def wait(event):
    async with asyncio.timeout(2):
        await event.wait()
