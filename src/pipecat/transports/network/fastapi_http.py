#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import asyncio
import json
import io
import wave

from typing import Awaitable, Callable
from pydantic.main import BaseModel

from pipecat.frames.frames import AudioRawFrame, CancelFrame, EndFrame, Frame, StartFrame, StartInterruptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

from loguru import logger

try:
    from fastapi import Request, Response
    from starlette.background import BackgroundTask
    from sse_starlette.sse import EventSourceResponse
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use FastAPI HTTP SSE, you need to `pip install pipecat-ai[http]`.")
    raise Exception(f"Missing module: {e}")


class FastAPIHTTPParams(TransportParams):
    serializer: FrameSerializer


class FastAPIHTTPInputTransport(BaseInputTransport):

    def __init__(
            self,
            params: FastAPIHTTPParams,
            **kwargs):
        super().__init__(params, **kwargs)

        self._params = params
        self._request = None

    # todo: this should probably expect a list of frames, not just one frame
    async def handle_request(self, request: Request):
        self._request = request
        frames_list = await request.json()
        logger.debug(f"Received frames: {frames_list}")
        for frame in frames_list:
            logger.debug(f"Received frame: {frame}")
            frame = self._params.serializer.deserialize(frame)
            if frame and isinstance(frame, AudioRawFrame):
                await self.push_audio_frame(frame)
            else:
                await self.push_frame(frame)


class FastAPIHTTPOutputTransport(BaseOutputTransport):

    def __init__(self, params: FastAPIHTTPParams, **kwargs):
        super().__init__(params, **kwargs)

        self._params = params
        self._event_queue = asyncio.Queue()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self._write_frame(frame)

    async def write_raw_audio_frames(self, frames: bytes):
        pass

    async def _write_frame(self, frame: Frame):
        payload = self._params.serializer.serialize(frame)
        await self._event_queue.put(payload)

    async def event_generator(self):
        while True:
            event = await self._event_queue.get()
            logger.debug(f"Sending event {event}")
            yield event


class FastAPIHTTPTransport(BaseTransport):

    def __init__(
            self,
            params: FastAPIHTTPParams,
            input_name: str | None = None,
            output_name: str | None = None,
            loop: asyncio.AbstractEventLoop | None = None):
        super().__init__(input_name=input_name, output_name=output_name, loop=loop)
        self._params = params
        self._request = None

        self._input = FastAPIHTTPInputTransport(
            self._params, name=self._input_name)
        self._output = FastAPIHTTPOutputTransport(
            self._params, name=self._output_name)

    def input(self) -> FrameProcessor:
        return self._input

    def output(self) -> FrameProcessor:
        return self._output

    async def handle_request(self, request: Request):
        self._request = request
        await self._input.handle_request(request)
        return EventSourceResponse(self._output.event_generator())
