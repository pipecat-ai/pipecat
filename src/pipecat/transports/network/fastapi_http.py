#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio

from abc import abstractmethod
from typing import AsyncGenerator, Callable

from pipecat.frames.frames import AudioRawFrame, EndFrame, Frame, InputAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

from loguru import logger

try:
    from fastapi import Request, Response
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use FastAPI HTTP SSE, you need to `pip install pipecat-ai[http]`.")
    raise Exception(f"Missing module: {e}")


class FastAPIHTTPParams(TransportParams):
    serializer: FrameSerializer


class FastAPIHTTPInputTransport(BaseInputTransport):
    def __init__(
        self,
        generator: Callable[[Request], AsyncGenerator[str | bytes, None]],
        params: FastAPIHTTPParams,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self._generator = generator
        self._params = params

    async def handle_request(self, request: Request):
        async for data in self._generator(request):
            frame = self._params.serializer.deserialize(data)
            if not frame:
                continue

            if isinstance(frame, AudioRawFrame):
                await self.push_audio_frame(
                    InputAudioRawFrame(
                        audio=frame.audio,
                        sample_rate=frame.sample_rate,
                        num_channels=frame.num_channels,
                    )
                )
            else:
                await self.push_frame(frame)


class FastAPIHTTPOutputTransport(BaseOutputTransport):
    def __init__(self, params: FastAPIHTTPParams, **kwargs):
        super().__init__(params, **kwargs)

        self._params = params
        self._response_queue = asyncio.Queue()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._response_queue.put(None)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        payload = self._params.serializer.serialize(frame)
        if payload:
            await self._response_queue.put(payload)

    async def output_generator(self) -> AsyncGenerator[str | bytes, None]:
        running = True
        while running:
            data = await self._response_queue.get()
            running = data is not None
            if data:
                yield data


class FastAPIHTTPTransport(BaseTransport):
    def __init__(
        self,
        params: FastAPIHTTPParams,
        input_name: str | None = None,
        output_name: str | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        super().__init__(input_name=input_name, output_name=output_name, loop=loop)
        self._params = params

        self._input = FastAPIHTTPInputTransport(
            generator=self.input_generator, params=self._params, name=self._input_name
        )
        self._output = FastAPIHTTPOutputTransport(params=self._params, name=self._output_name)

    def input(self) -> FastAPIHTTPInputTransport:
        return self._input

    def output(self) -> FastAPIHTTPOutputTransport:
        return self._output

    @abstractmethod
    def input_generator(self, request: Request) -> AsyncGenerator[str | bytes, None]:
        pass

    @abstractmethod
    async def handle_request(self, request: Request) -> Response:
        pass
