#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from loguru import logger

from pipecat.frames.frames import InputAudioRawFrame, StartFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

try:
    import pyaudio
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use local audio, you need to `pip install pipecat-ai[local]`. On MacOS, you also need to `brew install portaudio`."
    )
    raise Exception(f"Missing module: {e}")


class LocalAudioTransportParams(TransportParams):
    input_device_index: Optional[int] = None
    output_device_index: Optional[int] = None


class LocalAudioInputTransport(BaseInputTransport):
    _params: LocalAudioTransportParams

    def __init__(self, py_audio: pyaudio.PyAudio, params: LocalAudioTransportParams):
        super().__init__(params)
        self._py_audio = py_audio

        self._in_stream = None
        self._sample_rate = 0

    async def start(self, frame: StartFrame):
        await super().start(frame)

        if self._in_stream:
            return

        self._sample_rate = self._params.audio_in_sample_rate or frame.audio_in_sample_rate
        num_frames = int(self._sample_rate / 100) * 2  # 20ms of audio

        self._in_stream = self._py_audio.open(
            format=self._py_audio.get_format_from_width(2),
            channels=self._params.audio_in_channels,
            rate=self._sample_rate,
            frames_per_buffer=num_frames,
            stream_callback=self._audio_in_callback,
            input=True,
            input_device_index=self._params.input_device_index,
        )
        self._in_stream.start_stream()

        await self.set_transport_ready(frame)

    async def cleanup(self):
        await super().cleanup()
        if self._in_stream:
            self._in_stream.stop_stream()
            self._in_stream.close()
            self._in_stream = None

    def _audio_in_callback(self, in_data, frame_count, time_info, status):
        frame = InputAudioRawFrame(
            audio=in_data,
            sample_rate=self._sample_rate,
            num_channels=self._params.audio_in_channels,
        )

        asyncio.run_coroutine_threadsafe(self.push_audio_frame(frame), self.get_event_loop())

        return (None, pyaudio.paContinue)


class LocalAudioOutputTransport(BaseOutputTransport):
    _params: LocalAudioTransportParams

    def __init__(self, py_audio: pyaudio.PyAudio, params: LocalAudioTransportParams):
        super().__init__(params)
        self._py_audio = py_audio

        self._out_stream = None
        self._sample_rate = 0

        # We only write audio frames from a single task, so only one thread
        # should be necessary.
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def start(self, frame: StartFrame):
        await super().start(frame)

        if self._out_stream:
            return

        self._sample_rate = self._params.audio_out_sample_rate or frame.audio_out_sample_rate

        self._out_stream = self._py_audio.open(
            format=self._py_audio.get_format_from_width(2),
            channels=self._params.audio_out_channels,
            rate=self._sample_rate,
            output=True,
            output_device_index=self._params.output_device_index,
        )
        self._out_stream.start_stream()

        await self.set_transport_ready(frame)

    async def cleanup(self):
        await super().cleanup()
        if self._out_stream:
            self._out_stream.stop_stream()
            self._out_stream.close()
            self._out_stream = None

    async def write_raw_audio_frames(self, frames: bytes, destination: Optional[str] = None):
        if self._out_stream:
            await self.get_event_loop().run_in_executor(
                self._executor, self._out_stream.write, frames
            )


class LocalAudioTransport(BaseTransport):
    def __init__(self, params: LocalAudioTransportParams):
        super().__init__()
        self._params = params
        self._pyaudio = pyaudio.PyAudio()

        self._input: Optional[LocalAudioInputTransport] = None
        self._output: Optional[LocalAudioOutputTransport] = None

    #
    # BaseTransport
    #

    def input(self) -> FrameProcessor:
        if not self._input:
            self._input = LocalAudioInputTransport(self._pyaudio, self._params)
        return self._input

    def output(self) -> FrameProcessor:
        if not self._output:
            self._output = LocalAudioOutputTransport(self._pyaudio, self._params)
        return self._output
