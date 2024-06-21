#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tkinter as tk

from pipecat.frames.frames import AudioRawFrame, ImageRawFrame, StartFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

from loguru import logger

try:
    import pyaudio
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use local audio, you need to `pip install pipecat-ai[audio]`. On MacOS, you also need to `brew install portaudio`.")
    raise Exception(f"Missing module: {e}")

try:
    import tkinter as tk
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("tkinter missing. Try `apt install python3-tk` or `brew install python-tk@3.10`.")
    raise Exception(f"Missing module: {e}")


class TkInputTransport(BaseInputTransport):

    def __init__(self, py_audio: pyaudio.PyAudio, params: TransportParams):
        super().__init__(params)

        sample_rate = self._params.audio_in_sample_rate
        num_frames = int(sample_rate / 100) * 2  # 20ms of audio

        self._in_stream = py_audio.open(
            format=py_audio.get_format_from_width(2),
            channels=params.audio_in_channels,
            rate=params.audio_in_sample_rate,
            frames_per_buffer=num_frames,
            stream_callback=self._audio_in_callback,
            input=True)

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._in_stream.start_stream()

    async def cleanup(self):
        await super().cleanup()
        self._in_stream.stop_stream()
        # This is not very pretty (taken from PyAudio docs).
        while self._in_stream.is_active():
            await asyncio.sleep(0.1)
        self._in_stream.close()

    def _audio_in_callback(self, in_data, frame_count, time_info, status):
        frame = AudioRawFrame(audio=in_data,
                              sample_rate=self._params.audio_in_sample_rate,
                              num_channels=self._params.audio_in_channels)

        asyncio.run_coroutine_threadsafe(self.push_audio_frame(frame), self.get_event_loop())

        return (None, pyaudio.paContinue)


class TkOutputTransport(BaseOutputTransport):

    def __init__(self, tk_root: tk.Tk, py_audio: pyaudio.PyAudio, params: TransportParams):
        super().__init__(params)

        self._executor = ThreadPoolExecutor(max_workers=5)

        self._out_stream = py_audio.open(
            format=py_audio.get_format_from_width(2),
            channels=params.audio_out_channels,
            rate=params.audio_out_sample_rate,
            output=True)

        # Start with a neutral gray background.
        array = np.ones((1024, 1024, 3)) * 128
        data = f"P5 {1024} {1024} 255 ".encode() + array.astype(np.uint8).tobytes()
        photo = tk.PhotoImage(width=1024, height=1024, data=data, format="PPM")
        self._image_label = tk.Label(tk_root, image=photo)
        self._image_label.pack()

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._out_stream.start_stream()

    async def cleanup(self):
        await super().cleanup()
        self._out_stream.stop_stream()
        # This is not very pretty (taken from PyAudio docs).
        while self._out_stream.is_active():
            await asyncio.sleep(0.1)
        self._out_stream.close()

    async def write_raw_audio_frames(self, frames: bytes):
        await self.get_event_loop().run_in_executor(self._executor, self._out_stream.write, frames)

    async def write_frame_to_camera(self, frame: ImageRawFrame):
        self.get_event_loop().call_soon(self._write_frame_to_tk, frame)

    def _write_frame_to_tk(self, frame: ImageRawFrame):
        width = frame.size[0]
        height = frame.size[1]
        data = f"P6 {width} {height} 255 ".encode() + frame.image
        photo = tk.PhotoImage(
            width=width,
            height=height,
            data=data,
            format="PPM")
        self._image_label.config(image=photo)

        # This holds a reference to the photo, preventing it from being garbage
        # collected.
        self._image_label.image = photo


class TkLocalTransport(BaseTransport):

    def __init__(self, tk_root: tk.Tk, params: TransportParams):
        self._tk_root = tk_root
        self._params = params
        self._pyaudio = pyaudio.PyAudio()

        self._input: TkInputTransport | None = None
        self._output: TkOutputTransport | None = None

    #
    # BaseTransport
    #

    def input(self) -> FrameProcessor:
        if not self._input:
            self._input = TkInputTransport(self._pyaudio, self._params)
        return self._input

    def output(self) -> FrameProcessor:
        if not self._output:
            self._output = TkOutputTransport(self._tk_root, self._pyaudio, self._params)
        return self._output
