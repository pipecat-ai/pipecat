#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio

import numpy as np
import tkinter as tk

from pipecat.frames.frames import ImageRawFrame
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

        self._in_stream = py_audio.open(
            format=py_audio.get_format_from_width(2),
            channels=params.audio_in_channels,
            rate=params.audio_in_sample_rate,
            frames_per_buffer=params.audio_in_sample_rate,
            input=True)

    def read_raw_audio_frames(self, frame_count: int) -> bytes:
        return self._in_stream.read(frame_count, exception_on_overflow=False)

    async def cleanup(self):
        # This is not very pretty (taken from PyAudio docs).
        while self._in_stream.is_active():
            await asyncio.sleep(0.1)
        self._in_stream.close()

        await super().cleanup()


class TkOutputTransport(BaseOutputTransport):

    def __init__(self, tk_root: tk.Tk, py_audio: pyaudio.PyAudio, params: TransportParams):
        super().__init__(params)

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

    def write_raw_audio_frames(self, frames: bytes):
        self._out_stream.write(frames)

    def write_frame_to_camera(self, frame: ImageRawFrame):
        asyncio.run_coroutine_threadsafe(self._write_frame_to_tk(frame), self.get_event_loop())

    async def cleanup(self):
        # This is not very pretty (taken from PyAudio docs).
        while self._out_stream.is_active():
            await asyncio.sleep(0.1)
        self._out_stream.close()

        await super().cleanup()

    async def _write_frame_to_tk(self, frame: ImageRawFrame):
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
