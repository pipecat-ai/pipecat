import asyncio
import numpy as np
import tkinter as tk

from dailyai.transports.threaded_transport import ThreadedTransport

try:
    import pyaudio
except ModuleNotFoundError as e:
    print(f"Exception: {e}")
    print(
        "In order to use the local transport, you need to `pip install dailyai[local]`. On MacOS, you also need to `brew install portaudio`.")
    raise Exception(f"Missing module: {e}")


class LocalTransport(ThreadedTransport):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._sample_width = kwargs.get("sample_width") or 2
        self._n_channels = kwargs.get("n_channels") or 1
        self._tk_root = kwargs.get("tk_root") or None
        self._pyaudio = None

        if self._camera_enabled and not self._tk_root:
            raise ValueError(
                "If camera is enabled, a tkinter root must be provided")

        if self._speaker_enabled:
            self._speaker_buffer_pending = bytearray()

    async def _write_frame_to_tkinter(self, frame: bytes):
        data = f"P6 {self._camera_width} {self._camera_height} 255 ".encode() + \
            frame
        photo = tk.PhotoImage(
            width=self._camera_width,
            height=self._camera_height,
            data=data,
            format="PPM")
        self._image_label.config(image=photo)

        # This holds a reference to the photo, preventing it from being garbage
        # collected.
        self._image_label.image = photo  # type: ignore

    def write_frame_to_camera(self, frame: bytes):
        if self._camera_enabled and self._loop:
            asyncio.run_coroutine_threadsafe(
                self._write_frame_to_tkinter(frame), self._loop
            )

    def write_frame_to_mic(self, frame: bytes):
        if self._mic_enabled:
            self._audio_stream.write(frame)

    def read_audio_frames(self, desired_frame_count):
        bytes = b""
        if self._speaker_enabled:
            bytes = self._speaker_stream.read(
                desired_frame_count,
                exception_on_overflow=False,
            )
        return bytes

    def _prerun(self):
        if self._mic_enabled:
            if not self._pyaudio:
                self._pyaudio = pyaudio.PyAudio()
            self._audio_stream = self._pyaudio.open(
                format=self._pyaudio.get_format_from_width(self._sample_width),
                channels=self._n_channels,
                rate=self._speaker_sample_rate,
                output=True,
            )

        if self._camera_enabled:
            # Start with a neutral gray background.
            array = np.ones((1024, 1024, 3)) * 128
            data = f"P5 {1024} {1024} 255 ".encode(
            ) + array.astype(np.uint8).tobytes()
            photo = tk.PhotoImage(
                width=1024,
                height=1024,
                data=data,
                format="PPM")
            self._image_label = tk.Label(self._tk_root, image=photo)
            self._image_label.pack()

        if self._speaker_enabled:
            if not self._pyaudio:
                self._pyaudio = pyaudio.PyAudio()
            self._speaker_stream = self._pyaudio.open(
                format=self._pyaudio.get_format_from_width(self._sample_width),
                channels=self._n_channels,
                rate=self._speaker_sample_rate,
                frames_per_buffer=self._speaker_sample_rate,
                input=True
            )
