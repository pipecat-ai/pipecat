# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import queue
from typing import Dict, Union, Optional

import pyaudio


class MicrophoneStream:
    """Opens a recording stream as responses yielding the audio chunks."""

    def __init__(self, rate: int, chunk: int, device: int = None) -> None:
        self._rate = rate
        self._chunk = chunk
        self._device = device

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            input_device_index=self._device,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def close(self) -> None:
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the responses to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def __exit__(self, type, value, traceback):
        self.close()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def __next__(self) -> bytes:
        if self.closed:
            raise StopIteration
        chunk = self._buff.get()
        if chunk is None:
            raise StopIteration
        data = [chunk]

        while True:
            try:
                chunk = self._buff.get(block=False)
                if chunk is None:
                    assert not self.closed
                data.append(chunk)
            except queue.Empty:
                break

        return b''.join(data)

    def __iter__(self):
        return self


def get_audio_device_info(device_id: int) -> Dict[str, Union[int, float, str]]:
    p = pyaudio.PyAudio()
    info = p.get_device_info_by_index(device_id)
    p.terminate()
    return info


def get_default_input_device_info() -> Optional[Dict[str, Union[int, float, str]]]:
    p = pyaudio.PyAudio()
    try:
        info = p.get_default_input_device_info()
    except OSError:
        info = None
    p.terminate()
    return info


def list_output_devices() -> None:
    p = pyaudio.PyAudio()
    print("Output audio devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxOutputChannels'] < 1:
            continue
        print(f"{info['index']}: {info['name']}")
    p.terminate()


def list_input_devices() -> None:
    p = pyaudio.PyAudio()
    print("Input audio devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] < 1:
            continue
        print(f"{info['index']}: {info['name']}")
    p.terminate()


class SoundCallBack:
    def __init__(
        self, output_device_index: Optional[int], sampwidth: int, nchannels: int, framerate: int,
    ) -> None:
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(
            output_device_index=output_device_index,
            format=self.pa.get_format_from_width(sampwidth),
            channels=nchannels,
            rate=framerate,
            output=True,
        )
        self.opened = True

    def __call__(self, audio_data: bytes, audio_length: float = None) -> None:
        self.stream.write(audio_data)

    def __enter__(self):
        return self

    def __exit__(self, type_, value, traceback) -> None:
        self.close()

    def close(self) -> None:
        self.stream.close()
        self.pa.terminate()
        self.opened = False
