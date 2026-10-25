#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Local audio transport implementation for Pipecat.

This module provides a local audio transport that uses PyAudio for real-time
audio input and output through the system's default audio devices.
"""

import asyncio
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from loguru import logger

from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame, StartFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameProcessorSetup
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.local._playback_clock import PlaybackClock

try:
    import pyaudio
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        'In order to use local audio, you need to `uv add "pipecat-ai[local]"`. On MacOS, you also need to `brew install portaudio`.'
    )
    raise ImportError(f"Missing module: {e}") from e


class LocalAudioTransportParams(TransportParams):
    """Configuration parameters for local audio transport.

    Parameters:
        input_device_index: PyAudio device index for audio input. If None, uses default.
        output_device_index: PyAudio device index for audio output. If None, uses default.
        audio_out_playback_clock: Use callback DAC timing for playback reports.
            Measures samples rendered by PortAudio, not physical audibility.
    """

    input_device_index: int | None = None
    output_device_index: int | None = None
    audio_out_playback_clock: bool = False


class LocalAudioInputTransport(BaseInputTransport):
    """Local audio input transport using PyAudio.

    Captures audio from the system's audio input device and converts it to
    InputAudioRawFrame objects for processing in the pipeline.
    """

    _params: LocalAudioTransportParams

    def __init__(self, py_audio: pyaudio.PyAudio, params: LocalAudioTransportParams):
        """Initialize the local audio input transport.

        Args:
            py_audio: PyAudio instance for audio device management.
            params: Transport configuration parameters.
        """
        super().__init__(params)
        self._py_audio = py_audio

        self._in_stream = None
        self._sample_rate = 0

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the transport and open the audio input stream.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)

        num_frames = int(self.sample_rate / 100) * 2  # 20ms of audio

        self._in_stream = self._py_audio.open(
            format=self._py_audio.get_format_from_width(2),
            channels=self._params.audio_in_channels,
            rate=self._sample_rate,
            frames_per_buffer=num_frames,
            stream_callback=self._audio_in_callback,
            input=True,
            input_device_index=self._params.input_device_index,
        )

    async def cleanup(self):
        """Stop and cleanup the audio input stream."""
        await super().cleanup()
        if self._in_stream:
            self._in_stream.stop_stream()
            self._in_stream.close()
            self._in_stream = None

    async def start(self, frame: StartFrame):
        """Start the audio input stream.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)

        if self._in_stream:
            self._in_stream.start_stream()
            await self.set_transport_ready(frame)

    def _audio_in_callback(self, in_data, frame_count, time_info, status):
        """Callback function for PyAudio input stream."""
        frame = InputAudioRawFrame(
            audio=in_data,
            sample_rate=self._sample_rate,
            num_channels=self._params.audio_in_channels,
        )

        asyncio.run_coroutine_threadsafe(self.push_audio_frame(frame), self.get_event_loop())

        return (None, pyaudio.paContinue)


class LocalAudioOutputTransport(BaseOutputTransport):
    """Local audio output transport using PyAudio.

    Plays audio frames through the system's audio output device by converting
    OutputAudioRawFrame objects to playable audio data.
    """

    _params: LocalAudioTransportParams

    def __init__(self, py_audio: pyaudio.PyAudio, params: LocalAudioTransportParams):
        """Initialize the local audio output transport.

        Args:
            py_audio: PyAudio instance for audio device management.
            params: Transport configuration parameters.
        """
        super().__init__(params)
        self._py_audio = py_audio

        self._out_stream = None
        self._sample_rate = 0
        self._callback_lock = threading.Lock()
        self._callback_queue = deque()
        self._playback_clock = PlaybackClock()

        # We only write audio frames from a single task, so only one thread
        # should be necessary.
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the transport and open the audio output stream.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)

        self._out_stream = self._py_audio.open(
            format=self._py_audio.get_format_from_width(2),
            channels=self._params.audio_out_channels,
            rate=self.sample_rate,
            output=True,
            output_device_index=self._params.output_device_index,
            **(
                {
                    "stream_callback": self._audio_out_callback,
                    "frames_per_buffer": self.audio_chunk_size
                    // (2 * self._params.audio_out_channels),
                }
                if self._params.audio_out_playback_clock
                else {}
            ),
        )

    async def cleanup(self):
        """Stop and cleanup the audio output stream."""
        try:
            await super().cleanup()
            if self._out_stream:
                self._out_stream.stop_stream()
                self._out_stream.close()
                self._out_stream = None
        finally:
            self._executor.shutdown(wait=False)

    async def start(self, frame: StartFrame):
        """Start the audio output stream.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)

        if self._out_stream:
            self._out_stream.start_stream()
            await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Write an audio frame to the output stream.

        Args:
            frame: The audio frame to write to the output device.

        Returns:
            True if the audio frame was written successfully, False otherwise.
        """
        if self._out_stream and self._params.audio_out_playback_clock:
            completed = self.get_event_loop().create_future()
            with self._callback_lock:
                self._callback_queue.append([frame, 0, completed])
            return await completed
        if self._out_stream:
            await self.get_event_loop().run_in_executor(
                self._executor, self._out_stream.write, frame.audio
            )
            return True
        return False

    async def stop_audio_output(self, context_id: str | None, *, cancel: bool = False) -> None:
        """Remove unscheduled callback samples while retaining uninterruptible audio."""
        if not self._params.audio_out_playback_clock:
            return
        with self._callback_lock:
            retained = deque()
            for item in self._callback_queue:
                frame, _, completed = item
                if (context_id is None or frame.context_id == context_id) and (
                    cancel or frame.interruptible
                ):
                    if not completed.done():
                        completed.set_result(False)
                else:
                    retained.append(item)
            self._callback_queue = retained

    def _audio_out_callback(self, in_data, frame_count, time_info, status):
        """Supply PCM and record valid speech samples at their callback DAC times."""
        width = self._params.audio_out_channels * 2
        required = frame_count * width
        output = bytearray()
        try:
            with self._callback_lock:
                while len(output) < required and self._callback_queue:
                    frame, offset, completed = self._callback_queue[0]
                    count = min(required - len(output), len(frame.audio) - offset)
                    valid_bytes = frame.metadata.get("speech_audio_bytes", len(frame.audio))
                    speech = min(count, max(0, valid_bytes - offset))
                    if frame.context_id and speech:
                        dac = time_info.get("output_buffer_dac_time", 0.0)
                        self._playback_clock.schedule(
                            frame.context_id,
                            dac + len(output) / (width * self.sample_rate),
                            speech / (width * self.sample_rate),
                            valid=not status,
                        )
                    output.extend(frame.audio[offset : offset + count])
                    self._callback_queue[0][1] += count
                    if offset + count == len(frame.audio):
                        self._callback_queue.popleft()
                        self.get_event_loop().call_soon_threadsafe(
                            self._complete_callback_write, completed
                        )
        except Exception:
            return (bytes(required), pyaudio.paAbort)
        output.extend(bytes(required - len(output)))
        return (bytes(output), pyaudio.paContinue)

    @staticmethod
    def _complete_callback_write(completed):
        if not completed.done():
            completed.set_result(True)

    async def get_audio_playback_ms(self, context_id: str) -> float | None:
        """Measure rendered samples with PortAudio's stream clock and DAC schedule."""
        if not self._params.audio_out_playback_clock or self._out_stream is None:
            return None
        with self._callback_lock:
            end = self._playback_clock.end(context_id)
        # The callback can run ahead of the DAC. Its final sample must pass the
        # device clock before a completed report can name it as played.
        async with asyncio.timeout(self._params.audio_out_write_timeout_secs):
            while self._out_stream.get_time() < end:
                await asyncio.sleep(0.005)
        with self._callback_lock:
            played = self._playback_clock.played_ms(context_id, self._out_stream.get_time())
            self._playback_clock.forget(context_id)
        return played


class LocalAudioTransport(BaseTransport):
    """Complete local audio transport with input and output capabilities.

    Provides a unified interface for local audio I/O using PyAudio, supporting
    both audio capture and playback through the system's audio devices.
    """

    def __init__(self, params: LocalAudioTransportParams):
        """Initialize the local audio transport.

        Args:
            params: Transport configuration parameters.
        """
        super().__init__()
        self._params = params
        self._pyaudio = pyaudio.PyAudio()

        self._input: LocalAudioInputTransport | None = None
        self._output: LocalAudioOutputTransport | None = None

    #
    # BaseTransport
    #

    def input(self) -> FrameProcessor:
        """Get the input frame processor for this transport.

        Returns:
            The audio input transport processor.
        """
        if not self._input:
            self._input = LocalAudioInputTransport(self._pyaudio, self._params)
        return self._input

    def output(self) -> FrameProcessor:
        """Get the output frame processor for this transport.

        Returns:
            The audio output transport processor.
        """
        if not self._output:
            self._output = LocalAudioOutputTransport(self._pyaudio, self._params)
        return self._output
