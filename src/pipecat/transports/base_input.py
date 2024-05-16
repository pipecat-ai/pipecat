#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import queue

from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    StartFrame,
    EndFrame,
    Frame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame)
from pipecat.transports.base_transport import TransportParams
from pipecat.vad.vad_analyzer import VADState

from loguru import logger


class BaseInputTransport(FrameProcessor):

    def __init__(self, params: TransportParams):
        super().__init__()

        self._params = params

        self._running = False

        # Start media threads.
        if self._params.audio_in_enabled or self._params.vad_enabled:
            self._audio_in_queue = queue.Queue()

    async def start(self):
        if self._running:
            return

        self._running = True

        if self._params.audio_in_enabled or self._params.vad_enabled:
            loop = self.get_event_loop()
            self._audio_in_thread = loop.run_in_executor(None, self._audio_in_thread_handler)
            self._audio_out_thread = loop.run_in_executor(None, self._audio_out_thread_handler)

    async def stop(self):
        if not self._running:
            return

        # This will exit all threads.
        self._running = False

        # Wait for the threads to finish.
        if self._params.audio_in_enabled or self._params.vad_enabled:
            await self._audio_in_thread
            await self._audio_out_thread

    def vad_analyze(self, audio_frames: bytes) -> VADState:
        pass

    def read_raw_audio_frames(self, frame_count: int) -> bytes:
        pass

    #
    # Frame processor
    #

    async def cleanup(self):
        pass

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, StartFrame):
            await self.start()
            await self.push_frame(frame, direction)
        elif isinstance(frame, CancelFrame) or isinstance(frame, EndFrame):
            await self.stop()
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    #
    # Audio input
    #

    def _handle_vad(self, audio_frames: bytes, vad_state: VADState):
        new_vad_state = self.vad_analyze(audio_frames)
        if new_vad_state != vad_state and new_vad_state != VADState.STARTING and new_vad_state != VADState.STOPPING:
            frame = None
            if new_vad_state == VADState.SPEAKING:
                frame = UserStartedSpeakingFrame()
            elif new_vad_state == VADState.QUIET:
                frame = UserStoppedSpeakingFrame()
            if frame:
                future = asyncio.run_coroutine_threadsafe(
                    self.push_frame(frame), self.get_event_loop())
                future.result()
                vad_state = new_vad_state
        return vad_state

    def _audio_in_thread_handler(self):
        sample_rate = self._params.audio_in_sample_rate
        num_channels = self._params.audio_in_channels
        num_frames = int(sample_rate / 100)  # 10ms of audio
        while self._running:
            try:
                audio_frames = self.read_raw_audio_frames(num_frames)
                if len(audio_frames) > 0:
                    frame = AudioRawFrame(
                        audio=audio_frames,
                        sample_rate=sample_rate,
                        num_channels=num_channels)
                    self._audio_in_queue.put(frame)
            except BaseException as e:
                logger.error(f"Error reading audio frames: {e}")

    def _audio_out_thread_handler(self):
        vad_state: VADState = VADState.QUIET
        while self._running:
            try:
                frame = self._audio_in_queue.get(timeout=1)

                audio_passthrough = True

                # Check VAD and push event if necessary. We just care about changes
                # from QUIET to SPEAKING and vice versa.
                if self._params.vad_enabled:
                    vad_state = self._handle_vad(frame.audio, vad_state)
                    audio_passthrough = self._params.vad_audio_passthrough

                # Push audio downstream if passthrough.
                if audio_passthrough:
                    future = asyncio.run_coroutine_threadsafe(
                        self.push_frame(frame), self.get_event_loop())
                    future.result()
            except queue.Empty:
                pass
            except BaseException as e:
                logger.error(f"Error pushing audio frames: {e}")
