#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import queue

from concurrent.futures import ThreadPoolExecutor

from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    StartFrame,
    EndFrame,
    Frame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame)
from pipecat.transports.base_transport import TransportParams
from pipecat.vad.vad_analyzer import VADAnalyzer, VADState

from loguru import logger


class BaseInputTransport(FrameProcessor):

    def __init__(self, params: TransportParams):
        super().__init__()

        self._params = params

        self._running = False
        self._allow_interruptions = False

        self._executor = ThreadPoolExecutor(max_workers=5)

        # Create push frame task. This is the task that will push frames in
        # order. We also guarantee that all frames are pushed in the same task.
        self._create_push_task()

    async def start(self, frame: StartFrame):
        # Make sure we have the latest params. Note that this transport might
        # have been started on another task that might not need interruptions,
        # for example.
        self._allow_interruptions = frame.allow_interruptions

        if self._running:
            return

        self._running = True

        # Create audio input queue and thread if needed.
        if self._params.audio_in_enabled or self._params.vad_enabled:
            self._audio_in_queue = queue.Queue()
            self._audio_thread = self._loop.run_in_executor(
                self._executor, self._audio_thread_handler)

    async def stop(self):
        if not self._running:
            return

        # This will exit all threads.
        self._running = False

        # Wait for the threads to finish.
        if self._params.audio_in_enabled or self._params.vad_enabled:
            await self._audio_thread

        self._push_frame_task.cancel()

    def vad_analyzer(self) -> VADAnalyzer | None:
        return self._params.vad_analyzer

    def push_audio_frame(self, frame: AudioRawFrame):
        self._audio_in_queue.put_nowait(frame)

    #
    # Frame processor
    #

    async def cleanup(self):
        pass

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, CancelFrame):
            # We don't queue a CancelFrame since we want to stop ASAP.
            await self.push_frame(frame, direction)
            await self.stop()
        elif isinstance(frame, StartFrame):
            self._allow_interruption = frame.allow_interruptions
            await self.start(frame)
            await self._internal_push_frame(frame, direction)
        elif isinstance(frame, EndFrame):
            await self._internal_push_frame(frame, direction)
            await self.stop()
        else:
            await self._internal_push_frame(frame, direction)

    #
    # Push frames task
    #

    def _create_push_task(self):
        loop = self.get_event_loop()
        self._push_frame_task = loop.create_task(self._push_frame_task_handler())
        self._push_queue = asyncio.Queue()

    async def _internal_push_frame(
            self,
            frame: Frame | None,
            direction: FrameDirection | None = FrameDirection.DOWNSTREAM):
        await self._push_queue.put((frame, direction))

    async def _push_frame_task_handler(self):
        while True:
            try:
                (frame, direction) = await self._push_queue.get()
                await self.push_frame(frame, direction)
            except asyncio.CancelledError:
                break

    #
    # Handle interruptions
    #

    async def _handle_interruptions(self, frame: Frame):
        if self._allow_interruptions:
            # Make sure we notify about interruptions quickly out-of-band
            if isinstance(frame, UserStartedSpeakingFrame):
                logger.debug("User started speaking")
                self._push_frame_task.cancel()
                self._create_push_task()
                await self.push_frame(StartInterruptionFrame())
            elif isinstance(frame, UserStoppedSpeakingFrame):
                logger.debug("User stopped speaking")
                await self.push_frame(StopInterruptionFrame())
        await self._internal_push_frame(frame)

    #
    # Audio input
    #

    def _vad_analyze(self, audio_frames: bytes) -> VADState:
        state = VADState.QUIET
        vad_analyzer = self.vad_analyzer()
        if vad_analyzer:
            state = vad_analyzer.analyze_audio(audio_frames)
        return state

    def _handle_vad(self, audio_frames: bytes, vad_state: VADState):
        new_vad_state = self._vad_analyze(audio_frames)
        if new_vad_state != vad_state and new_vad_state != VADState.STARTING and new_vad_state != VADState.STOPPING:
            frame = None
            if new_vad_state == VADState.SPEAKING:
                frame = UserStartedSpeakingFrame()
            elif new_vad_state == VADState.QUIET:
                frame = UserStoppedSpeakingFrame()

            if frame:
                future = asyncio.run_coroutine_threadsafe(
                    self._handle_interruptions(frame), self.get_event_loop())
                future.result()

            vad_state = new_vad_state
        return vad_state

    def _audio_thread_handler(self):
        vad_state: VADState = VADState.QUIET
        while self._running:
            try:
                frame: AudioRawFrame = self._audio_in_queue.get(timeout=1)

                audio_passthrough = True

                # Check VAD and push event if necessary. We just care about
                # changes from QUIET to SPEAKING and vice versa.
                if self._params.vad_enabled:
                    vad_state = self._handle_vad(frame.audio, vad_state)
                    audio_passthrough = self._params.vad_audio_passthrough

                    # Push audio downstream if passthrough.
                if audio_passthrough:
                    future = asyncio.run_coroutine_threadsafe(
                        self._internal_push_frame(frame), self._loop)
                    future.result()
            except queue.Empty:
                pass
            except BaseException as e:
                logger.error(f"Error reading audio frames: {e}")
