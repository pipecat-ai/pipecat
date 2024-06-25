#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio

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

    def __init__(self, params: TransportParams, **kwargs):
        super().__init__(**kwargs)

        self._params = params

        self._executor = ThreadPoolExecutor(max_workers=5)

        # Create push frame task. This is the task that will push frames in
        # order. We also guarantee that all frames are pushed in the same task.
        self._create_push_task()

    async def start(self, frame: StartFrame):
        # Create audio input queue and task if needed.
        if self._params.audio_in_enabled or self._params.vad_enabled:
            self._audio_in_queue = asyncio.Queue()
            self._audio_task = self.get_event_loop().create_task(self._audio_task_handler())

    async def stop(self):
        # Wait for the task to finish.
        if self._params.audio_in_enabled or self._params.vad_enabled:
            self._audio_task.cancel()
            await self._audio_task

    def vad_analyzer(self) -> VADAnalyzer | None:
        return self._params.vad_analyzer

    async def push_audio_frame(self, frame: AudioRawFrame):
        if self._params.audio_in_enabled or self._params.vad_enabled:
            await self._audio_in_queue.put(frame)

    #
    # Frame processor
    #

    async def cleanup(self):
        self._push_frame_task.cancel()
        await self._push_frame_task

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, CancelFrame):
            await self.stop()
            # We don't queue a CancelFrame since we want to stop ASAP.
            await self.push_frame(frame, direction)
        elif isinstance(frame, StartFrame):
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
        self._push_queue = asyncio.Queue()
        self._push_frame_task = loop.create_task(self._push_frame_task_handler())

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
        if self.interruptions_allowed:
            # Make sure we notify about interruptions quickly out-of-band
            if isinstance(frame, UserStartedSpeakingFrame):
                logger.debug("User started speaking")
                # Cancel the task. This will stop pushing frames downstream.
                self._push_frame_task.cancel()
                await self._push_frame_task
                # Push an out-of-band frame (i.e. not using the ordered push
                # frame task) to stop everything, specially at the output
                # transport.
                await self.push_frame(StartInterruptionFrame())
                # Create a new queue and task.
                self._create_push_task()
            elif isinstance(frame, UserStoppedSpeakingFrame):
                logger.debug("User stopped speaking")
                await self.push_frame(StopInterruptionFrame())
        await self._internal_push_frame(frame)

    #
    # Audio input
    #

    async def _vad_analyze(self, audio_frames: bytes) -> VADState:
        state = VADState.QUIET
        vad_analyzer = self.vad_analyzer()
        if vad_analyzer:
            state = await self.get_event_loop().run_in_executor(
                self._executor, vad_analyzer.analyze_audio, audio_frames)
        return state

    async def _handle_vad(self, audio_frames: bytes, vad_state: VADState):
        new_vad_state = await self._vad_analyze(audio_frames)
        if new_vad_state != vad_state and new_vad_state != VADState.STARTING and new_vad_state != VADState.STOPPING:
            frame = None
            if new_vad_state == VADState.SPEAKING:
                frame = UserStartedSpeakingFrame()
            elif new_vad_state == VADState.QUIET:
                frame = UserStoppedSpeakingFrame()

            if frame:
                await self._handle_interruptions(frame)

            vad_state = new_vad_state
        return vad_state

    async def _audio_task_handler(self):
        vad_state: VADState = VADState.QUIET
        while True:
            try:
                frame: AudioRawFrame = await self._audio_in_queue.get()

                audio_passthrough = True

                # Check VAD and push event if necessary. We just care about
                # changes from QUIET to SPEAKING and vice versa.
                if self._params.vad_enabled:
                    vad_state = await self._handle_vad(frame.audio, vad_state)
                    audio_passthrough = self._params.vad_audio_passthrough

                # Push audio downstream if passthrough.
                if audio_passthrough:
                    await self._internal_push_frame(frame)
            except asyncio.CancelledError:
                break
            except BaseException as e:
                logger.error(f"{self} error reading audio frames: {e}")
