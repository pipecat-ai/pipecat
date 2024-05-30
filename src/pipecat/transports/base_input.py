#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio

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
from pipecat.vad.vad_analyzer import VADState

from loguru import logger


class BaseInputTransport(FrameProcessor):

    def __init__(self, params: TransportParams):
        super().__init__()

        self._params = params

        self._allow_interruptions = False

        # Create audio input queue if needed.
        if self._params.audio_in_enabled or self._params.vad_enabled:
            self._audio_in_queue = asyncio.Queue()

        # Create push frame task. This is the task that will push frames in
        # order. We also guarantee that all frames are pushed in the same task.
        self._create_push_task()

    async def start(self, frame: StartFrame):
        # Make sure we have the latest params. Note that this transport might
        # have been started on another task that might not need interruptions,
        # for example.
        self._allow_interruptions = frame.allow_interruptions

        if self._params.audio_in_enabled or self._params.vad_enabled:
            loop = self.get_event_loop()
            self._audio_task = loop.create_task(self._audio_task_handler())

    async def stop(self):
        # Wait for the tasks to finish.
        if self._params.audio_in_enabled or self._params.vad_enabled:
            self._audio_task.cancel()

        self._push_frame_task.cancel()

    async def vad_analyze(self, audio_frames: bytes) -> VADState:
        pass

    async def read_raw_audio_frames(self, frame_count: int) -> bytes:
        pass

    #
    # Frame processor
    #

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, CancelFrame):
            await self.stop()
            # We don't queue a CancelFrame since we want to stop ASAP.
            await self.push_frame(frame, direction)
        elif isinstance(frame, StartFrame):
            self._allow_interruption = frame.allow_interruptions
            await self.start(frame)
            await self._internal_push_frame(frame, direction)
        elif isinstance(frame, EndFrame):
            await self.stop()
            await self._internal_push_frame(frame, direction)
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

    async def _handle_vad(self, audio_frames: bytes, vad_state: VADState):
        new_vad_state = await self.vad_analyze(audio_frames)
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

        sample_rate = self._params.audio_in_sample_rate
        num_channels = self._params.audio_in_channels
        num_frames = int(sample_rate / 100)  # 10ms of audio

        while True:
            try:
                audio_frames = await self.read_raw_audio_frames(num_frames)
                if len(audio_frames) > 0:
                    frame = AudioRawFrame(
                        audio=audio_frames,
                        sample_rate=sample_rate,
                        num_channels=num_channels)

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
                logger.error(f"Error reading audio frames: {e}")
