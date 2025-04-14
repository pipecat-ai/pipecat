#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from loguru import logger

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADState
from pipecat.frames.frames import (
    BotInterruptionFrame,
    CancelFrame,
    EmulateUserStartedSpeakingFrame,
    EmulateUserStoppedSpeakingFrame,
    EndFrame,
    FilterUpdateSettingsFrame,
    Frame,
    InputAudioRawFrame,
    StartFrame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    SystemFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADParamsUpdateFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_transport import TransportParams


class BaseInputTransport(FrameProcessor):
    def __init__(self, params: TransportParams, **kwargs):
        super().__init__(**kwargs)

        self._params = params

        # Input sample rate. It will be initialized on StartFrame.
        self._sample_rate = 0

        # We read audio from a single queue one at a time and we then run VAD in
        # a thread. Therefore, only one thread should be necessary.
        self._executor = ThreadPoolExecutor(max_workers=1)

        # Task to process incoming audio (VAD) and push audio frames downstream
        # if passthrough is enabled.
        self._audio_task = None

    def enable_audio_in_stream_on_start(self, enabled: bool) -> None:
        logger.debug(f"Enabling audio on start. {enabled}")
        self._params.audio_in_stream_on_start = enabled

    def start_audio_in_streaming(self):
        pass

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def vad_analyzer(self) -> Optional[VADAnalyzer]:
        return self._params.vad_analyzer

    async def start(self, frame: StartFrame):
        self._sample_rate = self._params.audio_in_sample_rate or frame.audio_in_sample_rate

        # Configure VAD analyzer.
        if self._params.vad_enabled and self._params.vad_analyzer:
            self._params.vad_analyzer.set_sample_rate(self._sample_rate)
        # Start audio filter.
        if self._params.audio_in_filter:
            await self._params.audio_in_filter.start(self._sample_rate)
        # Create audio input queue and task if needed.
        if not self._audio_task and (self._params.audio_in_enabled or self._params.vad_enabled):
            self._audio_in_queue = asyncio.Queue()
            self._audio_task = self.create_task(self._audio_task_handler())

    async def stop(self, frame: EndFrame):
        # Cancel and wait for the audio input task to finish.
        if self._audio_task and (self._params.audio_in_enabled or self._params.vad_enabled):
            await self.cancel_task(self._audio_task)
            self._audio_task = None
        # Stop audio filter.
        if self._params.audio_in_filter:
            await self._params.audio_in_filter.stop()

    async def cancel(self, frame: CancelFrame):
        # Cancel and wait for the audio input task to finish.
        if self._audio_task and (self._params.audio_in_enabled or self._params.vad_enabled):
            await self.cancel_task(self._audio_task)
            self._audio_task = None

    async def push_audio_frame(self, frame: InputAudioRawFrame):
        if self._params.audio_in_enabled or self._params.vad_enabled:
            await self._audio_in_queue.put(frame)

    #
    # Frame processor
    #

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Specific system frames
        if isinstance(frame, StartFrame):
            # Push StartFrame before start(), because we want StartFrame to be
            # processed by every processor before any other frame is processed.
            await self.push_frame(frame, direction)
            await self.start(frame)
        elif isinstance(frame, CancelFrame):
            await self.cancel(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, BotInterruptionFrame):
            await self._handle_bot_interruption(frame)
        elif isinstance(frame, EmulateUserStartedSpeakingFrame):
            logger.debug("Emulating user started speaking")
            await self._handle_user_interruption(UserStartedSpeakingFrame(emulated=True))
        elif isinstance(frame, EmulateUserStoppedSpeakingFrame):
            logger.debug("Emulating user stopped speaking")
            await self._handle_user_interruption(UserStoppedSpeakingFrame(emulated=True))
        # All other system frames
        elif isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
        # Control frames
        elif isinstance(frame, EndFrame):
            # Push EndFrame before stop(), because stop() waits on the task to
            # finish and the task finishes when EndFrame is processed.
            await self.push_frame(frame, direction)
            await self.stop(frame)
        elif isinstance(frame, VADParamsUpdateFrame):
            if self.vad_analyzer:
                self.vad_analyzer.set_params(frame.params)
        elif isinstance(frame, FilterUpdateSettingsFrame) and self._params.audio_in_filter:
            await self._params.audio_in_filter.process_frame(frame)
        # Other frames
        else:
            await self.push_frame(frame, direction)

    #
    # Handle interruptions
    #

    async def _handle_bot_interruption(self, frame: BotInterruptionFrame):
        logger.debug("Bot interruption")
        if self.interruptions_allowed:
            await self._start_interruption()
            await self.push_frame(StartInterruptionFrame())

    async def _handle_user_interruption(self, frame: Frame):
        if isinstance(frame, UserStartedSpeakingFrame):
            logger.debug("User started speaking")
            await self.push_frame(frame)
            # Make sure we notify about interruptions quickly out-of-band.
            if self.interruptions_allowed:
                await self._start_interruption()
                # Push an out-of-band frame (i.e. not using the ordered push
                # frame task) to stop everything, specially at the output
                # transport.
                await self.push_frame(StartInterruptionFrame())
        elif isinstance(frame, UserStoppedSpeakingFrame):
            logger.debug("User stopped speaking")
            await self.push_frame(frame)
            if self.interruptions_allowed:
                await self._stop_interruption()
                await self.push_frame(StopInterruptionFrame())

    #
    # Audio input
    #

    async def _vad_analyze(self, audio_frame: InputAudioRawFrame) -> VADState:
        state = VADState.QUIET
        if self.vad_analyzer:
            state = await self.get_event_loop().run_in_executor(
                self._executor, self.vad_analyzer.analyze_audio, audio_frame.audio
            )
        return state

    async def _handle_vad(self, audio_frame: InputAudioRawFrame, vad_state: VADState):
        new_vad_state = await self._vad_analyze(audio_frame)
        if (
            new_vad_state != vad_state
            and new_vad_state != VADState.STARTING
            and new_vad_state != VADState.STOPPING
        ):
            frame = None
            if new_vad_state == VADState.SPEAKING:
                frame = UserStartedSpeakingFrame()
            elif new_vad_state == VADState.QUIET:
                frame = UserStoppedSpeakingFrame()

            if frame:
                await self._handle_user_interruption(frame)

            vad_state = new_vad_state
        return vad_state

    async def _audio_task_handler(self):
        vad_state: VADState = VADState.QUIET
        while True:
            frame: InputAudioRawFrame = await self._audio_in_queue.get()

            audio_passthrough = True

            # If an audio filter is available, run it before VAD.
            if self._params.audio_in_filter:
                frame.audio = await self._params.audio_in_filter.filter(frame.audio)

            # Check VAD and push event if necessary. We just care about
            # changes from QUIET to SPEAKING and vice versa.
            if self._params.vad_enabled:
                vad_state = await self._handle_vad(frame, vad_state)
                audio_passthrough = self._params.vad_audio_passthrough

            # Push audio downstream if passthrough.
            if audio_passthrough:
                await self.push_frame(frame)

            self._audio_in_queue.task_done()
