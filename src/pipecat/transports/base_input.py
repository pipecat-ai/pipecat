#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from concurrent.futures import ThreadPoolExecutor

from loguru import logger

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADState
from pipecat.frames.frames import (
    BotInterruptionFrame,
    CancelFrame,
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

        self._executor = ThreadPoolExecutor(max_workers=5)

        # Task to process incoming audio (VAD) and push audio frames downstream
        # if passthrough is enabled.
        self._audio_task = None

    async def start(self, frame: StartFrame):
        # Start audio filter.
        if self._params.audio_in_filter:
            await self._params.audio_in_filter.start(self._params.audio_in_sample_rate)
        # Create audio input queue and task if needed.
        if self._params.audio_in_enabled or self._params.vad_enabled:
            self._audio_in_queue = asyncio.Queue()
            self._audio_task = self.get_event_loop().create_task(self._audio_task_handler())

    async def stop(self, frame: EndFrame):
        # Cancel and wait for the audio input task to finish.
        if self._audio_task and (self._params.audio_in_enabled or self._params.vad_enabled):
            self._audio_task.cancel()
            await self._audio_task
            self._audio_task = None
        # Stop audio filter.
        if self._params.audio_in_filter:
            await self._params.audio_in_filter.stop()

    async def cancel(self, frame: CancelFrame):
        # Cancel and wait for the audio input task to finish.
        if self._audio_task and (self._params.audio_in_enabled or self._params.vad_enabled):
            self._audio_task.cancel()
            await self._audio_task
            self._audio_task = None

    def vad_analyzer(self) -> VADAnalyzer | None:
        return self._params.vad_analyzer

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
            logger.debug("Bot interruption")
            await self._start_interruption()
            await self.push_frame(StartInterruptionFrame())
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
            vad_analyzer = self.vad_analyzer()
            if vad_analyzer:
                vad_analyzer.set_params(frame.params)
        elif isinstance(frame, FilterUpdateSettingsFrame) and self._params.audio_in_filter:
            await self._params.audio_in_filter.process_frame(frame)
        # Other frames
        else:
            await self.push_frame(frame, direction)

    #
    # Handle interruptions
    #

    async def _handle_interruptions(self, frame: Frame):
        if self.interruptions_allowed:
            # Make sure we notify about interruptions quickly out-of-band.
            if isinstance(frame, UserStartedSpeakingFrame):
                logger.debug("User started speaking")
                await self._start_interruption()
                # Push an out-of-band frame (i.e. not using the ordered push
                # frame task) to stop everything, specially at the output
                # transport.
                await self.push_frame(StartInterruptionFrame())
            elif isinstance(frame, UserStoppedSpeakingFrame):
                logger.debug("User stopped speaking")
                await self._stop_interruption()
                await self.push_frame(StopInterruptionFrame())

        await self.push_frame(frame)

    #
    # Audio input
    #

    async def _vad_analyze(self, audio_frames: bytes) -> VADState:
        state = VADState.QUIET
        vad_analyzer = self.vad_analyzer()
        if vad_analyzer:
            state = await self.get_event_loop().run_in_executor(
                self._executor, vad_analyzer.analyze_audio, audio_frames
            )
        return state

    async def _handle_vad(self, audio_frames: bytes, vad_state: VADState):
        new_vad_state = await self._vad_analyze(audio_frames)
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
                await self._handle_interruptions(frame)

            vad_state = new_vad_state
        return vad_state

    async def _audio_task_handler(self):
        vad_state: VADState = VADState.QUIET
        while True:
            try:
                frame: InputAudioRawFrame = await self._audio_in_queue.get()

                audio_passthrough = True

                # If an audio filter is available, run it before VAD.
                if self._params.audio_in_filter:
                    frame.audio = await self._params.audio_in_filter.filter(frame.audio)

                # Check VAD and push event if necessary. We just care about
                # changes from QUIET to SPEAKING and vice versa.
                if self._params.vad_enabled:
                    vad_state = await self._handle_vad(frame.audio, vad_state)
                    audio_passthrough = self._params.vad_audio_passthrough

                # Push audio downstream if passthrough.
                if audio_passthrough:
                    await self.push_frame(frame)

                self._audio_in_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"{self} error reading audio frames: {e}")
