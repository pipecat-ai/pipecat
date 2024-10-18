#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams, VADState
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from loguru import logger


class SileroVAD(FrameProcessor):
    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        vad_params: VADParams = VADParams(),
        audio_passthrough: bool = False,
    ):
        super().__init__()

        self._vad_analyzer = SileroVADAnalyzer(sample_rate=sample_rate, params=vad_params)
        self._audio_passthrough = audio_passthrough

        self._processor_vad_state: VADState = VADState.QUIET

    #
    # FrameProcessor
    #

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            await self._analyze_audio(frame)
            if self._audio_passthrough:
                await self.push_frame(frame, direction)
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

    async def _analyze_audio(self, frame: AudioRawFrame):
        # Check VAD and push event if necessary. We just care about changes
        # from QUIET to SPEAKING and vice versa.
        new_vad_state = self._vad_analyzer.analyze_audio(frame.audio)
        if (
            new_vad_state != self._processor_vad_state
            and new_vad_state != VADState.STARTING
            and new_vad_state != VADState.STOPPING
        ):
            new_frame = None

            if new_vad_state == VADState.SPEAKING:
                new_frame = UserStartedSpeakingFrame()
            elif new_vad_state == VADState.QUIET:
                new_frame = UserStoppedSpeakingFrame()

            if new_frame:
                await self._handle_interruptions(new_frame)

            self._processor_vad_state = new_vad_state
