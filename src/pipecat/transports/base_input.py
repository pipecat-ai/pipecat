#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from loguru import logger

from pipecat.audio.turn.base_turn_analyzer import (
    BaseTurnAnalyzer,
    EndOfTurnState,
)
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
    MetricsFrame,
    StartFrame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    SystemFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADParamsUpdateFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import MetricsData
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

        if self._params.vad_enabled:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "Parameter 'vad_enabled' is deprecated, use 'audio_in_enabled' and 'vad_analyzer' instead.",
                    DeprecationWarning,
                )
            self._params.audio_in_enabled = True

        if self._params.vad_audio_passthrough:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "Parameter 'vad_audio_passthrough' is deprecated, audio passthrough is now always enabled. Use 'audio_in_passthrough' to disable.",
                    DeprecationWarning,
                )
            self._params.audio_in_passthrough = True

        if self._params.camera_in_enabled:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "Parameters 'camera_*' are deprecated, use 'video_*' instead.",
                    DeprecationWarning,
                )
            self._params.video_in_enabled = self._params.camera_in_enabled
            self._params.video_out_enabled = self._params.camera_out_enabled
            self._params.video_out_is_live = self._params.camera_out_is_live
            self._params.video_out_width = self._params.camera_out_width
            self._params.video_out_height = self._params.camera_out_height
            self._params.video_out_bitrate = self._params.camera_out_bitrate
            self._params.video_out_framerate = self._params.camera_out_framerate
            self._params.video_out_color_format = self._params.camera_out_color_format

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

    @property
    def turn_analyzer(self) -> Optional[BaseTurnAnalyzer]:
        return self._params.turn_analyzer

    async def start(self, frame: StartFrame):
        self._sample_rate = self._params.audio_in_sample_rate or frame.audio_in_sample_rate

        # Configure VAD analyzer.
        if self._params.vad_analyzer:
            self._params.vad_analyzer.set_sample_rate(self._sample_rate)

        # Configure End of turn analyzer.
        if self._params.turn_analyzer:
            self._params.turn_analyzer.set_sample_rate(self._sample_rate)

        # Start audio filter.
        if self._params.audio_in_filter:
            await self._params.audio_in_filter.start(self._sample_rate)

    async def stop(self, frame: EndFrame):
        # Cancel and wait for the audio input task to finish.
        if self._audio_task and self._params.audio_in_enabled:
            await self.cancel_task(self._audio_task)
            self._audio_task = None
        # Stop audio filter.
        if self._params.audio_in_filter:
            await self._params.audio_in_filter.stop()

    async def cancel(self, frame: CancelFrame):
        # Cancel and wait for the audio input task to finish.
        if self._audio_task and self._params.audio_in_enabled:
            await self.cancel_task(self._audio_task)
            self._audio_task = None

    async def set_transport_ready(self, frame: StartFrame):
        """To be called when the transport is ready to stream."""
        # Create audio input queue and task if needed.
        if not self._audio_task and self._params.audio_in_enabled:
            self._audio_in_queue = asyncio.Queue()
            self._audio_task = self.create_task(self._audio_task_handler())

    async def push_audio_frame(self, frame: InputAudioRawFrame):
        if self._params.audio_in_enabled:
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
            # If the turn analyser is enabled, this will prevent:
            # - Creating the UserStoppedSpeakingFrame
            # - Creating the UserStartedSpeakingFrame multiple times
            can_create_user_frames = (
                self._params.turn_analyzer is None
                or not self._params.turn_analyzer.speech_triggered
            )
            if new_vad_state == VADState.SPEAKING:
                await self.push_frame(VADUserStartedSpeakingFrame())
                if can_create_user_frames:
                    frame = UserStartedSpeakingFrame()
            elif new_vad_state == VADState.QUIET:
                await self.push_frame(VADUserStoppedSpeakingFrame())
                if can_create_user_frames:
                    frame = UserStoppedSpeakingFrame()

            if frame:
                await self._handle_user_interruption(frame)

            vad_state = new_vad_state
        return vad_state

    async def _handle_end_of_turn(self):
        if self.turn_analyzer:
            state, prediction = await self.turn_analyzer.analyze_end_of_turn()
            await self._handle_prediction_result(prediction)
            await self._handle_end_of_turn_complete(state)

    async def _handle_end_of_turn_complete(self, state: EndOfTurnState):
        if state == EndOfTurnState.COMPLETE:
            await self._handle_user_interruption(UserStoppedSpeakingFrame())

    async def _run_turn_analyzer(
        self, frame: InputAudioRawFrame, vad_state: VADState, previous_vad_state: VADState
    ):
        is_speech = vad_state == VADState.SPEAKING or vad_state == VADState.STARTING
        # If silence exceeds threshold, we are going to receive EndOfTurnState.COMPLETE
        end_of_turn_state = self._params.turn_analyzer.append_audio(frame.audio, is_speech)
        if end_of_turn_state == EndOfTurnState.COMPLETE:
            await self._handle_end_of_turn_complete(end_of_turn_state)
        # Otherwise we are going to trigger to check if the turn is completed based on the VAD
        elif vad_state == VADState.QUIET and vad_state != previous_vad_state:
            await self._handle_end_of_turn()

    async def _audio_task_handler(self):
        vad_state: VADState = VADState.QUIET
        while True:
            frame: InputAudioRawFrame = await self._audio_in_queue.get()

            # If an audio filter is available, run it before VAD.
            if self._params.audio_in_filter:
                frame.audio = await self._params.audio_in_filter.filter(frame.audio)

            # Check VAD and push event if necessary. We just care about
            # changes from QUIET to SPEAKING and vice versa.
            previous_vad_state = vad_state
            if self._params.vad_analyzer:
                vad_state = await self._handle_vad(frame, vad_state)

            if self._params.turn_analyzer:
                await self._run_turn_analyzer(frame, vad_state, previous_vad_state)

            # Push audio downstream if passthrough is set.
            if self._params.audio_in_passthrough:
                await self.push_frame(frame)

            self._audio_in_queue.task_done()

    async def _handle_prediction_result(self, result: MetricsData):
        """Handle a prediction result event from the turn analyzer.

        Args:
            result: The prediction result MetricsData.
        """
        await self.push_frame(MetricsFrame(data=[result]))
