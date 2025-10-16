#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base input transport implementation for Pipecat.

This module provides the BaseInputTransport class which handles audio and video
input processing, including VAD, turn analysis, and interruption management.
"""

import asyncio
from typing import Optional

from loguru import logger

from pipecat.audio.turn.base_turn_analyzer import (
    BaseTurnAnalyzer,
    EndOfTurnState,
)
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADState
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EmulateUserStartedSpeakingFrame,
    EmulateUserStoppedSpeakingFrame,
    EndFrame,
    FilterUpdateSettingsFrame,
    Frame,
    InputAudioRawFrame,
    InputImageRawFrame,
    MetricsFrame,
    SpeechControlParamsFrame,
    StartFrame,
    StopFrame,
    SystemFrame,
    UserSpeakingFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADParamsUpdateFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import MetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_transport import TransportParams

AUDIO_INPUT_TIMEOUT_SECS = 0.5


class BaseInputTransport(FrameProcessor):
    """Base class for input transport implementations.

    Handles audio and video input processing including Voice Activity Detection,
    turn analysis, audio filtering, and user interaction management. Supports
    interruption handling and provides hooks for transport-specific implementations.
    """

    def __init__(self, params: TransportParams, **kwargs):
        """Initialize the base input transport.

        Args:
            params: Transport configuration parameters.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)

        self._params = params

        # Input sample rate. It will be initialized on StartFrame.
        self._sample_rate = 0

        # Track bot speaking state for interruption logic
        self._bot_speaking = False

        # Track user speaking state for interruption logic
        self._user_speaking = False

        # Task to process incoming audio (VAD) and push audio frames downstream
        # if passthrough is enabled.
        self._audio_task = None

        # If the transport is stopped with `StopFrame` we might still be
        # receiving frames from the transport but we really don't want to push
        # them downstream until we get another `StartFrame`.
        self._paused = False

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
        """Enable or disable audio streaming on transport start.

        Args:
            enabled: Whether to start audio streaming immediately on transport start.
        """
        logger.debug(f"Enabling audio on start. {enabled}")
        self._params.audio_in_stream_on_start = enabled

    async def start_audio_in_streaming(self):
        """Start audio input streaming.

        Override in subclasses to implement transport-specific audio streaming.
        """
        pass

    @property
    def sample_rate(self) -> int:
        """Get the current audio sample rate.

        Returns:
            The sample rate in Hz.
        """
        return self._sample_rate

    @property
    def vad_analyzer(self) -> Optional[VADAnalyzer]:
        """Get the Voice Activity Detection analyzer.

        Returns:
            The VAD analyzer instance if configured, None otherwise.
        """
        return self._params.vad_analyzer

    @property
    def turn_analyzer(self) -> Optional[BaseTurnAnalyzer]:
        """Get the turn-taking analyzer.

        Returns:
            The turn analyzer instance if configured, None otherwise.
        """
        return self._params.turn_analyzer

    async def start(self, frame: StartFrame):
        """Start the input transport and initialize components.

        Args:
            frame: The start frame containing initialization parameters.
        """
        self._paused = False
        self._user_speaking = False

        self._sample_rate = self._params.audio_in_sample_rate or frame.audio_in_sample_rate

        # Configure VAD analyzer.
        if self._params.vad_analyzer:
            self._params.vad_analyzer.set_sample_rate(self._sample_rate)

        # Configure End of turn analyzer.
        if self._params.turn_analyzer:
            self._params.turn_analyzer.set_sample_rate(self._sample_rate)

        if self._params.vad_analyzer or self._params.turn_analyzer:
            vad_params = self._params.vad_analyzer.params if self._params.vad_analyzer else None
            turn_params = self._params.turn_analyzer.params if self._params.turn_analyzer else None

            speech_frame = SpeechControlParamsFrame(vad_params=vad_params, turn_params=turn_params)
            await self.push_frame(speech_frame)

        # Start audio filter.
        if self._params.audio_in_filter:
            await self._params.audio_in_filter.start(self._sample_rate)

    async def stop(self, frame: EndFrame):
        """Stop the input transport and cleanup resources.

        Args:
            frame: The end frame signaling transport shutdown.
        """
        # Cancel and wait for the audio input task to finish.
        await self._cancel_audio_task()
        # Stop audio filter.
        if self._params.audio_in_filter:
            await self._params.audio_in_filter.stop()

    async def pause(self, frame: StopFrame):
        """Pause the input transport temporarily.

        Args:
            frame: The stop frame signaling transport pause.
        """
        self._paused = True
        # Cancel task so we clear the queue
        await self._cancel_audio_task()
        # Retart the task
        self._create_audio_task()

    async def cancel(self, frame: CancelFrame):
        """Cancel the input transport and stop all processing.

        Args:
            frame: The cancel frame signaling immediate cancellation.
        """
        # Cancel and wait for the audio input task to finish.
        await self._cancel_audio_task()
        # Stop audio filter.
        if self._params.audio_in_filter:
            await self._params.audio_in_filter.stop()

    async def set_transport_ready(self, frame: StartFrame):
        """Called when the transport is ready to stream.

        Args:
            frame: The start frame containing initialization parameters.
        """
        # Create audio input queue and task if needed.
        self._create_audio_task()

    async def push_video_frame(self, frame: InputImageRawFrame):
        """Push a video frame downstream if video input is enabled.

        Args:
            frame: The input video frame to process.
        """
        if self._params.video_in_enabled and not self._paused:
            await self.push_frame(frame)

    async def push_audio_frame(self, frame: InputAudioRawFrame):
        """Push an audio frame to the processing queue if audio input is enabled.

        Args:
            frame: The input audio frame to process.
        """
        if self._params.audio_in_enabled and not self._paused:
            await self._audio_in_queue.put(frame)

    #
    # Frame processor
    #

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and handle transport-specific logic.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
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
        elif isinstance(frame, BotStartedSpeakingFrame):
            await self._handle_bot_started_speaking(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self._handle_bot_stopped_speaking(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, EmulateUserStartedSpeakingFrame):
            logger.debug("Emulating user started speaking")
            await self._handle_user_interruption(VADState.SPEAKING, emulated=True)
        elif isinstance(frame, EmulateUserStoppedSpeakingFrame):
            logger.debug("Emulating user stopped speaking")
            await self._handle_user_interruption(VADState.QUIET, emulated=True)
        # All other system frames
        elif isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
        # Control frames
        elif isinstance(frame, EndFrame):
            # Push EndFrame before stop(), because stop() waits on the task to
            # finish and the task finishes when EndFrame is processed.
            await self.push_frame(frame, direction)
            await self.stop(frame)
        elif isinstance(frame, StopFrame):
            await self.push_frame(frame, direction)
            await self.pause(frame)
        elif isinstance(frame, VADParamsUpdateFrame):
            if self.vad_analyzer:
                self.vad_analyzer.set_params(frame.params)
                speech_frame = SpeechControlParamsFrame(
                    vad_params=frame.params,
                    turn_params=self._params.turn_analyzer.params
                    if self._params.turn_analyzer
                    else None,
                )
                await self.push_frame(speech_frame)
        elif isinstance(frame, FilterUpdateSettingsFrame) and self._params.audio_in_filter:
            await self._params.audio_in_filter.process_frame(frame)
        # Other frames
        else:
            await self.push_frame(frame, direction)

    #
    # Handle interruptions
    #

    async def _handle_user_interruption(self, vad_state: VADState, emulated: bool = False):
        """Handle user interruption events based on speaking state."""
        if vad_state == VADState.SPEAKING:
            logger.debug("User started speaking")
            self._user_speaking = True

            upstream_frame = UserStartedSpeakingFrame(emulated=emulated)
            downstream_frame = UserStartedSpeakingFrame(emulated=emulated)
            await self.push_frame(downstream_frame)
            await self.push_frame(upstream_frame, FrameDirection.UPSTREAM)

            # Only push InterruptionFrame if:
            # 1. No interruption config is set, OR
            # 2. Interruption config is set but bot is not speaking
            should_push_immediate_interruption = (
                not self.interruption_strategies or not self._bot_speaking
            )

            # Make sure we notify about interruptions quickly out-of-band.
            if should_push_immediate_interruption and self.interruptions_allowed:
                await self.push_interruption_task_frame_and_wait()
            elif self.interruption_strategies and self._bot_speaking:
                logger.debug(
                    "User started speaking while bot is speaking with interruption config - "
                    "deferring interruption to aggregator"
                )
        elif vad_state == VADState.QUIET:
            logger.debug("User stopped speaking")
            self._user_speaking = False

            upstream_frame = UserStoppedSpeakingFrame(emulated=emulated)
            downstream_frame = UserStoppedSpeakingFrame(emulated=emulated)
            await self.push_frame(downstream_frame)
            await self.push_frame(upstream_frame, FrameDirection.UPSTREAM)

    #
    # Handle bot speaking state
    #

    async def _handle_bot_started_speaking(self, frame: BotStartedSpeakingFrame):
        """Update bot speaking state when bot starts speaking."""
        self._bot_speaking = True

    async def _handle_bot_stopped_speaking(self, frame: BotStoppedSpeakingFrame):
        """Update bot speaking state when bot stops speaking."""
        self._bot_speaking = False

    #
    # Audio input
    #

    def _create_audio_task(self):
        """Create the audio processing task if audio input is enabled."""
        if not self._audio_task and self._params.audio_in_enabled:
            self._audio_in_queue = asyncio.Queue()
            self._audio_task = self.create_task(self._audio_task_handler())

    async def _cancel_audio_task(self):
        """Cancel and cleanup the audio processing task."""
        if self._audio_task:
            await self.cancel_task(self._audio_task)
            self._audio_task = None

    async def _vad_analyze(self, audio_frame: InputAudioRawFrame) -> VADState:
        """Analyze audio frame for voice activity."""
        state = VADState.QUIET
        if self.vad_analyzer:
            state = await self.vad_analyzer.analyze_audio(audio_frame.audio)
        return state

    async def _handle_vad(self, audio_frame: InputAudioRawFrame, vad_state: VADState) -> VADState:
        """Handle Voice Activity Detection results and generate appropriate frames."""
        new_vad_state = await self._vad_analyze(audio_frame)
        if (
            new_vad_state != vad_state
            and new_vad_state != VADState.STARTING
            and new_vad_state != VADState.STOPPING
        ):
            interruption_state = None

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
                    interruption_state = VADState.SPEAKING
            elif new_vad_state == VADState.QUIET:
                await self.push_frame(VADUserStoppedSpeakingFrame())
                if can_create_user_frames:
                    interruption_state = VADState.QUIET

            if interruption_state:
                await self._handle_user_interruption(interruption_state)

            vad_state = new_vad_state
        return vad_state

    async def _handle_end_of_turn(self):
        """Handle end-of-turn analysis and generate prediction results."""
        if self.turn_analyzer:
            state, prediction = await self.turn_analyzer.analyze_end_of_turn()
            await self._handle_prediction_result(prediction)
            await self._handle_end_of_turn_complete(state)

    async def _handle_end_of_turn_complete(self, state: EndOfTurnState):
        """Handle completion of end-of-turn analysis."""
        if state == EndOfTurnState.COMPLETE:
            await self._handle_user_interruption(VADState.QUIET)

    async def _run_turn_analyzer(
        self, frame: InputAudioRawFrame, vad_state: VADState, previous_vad_state: VADState
    ):
        """Run turn analysis on audio frame and handle results."""
        is_speech = vad_state == VADState.SPEAKING or vad_state == VADState.STARTING
        # If silence exceeds threshold, we are going to receive EndOfTurnState.COMPLETE
        end_of_turn_state = self._params.turn_analyzer.append_audio(frame.audio, is_speech)
        if end_of_turn_state == EndOfTurnState.COMPLETE:
            await self._handle_end_of_turn_complete(end_of_turn_state)
        # Otherwise we are going to trigger to check if the turn is completed based on the VAD
        elif vad_state == VADState.QUIET and vad_state != previous_vad_state:
            await self._handle_end_of_turn()

    async def _audio_task_handler(self):
        """Main audio processing task handler for VAD and turn analysis."""
        vad_state: VADState = VADState.QUIET
        while True:
            try:
                frame: InputAudioRawFrame = await asyncio.wait_for(
                    self._audio_in_queue.get(), timeout=AUDIO_INPUT_TIMEOUT_SECS
                )

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

                if vad_state == VADState.SPEAKING:
                    await self.push_frame(UserSpeakingFrame())
                    await self.push_frame(UserSpeakingFrame(), FrameDirection.UPSTREAM)

                # Push audio downstream if passthrough is set.
                if self._params.audio_in_passthrough:
                    await self.push_frame(frame)

                self._audio_in_queue.task_done()
            except asyncio.TimeoutError:
                if self._user_speaking:
                    logger.warning(
                        "Forcing user stopped speaking due to timeout receiving audio frame!"
                    )
                    vad_state = VADState.QUIET
                    if self._params.turn_analyzer:
                        self._params.turn_analyzer.clear()
                    await self._handle_user_interruption(VADState.QUIET)

    async def _handle_prediction_result(self, result: MetricsData):
        """Handle a prediction result event from the turn analyzer."""
        await self.push_frame(MetricsFrame(data=[result]))
