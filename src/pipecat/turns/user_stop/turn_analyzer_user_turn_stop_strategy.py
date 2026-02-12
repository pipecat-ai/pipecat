#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User turn stop strategy based on turn detection analyzers."""

import asyncio
from typing import Optional

from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, EndOfTurnState
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    MetricsFrame,
    SpeechControlParamsFrame,
    StartFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import MetricsData
from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy
from pipecat.utils.asyncio.task_manager import BaseTaskManager


class TurnAnalyzerUserTurnStopStrategy(BaseUserTurnStopStrategy):
    """User turn stop strategy that uses a turn detection model to determine if the user is done speaking.

    This strategy feeds audio, VAD, and transcription frames to a turn
    detection model (``BaseTurnAnalyzer``) that predicts when the user has
    finished their turn. Once the model indicates the turn is complete, the
    strategy waits for a final transcription before triggering the end of
    the user's turn.

    For services that support finalization (TranscriptionFrame.finalized=True),
    the turn can be triggered immediately once the finalized transcript is
    received. Otherwise, an STT timeout (adjusted by VAD stop_secs) is used
    as a fallback.
    """

    def __init__(self, *, turn_analyzer: BaseTurnAnalyzer, **kwargs):
        """Initialize the user turn stop strategy.

        Args:
            turn_analyzer: The turn detection analyzer instance to detect end of user turn.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self._turn_analyzer = turn_analyzer
        self._stt_timeout: float = 0.0  # STT P99 latency from STTMetadataFrame
        self._stop_secs: float = 0.0  # VAD stop_secs from VADUserStoppedSpeakingFrame

        self._text = ""
        self._turn_complete = False
        self._vad_user_speaking = False
        self._vad_stopped_time: Optional[float] = None  # Track when VAD stopped was received
        self._transcript_finalized = False
        self._timeout_task: Optional[asyncio.Task] = None

    async def reset(self):
        """Reset the strategy to its initial state."""
        await super().reset()
        self._text = ""
        self._turn_complete = False
        self._vad_user_speaking = False
        self._vad_stopped_time = None
        self._transcript_finalized = False

    async def setup(self, task_manager: BaseTaskManager):
        """Initialize the strategy with the given task manager.

        Args:
            task_manager: The task manager to be associated with this instance.
        """
        await super().setup(task_manager)

    async def cleanup(self):
        """Cleanup the strategy."""
        await super().cleanup()
        await self._turn_analyzer.cleanup()
        if self._timeout_task:
            await self.task_manager.cancel_task(self._timeout_task)
            self._timeout_task = None

    async def process_frame(self, frame: Frame):
        """Process an incoming frame to update the turn analyzer and strategy state.

        Args:
            frame: The frame to be analyzed.
        """
        await super().process_frame(frame)

        if isinstance(frame, StartFrame):
            await self._start(frame)
        elif isinstance(frame, STTMetadataFrame):
            self._stt_timeout = frame.ttfs_p99_latency
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            await self._handle_vad_user_started_speaking(frame)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._handle_vad_user_stopped_speaking(frame)
        elif isinstance(frame, InputAudioRawFrame):
            await self._handle_input_audio(frame)
        elif isinstance(frame, TranscriptionFrame):
            await self._handle_transcription(frame)

    async def _start(self, frame: StartFrame):
        """Process the start frame to configure the turn analyzer."""
        self._turn_analyzer.set_sample_rate(frame.audio_in_sample_rate)
        await self.broadcast_frame(SpeechControlParamsFrame, turn_params=self._turn_analyzer.params)

    async def _handle_input_audio(self, frame: InputAudioRawFrame):
        """Handle input audio to check if the turn is completed."""
        state = self._turn_analyzer.append_audio(frame.audio, self._vad_user_speaking)

        # If at this point the model says the turn is complete it will be due to
        # a timeout, so we mark turn as complete and we trigger the user end of
        # turn.
        if state == EndOfTurnState.COMPLETE:
            self._turn_complete = True
            await self._maybe_trigger_user_turn_stopped()

    async def _handle_vad_user_started_speaking(self, frame: VADUserStartedSpeakingFrame):
        """Handle when the VAD indicates the user is speaking."""
        # Sync Smart Turn pre-speech buffering with VAD start delay
        self._turn_analyzer.update_vad_start_secs(frame.start_secs)
        self._turn_complete = False
        self._vad_user_speaking = True
        self._vad_stopped_time = None
        self._transcript_finalized = False
        # Cancel any pending timeout
        if self._timeout_task:
            await self.task_manager.cancel_task(self._timeout_task)
            self._timeout_task = None

    async def _handle_vad_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        """Handle when the VAD indicates the user has stopped speaking."""
        self._vad_user_speaking = False
        self._stop_secs = frame.stop_secs
        self._vad_stopped_time = frame.timestamp

        state, prediction = await self._turn_analyzer.analyze_end_of_turn()
        await self._handle_prediction_result(prediction)

        # The user stopped speaking and the turn is complete, we now need to
        # wait for transcriptions.
        self._turn_complete = state == EndOfTurnState.COMPLETE

        # Start the STT timeout (adjusted by VAD stop_secs since that time already elapsed)
        timeout = max(0, self._stt_timeout - self._stop_secs)
        self._timeout_task = self.task_manager.create_task(
            self._timeout_handler(timeout), f"{self}::_timeout_handler"
        )

    async def _handle_transcription(self, frame: TranscriptionFrame):
        """Handle user transcription."""
        # We don't really care about the content.
        self._text = frame.text
        if frame.finalized:
            self._transcript_finalized = True
            # For finalized transcripts, trigger immediately if turn is complete
            await self._maybe_trigger_user_turn_stopped()

        # Fallback: handle transcripts when no VAD stop was received.
        # This handles edge cases where transcripts arrive without VAD firing.
        # _vad_stopped_time is None means VAD stopped hasn't been received yet.
        # In fallback mode, reset timeout on each transcript to wait for inactivity.
        if not self._vad_user_speaking and self._vad_stopped_time is None:
            # Cancel existing fallback timeout if any
            if self._timeout_task:
                await self.task_manager.cancel_task(self._timeout_task)
            # Without VAD/turn analyzer data, assume turn is complete
            self._turn_complete = True
            timeout = max(0, self._stt_timeout - self._stop_secs)
            self._timeout_task = self.task_manager.create_task(
                self._timeout_handler(timeout), f"{self}::_timeout_handler"
            )

    async def _handle_prediction_result(self, result: Optional[MetricsData]):
        """Handle a prediction result event from the turn analyzer."""
        if result:
            await self.push_frame(MetricsFrame(data=[result]))

    async def _timeout_handler(self, timeout: float):
        """Wait for the timeout then trigger user turn stopped if conditions met.

        Args:
            timeout: The timeout in seconds to wait.
        """
        try:
            await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            return
        finally:
            self._timeout_task = None

        await self._maybe_trigger_user_turn_stopped()

    async def _maybe_trigger_user_turn_stopped(self):
        """Trigger user turn stopped if conditions are met.

        Conditions:
        - We have transcription text
        - Turn analyzer indicates turn is complete
        - Either the timeout has elapsed OR we have a finalized transcript
        """
        if not self._text or not self._turn_complete:
            return

        # For finalized transcripts, trigger immediately
        if self._transcript_finalized:
            # Cancel any remaining timeout since we're triggering now
            if self._timeout_task:
                await self.task_manager.cancel_task(self._timeout_task)
                self._timeout_task = None
            await self.trigger_user_turn_stopped()
            return

        # For non-finalized, only trigger if timeout task has completed
        if self._timeout_task is None:
            await self.trigger_user_turn_stopped()
