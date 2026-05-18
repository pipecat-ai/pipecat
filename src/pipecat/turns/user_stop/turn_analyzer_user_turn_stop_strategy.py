#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User turn stop strategy based on turn detection analyzers."""

import asyncio

from loguru import logger

from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, EndOfTurnState
from pipecat.audio.vad.vad_analyzer import VAD_STOP_SECS
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
from pipecat.turns.types import ProcessFrameResult
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

    Set ``wait_for_transcript=False`` to make this strategy not consider
    user transcripts, so the user turn ends sooner — as soon as the
    analyzer concludes the turn is complete. Most callers don't set
    this directly: it's flipped automatically by
    ``wait_for_transcript_to_end_user_turn=False`` on
    ``LLMUserAggregatorParams``, which also wires the aggregator to
    gather user transcripts after the turn ends. That pattern fits
    pipelines where local turn detection drives a realtime service like
    Gemini Live — the realtime service consumes user audio directly,
    so user transcripts don't need to be in context before it can
    respond.
    """

    def __init__(
        self,
        *,
        turn_analyzer: BaseTurnAnalyzer,
        wait_for_transcript: bool = True,
        **kwargs,
    ):
        """Initialize the user turn stop strategy.

        Args:
            turn_analyzer: The turn detection analyzer instance to detect end of user turn.
            wait_for_transcript: Whether the strategy considers user
                transcripts in deciding when the user turn ends.
                Defaults to True. Usually flipped indirectly via
                ``wait_for_transcript_to_end_user_turn=False`` on
                ``LLMUserAggregatorParams``.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self._turn_analyzer = turn_analyzer
        self._wait_for_transcript = wait_for_transcript
        self._stt_timeout: float = 0.0  # STT P99 latency from STTMetadataFrame
        self._stop_secs: float = 0.0  # VAD stop_secs from VADUserStoppedSpeakingFrame

        self._stop_secs_warned: bool = False

        self._text = ""
        self._turn_complete = False
        self._vad_user_speaking = False
        self._vad_stopped_time: float | None = None  # Track when VAD stopped was received
        self._transcript_finalized = False
        self._timeout_task: asyncio.Task | None = None
        self._timeout_expired: bool = False

    async def reset(self):
        """Reset the strategy to its initial state."""
        await super().reset()
        self._text = ""
        self._turn_complete = False
        self._vad_user_speaking = False
        self._vad_stopped_time = None
        self._transcript_finalized = False
        self._timeout_expired = False
        if self._timeout_task:
            await self.task_manager.cancel_task(self._timeout_task)
            self._timeout_task = None

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

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        """Process an incoming frame to update the turn analyzer and strategy state.

        Args:
            frame: The frame to be analyzed.

        Returns:
            Always returns CONTINUE so subsequent stop strategies are evaluated.
        """
        await super().process_frame(frame)

        if isinstance(frame, StartFrame):
            await self._start(frame)
        elif isinstance(frame, STTMetadataFrame):
            self._stt_timeout = frame.ttfs_p99_latency
            self._stop_secs_warned = False
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            await self._handle_vad_user_started_speaking(frame)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._handle_vad_user_stopped_speaking(frame)
        elif isinstance(frame, InputAudioRawFrame):
            await self._handle_input_audio(frame)
        elif isinstance(frame, TranscriptionFrame):
            await self._handle_transcription(frame)

        return ProcessFrameResult.CONTINUE

    async def _start(self, frame: StartFrame):
        """Process the start frame to configure the turn analyzer."""
        self._turn_analyzer.set_sample_rate(frame.audio_in_sample_rate)
        await self.broadcast_frame(SpeechControlParamsFrame, turn_params=self._turn_analyzer.params)

    async def _handle_input_audio(self, frame: InputAudioRawFrame):
        """Handle input audio to check if the turn is completed."""
        state = self._turn_analyzer.append_audio(frame.audio, self._vad_user_speaking)

        # Streaming analyzers (e.g. KrispVivaTurn) detect turn completion
        # frame-by-frame inside append_audio, so COMPLETE is returned here
        # rather than in analyze_end_of_turn. Batch analyzers (BaseSmartTurn)
        # return COMPLETE here only on a silence timeout. In either case we
        # consume and push metrics immediately while they're fresh.
        if state == EndOfTurnState.COMPLETE:
            _, prediction = await self._turn_analyzer.analyze_end_of_turn()
            await self._handle_prediction_result(prediction)
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
        self._timeout_expired = False
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

        if not self._wait_for_transcript:
            # No transcript to wait for. Trigger now if the turn is already
            # complete; otherwise the analyzer's audio path will trigger once
            # it indicates completion.
            await self._maybe_trigger_user_turn_stopped()
            return

        # Start the STT timeout (adjusted by VAD stop_secs since that time already elapsed)
        timeout = max(0, self._stt_timeout - self._stop_secs)

        if not self._stop_secs_warned:
            if self._stop_secs != VAD_STOP_SECS:
                self._stop_secs_warned = True
                logger.warning(
                    f"{self}: VAD stop_secs ({self._stop_secs}s) differs from the "
                    f"recommended default ({VAD_STOP_SECS}s). Built-in p99 latency "
                    f"values assume stop_secs={VAD_STOP_SECS}. Re-run "
                    f"https://github.com/pipecat-ai/stt-benchmark with your settings "
                    f"and pass the TTFS P99 latency result as ttfs_p99_latency to "
                    f"your STT service."
                )
            if self._stt_timeout > 0 and self._stop_secs >= self._stt_timeout:
                self._stop_secs_warned = True
                logger.warning(
                    f"{self}: VAD stop_secs ({self._stop_secs}s) >= STT p99 latency "
                    f"({self._stt_timeout}s). STT wait timeout collapsed to 0s, which "
                    f"may cause delayed turn detection specified by the "
                    f"user_turn_stop_timeout parameter in the LLMUserAggregatorParams."
                )

        self._timeout_task = self.task_manager.create_task(
            self._timeout_handler(timeout), f"{self}::_timeout_handler"
        )
        # Make sure the task is scheduled.
        await asyncio.sleep(0)

    async def _handle_transcription(self, frame: TranscriptionFrame):
        """Handle user transcription."""
        # We don't really care about the content.
        self._text = frame.text
        if frame.finalized:
            self._transcript_finalized = True
            # For finalized transcripts, trigger immediately if turn is complete
            await self._maybe_trigger_user_turn_stopped()
        elif self._timeout_expired and self._turn_complete:
            # The p99 timeout already elapsed without a transcript. Now that
            # we have one, trigger the turn stop immediately. This handles the
            # case where the transcript is slower to arrive than the p99 timeout,
            # trigger the user turn to stop immediately.
            await self.trigger_user_turn_stopped()
            return

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
            # Make sure the task is scheduled.
            await asyncio.sleep(0)

    async def _handle_prediction_result(self, result: MetricsData | None):
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

        self._timeout_expired = True
        await self._maybe_trigger_user_turn_stopped()

    async def _maybe_trigger_user_turn_stopped(self):
        """Trigger user turn stopped if conditions are met.

        Conditions:
        - We have transcription text (skipped when ``wait_for_transcript`` is False)
        - Turn analyzer indicates turn is complete
        - Either the timeout has elapsed OR we have a finalized transcript
        """
        if not self._turn_complete:
            return
        if self._wait_for_transcript and not self._text:
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
