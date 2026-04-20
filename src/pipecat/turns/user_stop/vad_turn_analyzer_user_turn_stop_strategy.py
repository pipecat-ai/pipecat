#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User turn stop strategy based on VAD and turn detection analyzers.

This strategy uses a turn analyzer to detect end-of-turn but does not
require STT transcriptions. It triggers immediately when the turn analyzer
indicates COMPLETE, making it suitable for speech-to-speech pipelines
where transcriptions arrive too late to be useful for turn decisions.
"""

from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, EndOfTurnState
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    MetricsFrame,
    SpeechControlParamsFrame,
    StartFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import MetricsData
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy
from pipecat.utils.asyncio.task_manager import BaseTaskManager


class VADTurnAnalyzerUserTurnStopStrategy(BaseUserTurnStopStrategy):
    """User turn stop strategy that uses a turn analyzer without waiting for transcriptions.

    This strategy feeds audio and VAD frames to a turn detection model
    (``BaseTurnAnalyzer``) and triggers immediately when the model indicates
    the turn is complete. Unlike ``TurnAnalyzerUserTurnStopStrategy``, it does
    not wait for STT transcriptions, making it ideal for speech-to-speech
    pipelines (e.g. Gemini Live) where audio goes directly to the LLM.

    The ``UserTurnController`` provides a safety-net timeout
    (``user_turn_stop_timeout``, default 5s) if the turn analyzer never
    returns COMPLETE.
    """

    def __init__(self, *, turn_analyzer: BaseTurnAnalyzer, **kwargs):
        """Initialize the user turn stop strategy.

        Args:
            turn_analyzer: The turn detection analyzer instance to detect end of user turn.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self._turn_analyzer = turn_analyzer
        self._vad_user_speaking = False

    async def reset(self):
        """Reset the strategy to its initial state."""
        await super().reset()
        self._vad_user_speaking = False

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
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            await self._handle_vad_user_started_speaking(frame)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._handle_vad_user_stopped_speaking(frame)
        elif isinstance(frame, InputAudioRawFrame):
            await self._handle_input_audio(frame)

        return ProcessFrameResult.CONTINUE

    async def _start(self, frame: StartFrame):
        """Process the start frame to configure the turn analyzer."""
        self._turn_analyzer.set_sample_rate(frame.audio_in_sample_rate)
        await self.broadcast_frame(SpeechControlParamsFrame, turn_params=self._turn_analyzer.params)

    async def _handle_input_audio(self, frame: InputAudioRawFrame):
        """Handle input audio to check if the turn is completed."""
        state = self._turn_analyzer.append_audio(frame.audio, self._vad_user_speaking)

        # Streaming analyzers (e.g. KrispVivaTurn) detect turn completion
        # frame-by-frame inside append_audio, so COMPLETE is returned here.
        if state == EndOfTurnState.COMPLETE:
            _, prediction = await self._turn_analyzer.analyze_end_of_turn()
            await self._handle_prediction_result(prediction)
            await self.trigger_user_turn_stopped()

    async def _handle_vad_user_started_speaking(self, frame: VADUserStartedSpeakingFrame):
        """Handle when the VAD indicates the user is speaking."""
        self._turn_analyzer.update_vad_start_secs(frame.start_secs)
        self._vad_user_speaking = True

    async def _handle_vad_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        """Handle when the VAD indicates the user has stopped speaking."""
        self._vad_user_speaking = False

        state, prediction = await self._turn_analyzer.analyze_end_of_turn()
        await self._handle_prediction_result(prediction)

        if state == EndOfTurnState.COMPLETE:
            await self.trigger_user_turn_stopped()

    async def _handle_prediction_result(self, result: MetricsData | None):
        """Handle a prediction result event from the turn analyzer."""
        if result:
            await self.push_frame(MetricsFrame(data=[result]))
