#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Bot turn start strategy based on turn detection analyzers."""

import asyncio
from typing import Optional

from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, EndOfTurnState
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    MetricsFrame,
    SpeechControlParamsFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import MetricsData
from pipecat.turns.bot.base_bot_turn_start_strategy import BaseBotTurnStartStrategy
from pipecat.utils.asyncio.task_manager import BaseTaskManager


class TurnAnalyzerBotTurnStartStrategy(BaseBotTurnStartStrategy):
    """Bot turn start strategy using a turn detection model to detect end of user turn.

    This strategy uses the turn detection models to determine when the user has
    finished speaking, combining audio, VAD, and transcription frames. Once the
    turn is considered complete, the bot turn is triggered.

    """

    def __init__(self, *, turn_analyzer: BaseTurnAnalyzer, timeout: float = 0.5):
        """Initialize the bot turn start strategy.

        Args:
            turn_analyzer: The turn detection analyzer instance to detect end of user turn.
            timeout: Short delay used internally to handle frame timing and event triggering.
        """
        super().__init__()
        self._turn_analyzer = turn_analyzer
        self._timeout = timeout
        self._text = ""
        self._vad_user_speaking = False
        self._event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    async def reset(self):
        """Reset the strategy to its initial state."""
        await super().reset()
        self._text = ""
        self._vad_user_speaking = False
        self._event.set()

    async def setup(self, task_manager: BaseTaskManager):
        """Initialize the strategy with the given task manager.

        Args:
            task_manager: The task manager to be associated with this instance.
        """
        await super().setup(task_manager)
        self._task = task_manager.create_task(self._task_handler(), f"{self}::_task_handler")

    async def cleanup(self):
        """Cleanup the strategy."""
        await super().cleanup()
        if self._task:
            await self.task_manager.cancel_task(self._task)
            self._task = None

    async def process_frame(self, frame: Frame):
        """Process an incoming frame to update the turn analyzer and strategy state.

        Args:
            frame: The frame to be analyzed.
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
        elif isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            await self._handle_transcription(frame)

    async def _start(self, frame: StartFrame):
        """Process the start frame to configure the turn analyzer."""
        self._turn_analyzer.set_sample_rate(frame.audio_in_sample_rate)
        await self.broadcast_frame(SpeechControlParamsFrame, turn_params=self._turn_analyzer.params)

    async def _handle_input_audio(self, frame: InputAudioRawFrame):
        """Handle input audio to check if the turn is completed."""
        state = self._turn_analyzer.append_audio(frame.audio, self._vad_user_speaking)
        await self._handle_end_of_turn(state)

    async def _handle_vad_user_started_speaking(self, _: VADUserStartedSpeakingFrame):
        """Handle when the VAD indicates the user is speaking."""
        self._vad_user_speaking = True
        self._event.set()

    async def _handle_vad_user_stopped_speaking(self, _: VADUserStoppedSpeakingFrame):
        """Handle when the VAD indicates the user has stopped speaking."""
        self._vad_user_speaking = False
        self._event.set()

        state, prediction = await self._turn_analyzer.analyze_end_of_turn()
        await self._handle_prediction_result(prediction)
        await self._handle_end_of_turn(state)

    async def _handle_transcription(self, frame: TranscriptionFrame | InterimTranscriptionFrame):
        """Handle user transcription."""
        # We don't really care about the content.
        self._text = frame.text
        self._event.set()

    async def _handle_end_of_turn(self, state: EndOfTurnState):
        """Handle completion of end-of-turn analysis."""
        if state == EndOfTurnState.COMPLETE:
            self._event.set()

    async def _handle_prediction_result(self, result: Optional[MetricsData]):
        """Handle a prediction result event from the turn analyzer."""
        if result:
            await self.push_frame(MetricsFrame(data=[result]))

    async def _task_handler(self):
        """Asynchronously monitor events and trigger bot turn when appropriate.

        If we have not received a transcription in the specified amount of time
        (and we initially received one) and the turn analyzer said the turn is
        done, then the bot is ready to speak.

        """
        while True:
            try:
                await asyncio.wait_for(self._event.wait(), timeout=self._timeout)
                self._event.clear()
            except asyncio.TimeoutError:
                if self._text:
                    await self.trigger_bot_turn_started()
