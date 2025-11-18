#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Speaking strategy based on Smart Turn model."""

import asyncio
from typing import Optional

from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, EndOfTurnState
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    MetricsFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import MetricsData
from pipecat.processors.frame_processor import FrameDirection
from pipecat.turns.base_speaking_strategy import BaseSpeakingStrategy
from pipecat.utils.asyncio.task_manager import BaseTaskManager


class TurnAnalyzerSpeakingStrategy(BaseSpeakingStrategy):
    """Speaking strategy based on Smart Turn model.

    This is a speaking strategy based on Smart Turn model. It uses Smart Turn to
    identify when the user has finished speaking.
    """

    def __init__(self, *, turn_analyzer: BaseTurnAnalyzer, timeout: float = 0.4):
        """Initialize the strategy with the givem Smart Turn analyzer.

        Args:
            turn_analyzer: The Smart Turn implementation.
            timeout: Extra timeout to make sure the turn is complete.
        """
        super().__init__()
        self._turn_analyzer = turn_analyzer
        self._timeout = timeout
        self._text = ""
        self._vad_user_speaking = False
        self._event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    def reset(self):
        """Reset the speaking strategy."""
        super().reset()
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
        """Process an incoming frame.

        The analysis of incoming frames will decide if the bot should start
        speaking.

        Args:
            frame: The frame to be processed.

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
        """Process the start frame to configure the Smart Turn analyzer."""
        self._turn_analyzer.set_sample_rate(frame.audio_in_sample_rate)

    async def _handle_input_audio(self, frame: InputAudioRawFrame):
        """Handle input audio to check if the turn is completed."""
        state = self._turn_analyzer.append_audio(frame.audio, self._vad_user_speaking)
        await self._handle_end_of_turn(state)

    async def _handle_vad_user_started_speaking(self, frame: VADUserStartedSpeakingFrame):
        """Handle when the VAD indicates the user is speaking."""
        self._vad_user_speaking = True
        self._event.set()

    async def _handle_vad_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
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
            await self._call_event_handler(
                "on_push_frame",
                MetricsFrame(data=[result]),
                FrameDirection.DOWNSTREAM,
            )

    async def _task_handler(self):
        """Asynchronously check if the bot should start speaking.

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
                    await self._call_event_handler("on_should_speak")
