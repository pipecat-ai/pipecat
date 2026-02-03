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
    InterimTranscriptionFrame,
    MetricsFrame,
    SpeechControlParamsFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import MetricsData
from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy
from pipecat.utils.asyncio.task_manager import BaseTaskManager


class TurnAnalyzerUserTurnStopStrategy(BaseUserTurnStopStrategy):
    """User turn stop strategy using a turn detection model to detect end of user turn.

    This strategy uses the turn detection models to determine when the user has
    finished speaking, combining audio, VAD, and transcription frames. Once the
    turn is considered complete, the user end of turn is triggered.

    """

    def __init__(self, *, turn_analyzer: BaseTurnAnalyzer, timeout: float = 0.5, **kwargs):
        """Initialize the user turn stop strategy.

        Args:
            turn_analyzer: The turn detection analyzer instance to detect end of user turn.
            timeout: Short delay used internally to handle frame timing and event triggering.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self._turn_analyzer = turn_analyzer
        self._timeout = timeout
        self._text = ""
        self._turn_complete = False
        self._vad_user_speaking = False
        self._event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    async def reset(self):
        """Reset the strategy to its initial state."""
        await super().reset()
        self._text = ""
        self._turn_complete = False
        self._vad_user_speaking = False
        self._event.clear()

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
        await self._turn_analyzer.cleanup()
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
        elif isinstance(frame, SpeechControlParamsFrame):
            await self._handle_speech_control_params(frame)
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            await self._handle_vad_user_started_speaking(frame)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._handle_vad_user_stopped_speaking(frame)
        elif isinstance(frame, InputAudioRawFrame):
            await self._handle_input_audio(frame)
        elif isinstance(frame, TranscriptionFrame):
            await self._handle_transcription(frame)
        elif isinstance(frame, InterimTranscriptionFrame):
            await self._handle_interim_transcription(frame)

    async def _start(self, frame: StartFrame):
        """Process the start frame to configure the turn analyzer."""
        self._turn_analyzer.set_sample_rate(frame.audio_in_sample_rate)
        await self.broadcast_frame(SpeechControlParamsFrame, turn_params=self._turn_analyzer.params)

    async def _handle_speech_control_params(self, frame: SpeechControlParamsFrame):
        """Sync Smart Turn pre-speech buffering with VAD start delay.

        `VADUserStartedSpeakingFrame` is emitted only once VAD has confirmed speech
        (after `vad_params.start_secs`). Smart Turn should still include the initial
        audio collected during that confirmation window, so we let the analyzer know
        when this value has changed.
        """
        if frame.vad_params:
            self._turn_analyzer.update_vad_start_secs(frame.vad_params.start_secs)

    async def _handle_input_audio(self, frame: InputAudioRawFrame):
        """Handle input audio to check if the turn is completed."""
        state = self._turn_analyzer.append_audio(frame.audio, self._vad_user_speaking)

        # If at this point the model says the turn is complete it will be due to
        # a timeout, so we mark turn as complete and we trigger the user end of
        # turn.
        if state == EndOfTurnState.COMPLETE:
            self._turn_complete = True
            await self._maybe_trigger_user_turn_stopped()

    async def _handle_vad_user_started_speaking(self, _: VADUserStartedSpeakingFrame):
        """Handle when the VAD indicates the user is speaking."""
        self._turn_complete = False
        self._vad_user_speaking = True

    async def _handle_vad_user_stopped_speaking(self, _: VADUserStoppedSpeakingFrame):
        """Handle when the VAD indicates the user has stopped speaking."""
        self._vad_user_speaking = False

        state, prediction = await self._turn_analyzer.analyze_end_of_turn()
        await self._handle_prediction_result(prediction)

        # The user stopped speaking and the turn is complete, we now need to
        # wait for transcriptions.
        self._turn_complete = state == EndOfTurnState.COMPLETE

        # Reset transcription timeout.
        self._event.set()

    async def _handle_transcription(self, frame: TranscriptionFrame):
        """Handle user transcription."""
        # We don't really care about the content.
        self._text = frame.text
        # Reset transcription timeout.
        self._event.set()

    async def _handle_interim_transcription(self, frame: InterimTranscriptionFrame):
        """Handle user interim transcription."""
        # Reset transcription timeout.
        self._event.set()

    async def _handle_prediction_result(self, result: Optional[MetricsData]):
        """Handle a prediction result event from the turn analyzer."""
        if result:
            await self.push_frame(MetricsFrame(data=[result]))

    async def _task_handler(self):
        """Asynchronously monitor events and trigger user end of turn when appropriate.

        If we have not received a transcription in the specified amount of time
        (and we initially received one) and the turn analyzer said the turn is
        done, then the user is done speaking.

        """
        while True:
            try:
                await asyncio.wait_for(self._event.wait(), timeout=self._timeout)
                self._event.clear()
            except asyncio.TimeoutError:
                await self._maybe_trigger_user_turn_stopped()

    async def _maybe_trigger_user_turn_stopped(self):
        if self._text and self._turn_complete:
            await self.trigger_user_turn_stopped()
