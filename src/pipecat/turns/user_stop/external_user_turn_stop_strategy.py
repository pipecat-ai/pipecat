#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User turn stop strategy driven by another component in the pipeline."""

import asyncio

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy


class ExternalUserTurnStopStrategy(BaseUserTurnStopStrategy):
    """User turn stop strategy driven by another component in the pipeline.

    The end-of-turn counterpart to
    :class:`~pipecat.turns.user_start.ExternalUserTurnStartStrategy`, taking the
    same two signals:

    - :class:`~pipecat.frames.frames.ProposedUserStoppedSpeakingFrame` — a
      service proposing that the turn has ended. This strategy decides, and
      emits the :class:`~pipecat.frames.frames.UserStoppedSpeakingFrame` itself.
      It may also hold the turn open past the proposal, which is what
      ``wait_for_transcript`` does.

    - :class:`~pipecat.frames.frames.UserStoppedSpeakingFrame` — the turn end was
      already decided and announced elsewhere, typically by a shared
      :class:`~pipecat.turns.user_turn_processor.UserTurnProcessor`. This
      strategy adopts that decision and emits nothing.

    To shift the timing further, subclass this strategy and override
    :meth:`~pipecat.turns.user_stop.BaseUserTurnStopStrategy.trigger_user_turn_stopped`,
    which both paths reach once they decide the turn is over. Its
    ``enable_user_speaking_frames`` argument already carries whichever path got
    there, so pass it through when the override eventually finalizes. See
    ``examples/turn-management/turn-management-custom-external-turn-strategy.py``.
    """

    def __init__(
        self,
        *,
        timeout: float = 0.5,
        wait_for_transcript: bool = True,
        **kwargs,
    ):
        """Initialize the external user turn stop strategy.

        Args:
            timeout: A short delay used internally to handle consecutive or
                slightly delayed transcriptions.
            wait_for_transcript: When True (default), turn-stop signaling
                waits for transcript text to arrive after the external
                stop signal. When False, the strategy signals turn-stop as
                soon as that signal arrives — independent of transcripts.
                Set this to False when local turn detection is the intended
                driver of the conversation (e.g. with a realtime LLM service
                consuming audio directly), so transcripts are off the latency
                critical path. ``LLMContextAggregatorPair`` flips this for you
                when ``realtime_service_mode=True``.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self._timeout = timeout
        self._wait_for_transcript = wait_for_transcript
        self._text = ""
        self._user_speaking = False
        self._seen_interim_results = False
        self._turn_announced_elsewhere = False
        self._turn_open = False
        self._event = asyncio.Event()
        self._task: asyncio.Task | None = None

    @property
    def resolves_proposed_turn_stop_frames(self) -> bool:
        """Whether this strategy resolves proposals into turn stops."""
        return True

    @property
    def wait_for_transcript(self) -> bool:
        """Whether turn-stop signaling waits for transcript text."""
        return self._wait_for_transcript

    @wait_for_transcript.setter
    def wait_for_transcript(self, value: bool):
        self._wait_for_transcript = value

    async def handle_user_turn_started(self):
        """Ready the strategy to detect the end of the turn now starting."""
        await self._reset()
        self._turn_open = True

    async def handle_user_turn_stopped(self):
        """Clear per-turn state once the turn has ended."""
        await self._reset()
        self._turn_open = False

    async def _reset(self):
        """Clear per-turn state. Runs at both turn boundaries."""
        self._text = ""
        self._user_speaking = False
        self._seen_interim_results = False
        self._turn_announced_elsewhere = False
        self._event.clear()

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the strategy.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        self._task = self.create_task(self._task_handler(), f"{self}::_task_handler")

    async def cleanup(self):
        """Cleanup the strategy."""
        await super().cleanup()
        if self._task:
            await self.task_manager.cancel_task(self._task)
            self._task = None

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        """Process an incoming frame to update strategy state.

        Updates internal transcription text and speaking state. The user end
        turn will be triggered when appropriate based on the collected frames.

        Args:
            frame: The frame to be analyzed.

        Returns:
            Always returns CONTINUE so subsequent stop strategies are evaluated.
        """
        if isinstance(frame, ProposedUserStartedSpeakingFrame):
            await self._handle_user_started_speaking(announced_elsewhere=False)
        elif isinstance(frame, ProposedUserStoppedSpeakingFrame):
            await self._handle_user_stopped_speaking(announced_elsewhere=False)
        elif isinstance(frame, UserStartedSpeakingFrame):
            await self._handle_user_started_speaking(announced_elsewhere=True)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._handle_user_stopped_speaking(announced_elsewhere=True)
        elif isinstance(frame, InterimTranscriptionFrame):
            await self._handle_interim_transcription(frame)
        elif isinstance(frame, TranscriptionFrame):
            await self._handle_transcription(frame)

        return ProcessFrameResult.CONTINUE

    async def _handle_user_started_speaking(self, *, announced_elsewhere: bool):
        """Handle the external signal that the user is speaking."""
        self._user_speaking = True
        self._turn_announced_elsewhere = announced_elsewhere

    async def _handle_user_stopped_speaking(self, *, announced_elsewhere: bool):
        """Handle the external signal that the user has stopped speaking."""
        self._user_speaking = False
        self._turn_announced_elsewhere = announced_elsewhere
        await self._maybe_trigger_user_turn_stopped()

    async def _handle_interim_transcription(self, frame: InterimTranscriptionFrame):
        self._seen_interim_results = True

    async def _handle_transcription(self, frame: TranscriptionFrame):
        """Handle user transcription."""
        self._text += frame.text
        # We just got a final result, so let's reset interim results.
        self._seen_interim_results = False
        # Reset aggregation timer.
        self._event.set()

    async def _task_handler(self):
        """Asynchronously monitor transcriptions and trigger user end turn when ready.

        If transcription text exists and the user is not currently speaking,
        triggers the user end turn. Handles multiple or delayed transcriptions
        gracefully.

        """
        while True:
            try:
                await asyncio.wait_for(self._event.wait(), timeout=self._timeout)
                self._event.clear()
            except TimeoutError:
                # Note: with wait_for_transcript off (realtime mode) this fires
                # on every tick. The _turn_open check protects against
                # unnecessarily triggering over and over (harmless, since the
                # controller drops repeated triggers, but noisy).
                if self._turn_open:
                    await self._maybe_trigger_user_turn_stopped()

    async def _trigger_user_turn_stopped(self):
        """End the turn, emitting the turn frame unless it was already announced.

        Reads the flag rather than taking it as an argument: finalization can
        land here from the transcript timeout above, long after the signal that
        set it.
        """
        if self._turn_announced_elsewhere:
            logger.debug(f"{self}: adopting a user turn stop decided elsewhere")
        else:
            logger.debug(f"{self}: resolving a proposed user turn stop")
        await self.trigger_user_turn_stopped(
            enable_user_speaking_frames=False if self._turn_announced_elsewhere else None
        )

    async def _maybe_trigger_user_turn_stopped(self):
        if self._user_speaking:
            return
        if not self._wait_for_transcript:
            # Fire as soon as the external stop signal arrives —
            # transcripts (if any) are off the latency critical path.
            await self._trigger_user_turn_stopped()
            return
        if not self._seen_interim_results and self._text:
            await self._trigger_user_turn_stopped()
