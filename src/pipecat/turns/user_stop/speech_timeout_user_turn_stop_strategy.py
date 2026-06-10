#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Speech timeout-based user turn stop strategy."""

import asyncio

from loguru import logger

from pipecat.audio.vad.vad_analyzer import VAD_STOP_SECS
from pipecat.frames.frames import (
    Frame,
    STTMetadataFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy
from pipecat.utils.asyncio.task_manager import BaseTaskManager


class SpeechTimeoutUserTurnStopStrategy(BaseUserTurnStopStrategy):
    """User turn stop strategy using two independent timers after VAD stop.

    After the user stops speaking (detected by VAD), this strategy runs two
    independent timers. The user turn stop is triggered only when both have
    finished and at least one transcript has been received:

    - user_speech_timeout: Policy floor — the window in which the user may
      resume speaking after a pause. Always runs to completion.
    - stt_timeout: Safety net for STT latency — the P99 time for the STT
      service to return a final transcript after VAD stop, adjusted by the
      VAD stop_secs. Short-circuited when the STT service emits a finalized
      transcript (TranscriptionFrame.finalized=True), since finalization
      means STT has nothing more to send.

    Fallback: when a transcript arrives without a VAD stop event, the
    user_speech_timeout timer measures inactivity since the last transcript
    (rearmed on each transcript). stt_timeout has no meaning here since it
    is defined relative to VAD stop, and STT has already emitted a
    transcript — so the stt wait is marked done immediately.
    """

    def __init__(self, *, user_speech_timeout: float = 0.6, **kwargs):
        """Initialize the speech timeout-based user turn stop strategy.

        Args:
            user_speech_timeout: Time to wait for the user to potentially
                say more after they pause speaking. Defaults to 0.6 seconds.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self._user_speech_timeout = user_speech_timeout
        self._stt_timeout: float = 0.0  # STT P99 latency from STTMetadataFrame
        self._stop_secs: float = 0.0  # VAD stop_secs from VADUserStoppedSpeakingFrame
        self._stop_secs_warned: bool = False

        self._text = ""
        self._vad_user_speaking = False
        self._transcript_finalized = False
        self._vad_stopped_time: float | None = None

        self._user_speech_timeout_task: asyncio.Task | None = None
        self._stt_timeout_task: asyncio.Task | None = None
        self._user_speech_wait_done: bool = False
        self._stt_wait_done: bool = False

    async def reset(self):
        """Reset the strategy to its initial state."""
        await super().reset()
        self._text = ""
        self._vad_user_speaking = False
        self._transcript_finalized = False
        self._vad_stopped_time = None
        self._user_speech_wait_done = False
        self._stt_wait_done = False
        await self._cancel_all_tasks()

    async def setup(self, task_manager: BaseTaskManager):
        """Initialize the strategy with the given task manager.

        Args:
            task_manager: The task manager to be associated with this instance.
        """
        await super().setup(task_manager)

    async def cleanup(self):
        """Cleanup the strategy."""
        await super().cleanup()
        await self._cancel_all_tasks()

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        """Process an incoming frame to update strategy state.

        Updates internal transcription text and VAD state. The user end turn
        will be triggered when appropriate based on the collected frames.

        Args:
            frame: The frame to be analyzed.

        Returns:
            Always returns CONTINUE so subsequent stop strategies are evaluated.
        """
        if isinstance(frame, STTMetadataFrame):
            self._stt_timeout = frame.ttfs_p99_latency
            self._stop_secs_warned = False
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            logger.debug(f"{self} VADUserStartedSpeakingFrame received")
            await self._handle_vad_user_started_speaking(frame)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            logger.debug(f"{self} VADUserStoppedSpeakingFrame received")
            await self._handle_vad_user_stopped_speaking(frame)
        elif isinstance(frame, TranscriptionFrame):
            await self._handle_transcription(frame)

        return ProcessFrameResult.CONTINUE

    async def _handle_vad_user_started_speaking(self, _: VADUserStartedSpeakingFrame):
        """Handle when the VAD indicates the user is speaking."""
        self._vad_user_speaking = True
        self._transcript_finalized = False
        self._vad_stopped_time = None
        self._user_speech_wait_done = False
        self._stt_wait_done = False
        await self._cancel_all_tasks()

    async def _handle_vad_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        """Handle when the VAD indicates the user has stopped speaking."""
        self._vad_user_speaking = False
        self._stop_secs = frame.stop_secs
        self._vad_stopped_time = frame.timestamp

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

        # user_speech_timeout is the policy floor and always runs. A prior
        # fallback-mode run of the same timer is superseded here.
        await self._restart_user_speech_timer()

        # stt_timeout is a safety net. Short-circuit it if the transcript is
        # already finalized, or if the VAD stop_secs already covered it.
        self._stt_wait_done = False
        effective_stt_wait = max(0.0, self._stt_timeout - self._stop_secs)
        if self._transcript_finalized or effective_stt_wait <= 0:
            self._stt_wait_done = True
        else:
            self._stt_timeout_task = self.task_manager.create_task(
                self._stt_timeout_handler(effective_stt_wait),
                f"{self}::_stt_timeout_handler",
            )

        # Make sure the tasks are scheduled.
        await asyncio.sleep(0)

    async def _handle_transcription(self, frame: TranscriptionFrame):
        """Handle user transcription."""
        self._text += frame.text

        if frame.finalized:
            self._transcript_finalized = True
            # Short-circuit the stt_timeout safety net: STT has told us
            # there's nothing more coming.
            if not self._stt_wait_done:
                self._stt_wait_done = True
                if self._stt_timeout_task:
                    await self.task_manager.cancel_task(self._stt_timeout_task)
                    self._stt_timeout_task = None

        # If both waits are already done, the turn was waiting on text —
        # trigger now.
        if self._user_speech_wait_done and self._stt_wait_done:
            await self._maybe_trigger_user_turn_stopped()
            return

        # Fallback: transcript arrived without a VAD stop. Measure inactivity
        # since the last transcript with the user_speech_timer. stt_timeout
        # has no meaning here (it's defined relative to VAD stop), so mark
        # the stt wait done immediately.
        if not self._vad_user_speaking and self._vad_stopped_time is None:
            self._stt_wait_done = True
            await self._restart_user_speech_timer()

    async def _restart_user_speech_timer(self):
        """Cancel any running user_speech timer and start a fresh one."""
        if self._user_speech_timeout_task:
            await self.task_manager.cancel_task(self._user_speech_timeout_task)
            self._user_speech_timeout_task = None
        self._user_speech_wait_done = False
        self._user_speech_timeout_task = self.task_manager.create_task(
            self._user_speech_timeout_handler(self._user_speech_timeout),
            f"{self}::_user_speech_timeout_handler",
        )
        # Make sure the task is scheduled so it can't be cancelled before
        # starting (which would leave its coroutine un-awaited).
        await asyncio.sleep(0)

    async def _user_speech_timeout_handler(self, timeout: float):
        """Wait user_speech_timeout then attempt to trigger user turn stopped.

        Args:
            timeout: The timeout in seconds to wait.
        """
        try:
            await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            return
        finally:
            self._user_speech_timeout_task = None

        self._user_speech_wait_done = True
        await self._maybe_trigger_user_turn_stopped()

    async def _stt_timeout_handler(self, timeout: float):
        """Wait stt_timeout then attempt to trigger user turn stopped.

        Args:
            timeout: The timeout in seconds to wait.
        """
        try:
            await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            return
        finally:
            self._stt_timeout_task = None

        self._stt_wait_done = True
        await self._maybe_trigger_user_turn_stopped()

    async def _maybe_trigger_user_turn_stopped(self):
        """Trigger user turn stopped if all required conditions are met.

        Both timers must be done (stt is marked done immediately on the
        fallback path and when finalization short-circuits the safety net),
        the user must not be currently speaking, and at least one transcript
        must have been received.
        """
        if self._vad_user_speaking or not self._text:
            return

        if self._user_speech_wait_done and self._stt_wait_done:
            await self.trigger_user_turn_stopped()

    async def _cancel_all_tasks(self):
        """Cancel any running timer tasks and clear the handles."""
        if self._user_speech_timeout_task:
            await self.task_manager.cancel_task(self._user_speech_timeout_task)
            self._user_speech_timeout_task = None
        if self._stt_timeout_task:
            await self.task_manager.cancel_task(self._stt_timeout_task)
            self._stt_timeout_task = None
