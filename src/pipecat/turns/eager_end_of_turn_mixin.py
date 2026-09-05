#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Mixin for STT services that predict the end of a turn before committing to it."""

import uuid
from typing import Any

from loguru import logger

from pipecat.frames.frames import (
    EagerEndOfTurnCancelFrame,
    EagerEndOfTurnTranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601


class EagerEndOfTurnSTTServiceMixin(FrameProcessor):
    """Adds eager end-of-turn signalling to an STT service that predicts one.

    Some services report that a turn has probably ended before committing to it,
    and withdraw the prediction if the user turns out to be mid-sentence. A
    response can be generated during that gap, so the prediction carries an id
    identifying it across the pipeline — see
    :class:`~pipecat.turns.user_stop.EagerUserTurnStopStrategy`, which acts on
    the frames pushed here.

    A service drives the prediction through its own three lifecycle points::

        # the service predicts the turn has ended
        await self._push_eager_end_of_turn(transcript, user_id=..., language=...)

        # the user resumed speaking, so the prediction is void
        await self._cancel_eager_end_of_turn()

        # the turn was committed, so the prediction is resolved
        self._clear_eager_end_of_turn()

    All three are safe to call when no prediction is outstanding.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the eager end-of-turn mixin.

        Args:
            *args: Positional arguments passed to the parent class.
            **kwargs: Keyword arguments passed to the parent class.
        """
        super().__init__(*args, **kwargs)
        self._eager_speculation_id: str | None = None

    @property
    def eager_speculation_id(self) -> str | None:
        """The prediction awaiting a committed end of turn, if any."""
        return self._eager_speculation_id

    async def _push_eager_end_of_turn(
        self,
        transcript: str,
        *,
        user_id: str,
        language: Language | None = None,
        result: Any | None = None,
    ):
        """Report that the turn has probably ended.

        Args:
            transcript: What the service heard for the turn so far.
            user_id: Identifier for the user who spoke.
            language: Detected or specified language of the speech.
            result: Raw result from the STT service.
        """
        self._eager_speculation_id = str(uuid.uuid4())
        logger.trace(f"{self}: eager end of turn: [{transcript}]")
        await self.push_frame(
            EagerEndOfTurnTranscriptionFrame(
                transcript,
                user_id,
                time_now_iso8601(),
                self._eager_speculation_id,
                language,
                result=result,
            )
        )

    async def _cancel_eager_end_of_turn(self):
        """Withdraw the prediction, naming it so consumers can match it."""
        if not self._eager_speculation_id:
            return

        logger.trace(f"{self}: eager end of turn withdrawn")
        speculation_id, self._eager_speculation_id = self._eager_speculation_id, None
        await self.push_frame(EagerEndOfTurnCancelFrame(speculation_id))

    def _clear_eager_end_of_turn(self):
        """Resolve the prediction without withdrawing it.

        Called when the turn is committed: whatever was generated from the
        prediction is settled by the committed transcript, not by this.
        """
        self._eager_speculation_id = None
