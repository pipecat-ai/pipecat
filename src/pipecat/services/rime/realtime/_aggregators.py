#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Prism aggregation with separate application edits and provider transcripts."""

import copy
from dataclasses import dataclass

from pipecat.frames.frames import Frame, LLMContextFrame, LLMRunFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMAssistantAggregatorParams,
    LLMContextAggregator,
    LLMContextAggregatorPair,
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection


@dataclass
class _HistoryUpdate:
    messages: list
    append: bool


@dataclass
class _PendingHistory:
    update: _HistoryUpdate | None = None


@dataclass
class _RimeContextFrame(LLMContextFrame):
    history_update: _HistoryUpdate | None = None
    instruction: str | None = None
    response_metadata: dict[str, str] | None = None


@dataclass
class RimeReplyFrame(LLMRunFrame):
    """Request one Prism reply with optional instruction and labels.

    Session labels still apply. These labels replace matching session labels
    for this reply only. Prism starts speech replies without this frame.
    """

    instruction: str | None = None
    response_metadata: dict[str, str] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.response_metadata is not None:
            self.response_metadata = dict(self.response_metadata)


class _ApplicationHistory(LLMContextAggregator):
    """Keep application edits shared between both aggregator halves."""

    def __init__(self, *args, pending_history: _PendingHistory, **kwargs):
        self._pending_history = pending_history
        super().__init__(*args, **kwargs)

    def add_messages(self, messages):
        super().add_messages(messages)
        pending = self._pending_history
        if pending.update is None:
            pending.update = _HistoryUpdate([], append=True)
        pending.update.messages.extend(copy.deepcopy(messages))

    def set_messages(self, messages):
        super().set_messages(messages)
        self._pending_history.update = _HistoryUpdate(copy.deepcopy(self.messages), append=False)

    def transform_messages(self, transform):
        super().transform_messages(transform)
        self._pending_history.update = _HistoryUpdate(copy.deepcopy(self.messages), append=False)

    def _get_context_frame(self) -> _RimeContextFrame:
        return _RimeContextFrame(context=self.context)

    async def push_frame(
        self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ) -> None:
        """Attach pending edits only to context frames sent towards Prism."""
        towards_service = (
            FrameDirection.DOWNSTREAM
            if isinstance(self, LLMUserAggregator)
            else FrameDirection.UPSTREAM
        )
        if isinstance(frame, _RimeContextFrame) and direction == towards_service:
            frame.history_update = self._pending_history.update
            self._pending_history.update = None
        await super().push_frame(frame, direction)


class _RimeUserAggregator(_ApplicationHistory, LLMUserAggregator):
    async def _handle_llm_run(self, frame: LLMRunFrame) -> None:
        if isinstance(frame, RimeReplyFrame):
            await self.push_frame(
                _RimeContextFrame(
                    context=self.context,
                    instruction=frame.instruction,
                    response_metadata=frame.response_metadata,
                )
            )
        else:
            await super()._handle_llm_run(frame)

    async def push_aggregation(self) -> str:
        # Prism starts speech replies itself; this flush only records the transcript.
        return await self._push_aggregation(run_llm=False)


class _RimeAssistantAggregator(_ApplicationHistory, LLMAssistantAggregator):
    pass


class RimeContextAggregatorPair(LLMContextAggregatorPair):
    """Reuse Pipecat turn handling while preserving application history intent."""

    def __init__(
        self,
        context: LLMContext,
        *,
        user_params: LLMUserAggregatorParams | None = None,
        assistant_params: LLMAssistantAggregatorParams | None = None,
    ):
        """Share pending application edits between the user and assistant halves.

        Args:
            context: Shared conversation record.
            user_params: Standard user aggregator configuration.
            assistant_params: Standard assistant aggregator configuration.
        """
        pending = _PendingHistory()
        self._user = _RimeUserAggregator(
            context, params=user_params, pending_history=pending, _realtime_service_mode=True
        )
        self._assistant = _RimeAssistantAggregator(
            context,
            params=assistant_params,
            pending_history=pending,
            _realtime_service_mode=True,
            _paired_user_aggregator=self._user,
        )
