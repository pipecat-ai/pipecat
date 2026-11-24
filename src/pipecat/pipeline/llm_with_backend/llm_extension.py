#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The LLM extensions an ``LLMWithBackend`` installs on its frontend."""

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from loguru import logger

from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import (
    Frame,
    FunctionCallFromLLM,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    SystemFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.base_llm_extension import BaseLLMExtension, LLMExtensionTool

if TYPE_CHECKING:
    from pipecat.services.llm_service import FunctionCallRegistryItem


class BackendPairingLLMExtension(BaseLLMExtension):
    """Pairs the frontend of an ``LLMWithBackend`` with its backend.

    The frontend's half of the pairing: the instruction that says when and
    how to delegate and what to do with what the backend sends back, and the
    tools delegation goes through, ``delegate`` first of all. The composite
    builds both from its :class:`~pipecat.pipeline.llm_with_backend.BackendConnector`
    and installs the extension on the frontend; the connector runs the
    session the tools feed.
    """

    def __init__(self, *, instruction: str | None, tools: Sequence[FunctionSchema], **kwargs):
        """Initialize the extension.

        Args:
            instruction: What the pairing adds to the frontend's system
                instruction, or ``None`` for nothing.
            tools: The tools delegation goes through, each carrying its
                handler. They run as asynchronous tools: a delegation returns
                at once, and the backend's answer arrives later.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._instruction = instruction
        # A delegation is announced when it starts, as a call of the
        # application's own tools is: the application may speak on it, and the
        # pipeline counts it among the calls in progress.
        self._tools = [
            LLMExtensionTool(tool, cancel_on_interruption=False, announce_call_start=True)
            for tool in tools
        ]

    @property
    def system_instruction(self) -> str | None:
        """The pairing instruction."""
        return self._instruction

    def builtin_tools(
        self, registered: Mapping[str | None, "FunctionCallRegistryItem"]
    ) -> list[LLMExtensionTool]:
        """The delegation tools, whatever is registered.

        Args:
            registered: The service's registered tools by name.

        Returns:
            The tools.
        """
        return self._tools


class _PredictiveDelegationRaceState(Enum):
    """Where the predictive delegation race for a user's turn stands.

    The race is between a classifier's "yes" ("this turn needs the backend")
    and the frontend's first output for the turn.

    - ``PENDING``: the frontend has been asked to reply to the user's turn,
      but hasn't started its inference yet: normally for only a moment, longer
      if it's still busy with an earlier response. A classifier "yes" now is
      held until the inference starts.
    - ``RACING``: the frontend's inference is running and hasn't produced any
      output yet. A classifier "yes" now, or one held from ``PENDING``,
      delegates the turn early.
    - ``FRONTEND_WON``: the frontend's inference produced output before any
      classifier "yes", or the turn was interrupted. The frontend handles the
      turn as usual.
    - ``CLASSIFIER_WON``: a classifier "yes" came first. The turn is delegated
      early, and the frontend's inference is cancelled.
    """

    PENDING = "pending"
    RACING = "racing"
    FRONTEND_WON = "frontend won"
    CLASSIFIER_WON = "classifier won"


@dataclass
class _EarlyDelegationFrame(SystemFrame):
    """Asks the frontend to delegate a user's turn now, in place of its own inference.

    Sent by PredictiveDelegation when its classifier says "yes" ("this turn
    needs the backend"). If the frontend's inference for the turn hasn't
    produced anything yet, the frontend makes the given calls and cancels the
    inference; otherwise the frame does nothing. A system frame, so the
    frontend handles it at once, while the inference runs on another task.

    Parameters:
        turn: The context frame of the user's turn.
        function_calls: The calls made in place of the inference (the
            ``delegate`` call).
    """

    turn: LLMContextFrame | None = None
    function_calls: list[FunctionCallFromLLM] = field(default_factory=list)


class PredictiveDelegationRaceLLMExtension(BaseLLMExtension):
    """Decides predictive delegation's race for a user's turn.

    The race is between a classifier's "yes" ("this turn needs the backend")
    and the frontend's first output for the turn (see
    :mod:`pipecat.pipeline.llm_with_backend.predictive_delegation`). It is
    decided inside the frontend, where the inference's start and its first
    output can be seen. The work has two sides:

    - A :class:`~pipecat.pipeline.llm_with_backend.PredictiveDelegation`
      builds the extension and installs it on the frontend, starts a race for
      each turn (:meth:`start_race`), and sends its classifier's "yes" as an
      early delegation frame. A classifier "no" plays no part in the race.
    - The extension decides the race. A "yes" that arrives before the
      frontend's first output delegates the turn early; otherwise the
      frontend handles the turn as usual. The frontend is normally idle when
      a turn ends (the user's speech interrupts it), so its inference starts
      at once; a "yes" that comes while it is still busy is held until it
      starts.

    Early delegation makes the frame's calls (:meth:`run_function_calls`),
    then cancels the frontend's inference
    (:meth:`cancel_inference_in_progress`), which keeps the frontend's queue,
    including the acknowledgement run the call starts. Downstream sees an
    ordinary response holding the call; any other output the inference pushes
    before it's cancelled is dropped.

    The frontend's inference is recognized by the task its frames and calls
    come from, the frontend's process task; everything else the frontend
    pushes is left alone. All checks are synchronous, so the frontend's first
    output and early delegation can't interleave.

    NOT for an application to attach: on its own, the extension is half the
    feature. A ``PredictiveDelegation`` starts each race and sends its
    classifier's "yes", and ``LLMWithBackend(predictive_delegation=...)`` is
    how an application turns predictive delegation on.
    """

    # The race reads what the frontend's inference pushes before any other
    # extension filters it, and what it drops no other extension sees.
    order = BaseLLMExtension.ORDER_INNERMOST

    def __init__(self, **kwargs):
        """Initialize with no race started.

        Args:
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._turn: LLMContextFrame | None = None
        # The frontend's process task, which runs its inference for the turn
        # and goes on to process the frames after it.
        self._frontend_inference_task: asyncio.Task | None = None
        # None until the first user turn starts a race; after that, the latest
        # race's state, which stays put once decided until the next turn.
        self._state: _PredictiveDelegationRaceState | None = None
        # A "yes" that arrived while PENDING, held until the inference starts
        # its response.
        self._held_early_delegation: _EarlyDelegationFrame | None = None

    def start_race(self, frame: LLMContextFrame) -> None:
        """Start a race for a user's turn.

        Called just before the turn is passed to the frontend: the race
        starts before the frontend reaches the turn, which may wait in its
        queue behind an earlier response.

        Args:
            frame: The context frame of the user's turn.
        """
        self._turn = frame
        self._frontend_inference_task = None
        self._state = _PredictiveDelegationRaceState.PENDING
        self._held_early_delegation = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Follow the race through the frames the frontend processes.

        Racing starts when the frontend's inference for the turn starts, an
        interruption leaves the turn to the frontend, and an early delegation
        frame is carried out while racing.

        Args:
            frame: The frame the frontend is processing.
            direction: The direction of frame processing.
        """
        if isinstance(frame, LLMContextFrame):
            if frame is self._turn and self._state == _PredictiveDelegationRaceState.PENDING:
                self._frontend_inference_task = asyncio.current_task()
                self._state = _PredictiveDelegationRaceState.RACING
        elif isinstance(frame, InterruptionFrame):
            # The interruption cancels the inference: there is nothing left
            # for early delegation to replace.
            if self._state in (
                _PredictiveDelegationRaceState.PENDING,
                _PredictiveDelegationRaceState.RACING,
            ):
                self._state = _PredictiveDelegationRaceState.FRONTEND_WON
        elif isinstance(frame, _EarlyDelegationFrame):
            await self._delegate_early(frame)

    async def filter_push_frame(self, frame: Frame, direction: FrameDirection) -> list[Frame]:
        """Pass on what the frontend pushes, noting its inference's first output.

        Args:
            frame: The frame the frontend is pushing.
            direction: The direction of frame pushing.

        Returns:
            The frame, or nothing for what the inference pushes after the turn
            was delegated early and before its cancellation lands, other than
            the end of its response.
        """
        if not self._called_from_frontend_inference():
            return [frame]
        self._note_frontend_inference_push(frame)
        if self._state == _PredictiveDelegationRaceState.CLASSIFIER_WON and not isinstance(
            frame, (SystemFrame, LLMFullResponseEndFrame)
        ):
            return []
        return [frame]

    async def filter_function_calls(
        self, function_calls: Sequence[FunctionCallFromLLM]
    ) -> Sequence[FunctionCallFromLLM]:
        """Pass on the function calls the frontend is about to run, noting its inference's first.

        Args:
            function_calls: The function calls about to run.

        Returns:
            The calls, or none for calls the inference reaches after the turn
            was delegated early and before its cancellation lands.
        """
        if not self._called_from_frontend_inference():
            return function_calls
        self._note_frontend_inference_function_calls()
        if self._state == _PredictiveDelegationRaceState.CLASSIFIER_WON:
            return []
        return function_calls

    def _called_from_frontend_inference(self) -> bool:
        """Whether the caller is the frontend's process task, running its inference for the turn."""
        return (
            self._frontend_inference_task is not None
            and asyncio.current_task() is self._frontend_inference_task
        )

    def _note_frontend_inference_push(self, frame: Frame) -> None:
        """Note a frame the frontend's inference pushes while racing.

        The start of its response carries out an early delegation held from
        ``PENDING``. Any other output but a system frame is the inference's
        first, and leaves the turn to the frontend.
        """
        if self._state != _PredictiveDelegationRaceState.RACING:
            return
        if isinstance(frame, LLMFullResponseStartFrame):
            if self._held_early_delegation:
                # Carried out on the frontend's input task, like any early
                # delegation frame, once this start frame is downstream.
                held, self._held_early_delegation = self._held_early_delegation, None
                self.create_task(self.queue_frame(held))
        elif not isinstance(frame, SystemFrame):
            self._state = _PredictiveDelegationRaceState.FRONTEND_WON

    def _note_frontend_inference_function_calls(self) -> None:
        """Note the frontend's inference reaching function calls.

        While racing, its first call leaves the turn to the frontend.
        """
        if self._state == _PredictiveDelegationRaceState.RACING:
            self._state = _PredictiveDelegationRaceState.FRONTEND_WON

    async def _delegate_early(self, frame: _EarlyDelegationFrame) -> None:
        """If the turn is racing, make the frame's calls and cancel the inference.

        A frame that arrives while the turn is pending is held until the
        inference starts.
        """
        turn = frame.turn
        if turn is None or turn is not self._turn:
            return
        if self._state == _PredictiveDelegationRaceState.PENDING:
            logger.debug("Predictive delegation: turn pending; holding the early delegation")
            self._held_early_delegation = frame
            return
        if self._state != _PredictiveDelegationRaceState.RACING:
            logger.debug("Predictive delegation: not delegating early (the race is decided)")
            return
        logger.debug("Predictive delegation: delegating early, cancelling the frontend's inference")
        self._state = _PredictiveDelegationRaceState.CLASSIFIER_WON
        await self.run_function_calls(frame.function_calls)
        await self.cancel_inference_in_progress(turn)
