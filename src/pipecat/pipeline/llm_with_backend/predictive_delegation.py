#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Predictive delegation, an optimization that can speed up responses that need the backend.

Pass a :class:`PredictiveDelegation` as
``LLMWithBackend(predictive_delegation=...)`` to turn it on. Normally, a
text frontend spends a whole inference deciding to call ``delegate`` before
the backend can start. Predictive delegation uses a fast yes/no classifier
to determine whether it needs to wait for that inference or can go ahead and
trigger an early backend delegation.

How it works: when a user's turn ends, the frontend runs in parallel (i.e.
in a race) with the classifier (see :mod:`pipecat.classifiers`) that decides
whether the turn needs the backend.

- If the classifier says "yes" before the frontend has produced output:

  - the frontend's inference is cancelled;
  - a ``delegate`` call to the backend is made right away, in its place.

- Otherwise, the frontend handles the turn as usual, having waited for
  nothing (note that it may still choose to ``delegate``).
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger

from pipecat.classifiers.base_classifier import BaseClassifier, YesNoQuestion
from pipecat.frames.frames import (
    Frame,
    FunctionCallFromLLM,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    MetricsFrame,
    SystemFrame,
    UserStartedSpeakingFrame,
)
from pipecat.metrics.metrics import MetricsData
from pipecat.pipeline.llm_with_backend.connector import DELEGATE_TOOL_NAME, BackendConnector
from pipecat.pipeline.llm_with_backend.prompts import (
    BACKEND_MESSAGE_RELAY_INSTRUCTION,
    _is_backend_message,
)
from pipecat.processors.aggregators.llm_context import LLMContextMessage
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.services.llm_service import LLMService, _OutputGate
from pipecat.services.settings import LLMSettings

# ---------------------------------------------------------------------------
# _PredictiveDelegationRace tracks whether a classifier "yes" arrives before the
# frontend's first output, and so whether the turn is delegated early or
# handled by the frontend as usual. A classifier "no" plays no part. The
# frontend is normally idle when a turn ends (the user's speech interrupts
# it), so its inference starts at once; a "yes" that comes while it is still
# busy is held until it starts. The race is attached to the frontend as its
# output gate (`_OutputGate` in `pipecat.services.llm_service`), which the
# frontend consults from its process_frame, push_frame and run_function_calls.
# ---------------------------------------------------------------------------


class _PredictiveDelegationRaceState(Enum):
    """Where the predictive delegation race for a user's turn stands.

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

    Sent by PredictiveDelegation when its classifier says "yes". If the
    frontend's inference for the turn hasn't produced anything yet, the
    frontend makes the given calls and cancels the inference; otherwise the
    frame does nothing. A system frame, so the frontend handles it at once,
    while the inference runs on another task.

    Parameters:
        turn: The context frame of the user's turn.
        function_calls: The calls made in place of the inference (the
            ``delegate`` call).
    """

    turn: LLMContextFrame | None = None
    function_calls: list[FunctionCallFromLLM] = field(default_factory=list)


class _PredictiveDelegationRace(_OutputGate):
    """Tracks whether a classifier "yes" arrives before the frontend's first output for a turn.

    See the module docstring and the comment above. Early delegation makes
    the frame's calls, then cancels the frontend's inference with
    ``FrameProcessor._cancel_frame_in_progress``, which keeps the frontend's
    queue, including the acknowledgement run the call starts. Downstream sees
    an ordinary response holding the call; any other output the inference
    pushes before it's cancelled is dropped.

    The frontend's inference is recognized by the task its frames and calls
    come from, the frontend's process task; everything else the frontend
    pushes is left alone. All checks are synchronous, so the frontend's first
    output and early delegation can't interleave.
    """

    def __init__(self):
        """Initialize with no race started."""
        self._turn: LLMContextFrame | None = None
        # The frontend's process task, which runs its inference for the turn
        # and goes on to process the frames after it.
        self._frontend_inference_task: asyncio.Task | None = None
        # None until the first user turn starts a race; after that, the latest
        # race's state, which stays put once decided until the next turn.
        self._state: _PredictiveDelegationRaceState | None = None
        self._frontend: LLMService[Any] | None = None
        # A "yes" that arrived while PENDING, held until the inference starts
        # its response.
        self._held_early_delegation: _EarlyDelegationFrame | None = None

    def start(self, frame: LLMContextFrame) -> None:
        """Start a race for a user's turn, just before the turn is passed to the frontend.

        Args:
            frame: The context frame of the user's turn.
        """
        self._turn = frame
        self._frontend_inference_task = None
        self._state = _PredictiveDelegationRaceState.PENDING
        self._held_early_delegation = None

    async def process_frame(self, service: LLMService[Any], frame: Frame) -> None:
        """Follow the race through the frames the frontend processes.

        Implements ``_OutputGate.process_frame``. Racing starts when the
        frontend's inference for the turn starts, an interruption leaves the
        turn to the frontend, and an early delegation frame is carried out
        while racing.

        Args:
            service: The frontend service.
            frame: The frame it is processing.
        """
        if isinstance(frame, LLMContextFrame):
            if frame is self._turn and self._state == _PredictiveDelegationRaceState.PENDING:
                self._frontend = service
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
            await self._delegate_early(service, frame)

    def allow_push(self, frame: Frame) -> bool:
        """Whether the frontend may push a frame; notes its inference's first output.

        Implements ``_OutputGate.allow_push``.

        Args:
            frame: The frame the frontend is about to push.

        Returns:
            Whether the frontend may push the frame: False only for what the
            inference pushes after the turn was delegated early and before its
            cancellation lands, other than the end of its response.
        """
        if not self._called_from_frontend_inference():
            return True
        self._note_frontend_inference_push(frame)
        if self._state == _PredictiveDelegationRaceState.CLASSIFIER_WON:
            return isinstance(frame, (SystemFrame, LLMFullResponseEndFrame))
        return True

    def allow_function_calls(self) -> bool:
        """Whether the frontend may run function calls; notes its inference's first call.

        Implements ``_OutputGate.allow_function_calls``.

        Returns:
            Whether the frontend may run them: False only for calls the
            inference reaches after the turn was delegated early and before
            its cancellation lands.
        """
        if not self._called_from_frontend_inference():
            return True
        self._note_frontend_inference_function_calls()
        return self._state != _PredictiveDelegationRaceState.CLASSIFIER_WON

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
            if self._held_early_delegation and self._frontend:
                # Carried out on the frontend's input task, like any early
                # delegation frame, once this start frame is downstream.
                held, self._held_early_delegation = self._held_early_delegation, None
                self._frontend.create_task(self._frontend.queue_frame(held))
        elif not isinstance(frame, SystemFrame):
            self._state = _PredictiveDelegationRaceState.FRONTEND_WON

    def _note_frontend_inference_function_calls(self) -> None:
        """Note the frontend's inference reaching function calls.

        While racing, its first call leaves the turn to the frontend.
        """
        if self._state == _PredictiveDelegationRaceState.RACING:
            self._state = _PredictiveDelegationRaceState.FRONTEND_WON

    async def _delegate_early(
        self, frontend: LLMService[Any], frame: _EarlyDelegationFrame
    ) -> None:
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
        await frontend.run_function_calls(frame.function_calls)
        await frontend._cancel_frame_in_progress(turn)


#: How likely a classifier "yes" must be for a turn to be delegated early.
_DELEGATION_PREDICTION_THRESHOLD = 0.6

#: How many of the conversation's latest messages the classifier sees.
_DELEGATION_PREDICTION_HISTORY_LENGTH = 12


class PredictiveDelegation(FrameProcessor):
    """An optimization that can speed up responses that need backend involvement.

    **Alpha.** It uses a fast yes/no classifier to determine, after each user
    request, whether it needs to wait for the frontend's inference or can go
    ahead and trigger an early backend delegation. See
    :mod:`pipecat.pipeline.llm_with_backend.predictive_delegation` for how it
    works. Pass one to
    ``LLMWithBackend(predictive_delegation=...)``, which puts it just in front
    of the frontend; it needs a text frontend and a request strategy whose
    ``delegate`` call needs no arguments, such as the default
    :class:`TranscriptBackendRequestStrategy`.

    It races every turn that ends with a new user message, spoken or typed,
    and so also a user-role message the app appends and runs the frontend on.
    The classifier is asked whether that message needs the backend, given
    the frontend's system instruction (where the app says what the backend
    is for, and the pairing instruction when to delegate), the ``delegate``
    tool's description, and the latest messages of the conversation, the
    backend's included. A turn the frontend sees as speculative (eager end of
    turn), or with incomplete-turn filtering on, is left to the frontend.

    Example::

        llm = LLMWithBackend(
            frontend=OpenAILLMService(...),
            backend=BackendLLMWorker(...),
            predictive_delegation=PredictiveDelegation(JevClassifier(api_key=...)),
        )
    """

    def __init__(self, classifier: BaseClassifier, **kwargs):
        """Initialize predictive delegation.

        Args:
            classifier: The yes/no classifier to ask. It answers within a bound
                of its own; an answer that comes after the frontend's first
                output is ignored.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._classifier = classifier
        self._race = _PredictiveDelegationRace()
        self._frontend: LLMService[Any] | None = None
        self._question: YesNoQuestion | None = None
        self._last_raced: LLMContextMessage | None = None
        self._classify_task: asyncio.Task | None = None
        self._warned: set[str] = set()

    def _bind(self, frontend: LLMService[Any], connector: BackendConnector) -> None:
        """Attach the race to the frontend and build the classifier's question."""
        self._frontend = frontend
        frontend._output_gate = self._race
        self._question = YesNoQuestion(
            instructions=(
                "Does the user's latest message need to be handed to the assistant's backend "
                "now? The handoff works like this: " + connector.request_strategy.tool_description
            ),
            no="The assistant's instructions say not to hand this message to the backend.",
        )

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the classifier, reporting its metrics in the pipeline."""
        await super().setup(setup)
        await self._classifier.setup(self.task_manager)
        self._classifier.add_event_handler("on_metrics", self._on_classifier_metrics)

    async def cleanup(self):
        """Stop any classification in progress and clean up the classifier."""
        await self._cancel_classification()
        await self._classifier.cleanup()
        await super().cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Race each new user turn; pass every frame on.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)
        if isinstance(frame, (InterruptionFrame, UserStartedSpeakingFrame)):
            await self._cancel_classification()
        if (
            direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, LLMContextFrame)
            and self._should_start_race(frame)
        ):
            await self._cancel_classification()
            self._race.start(frame)
            state = self._classifier_state(frame)
            await self.push_frame(frame, direction)
            self._classify_task = self.create_task(
                self._classify(frame, state), f"{self}::classify"
            )
            return
        await self.push_frame(frame, direction)

    def _should_start_race(self, frame: LLMContextFrame) -> bool:
        """Whether a context frame should start a race.

        It should when it asks for a reply to a user message that hasn't been
        raced yet, unless the turn is speculative (eager end of turn) or the
        frontend filters incomplete user turns.
        """
        assert self._frontend is not None, "predictive delegation not bound"
        if frame.speculation:
            self._warn_once(
                "speculation",
                "Predictive delegation: speculative turns (eager end of turn) are left to the "
                "frontend",
            )
            return False
        if self._frontend_settings().filter_incomplete_user_turns:
            self._warn_once(
                "filtering",
                "Predictive delegation: turns are left to the frontend while it filters "
                "incomplete user turns",
            )
            return False
        messages = frame.context.get_messages()
        last = messages[-1] if messages else None
        if not isinstance(last, dict) or last.get("role") != "user":
            return False
        if last is self._last_raced:
            return False
        self._last_raced = last
        return True

    def _classifier_state(self, frame: LLMContextFrame) -> dict[str, Any]:
        """What the classifier is told about the turn."""
        # The instruction the frontend's model reads, the pairing instruction
        # included, so the classifier follows the same rules for delegating.
        return {
            "assistant": str(self._frontend_settings().system_instruction or ""),
            "conversation": _render_for_classifier(frame.context.get_messages()),
        }

    async def _classify(self, frame: LLMContextFrame, state: dict[str, Any]) -> None:
        """Ask the classifier about a turn; on a "yes", ask the frontend to delegate it early."""
        assert self._question is not None
        try:
            answers = await self._classifier.yes_no(state, {"needs_backend": self._question})
        except Exception as e:
            logger.warning(f"Predictive delegation: the classifier failed: {e}")
            return
        probability = answers["needs_backend"].probability
        logger.debug(f"Predictive delegation: classifier says {probability:.2f}")
        if probability < _DELEGATION_PREDICTION_THRESHOLD:
            return
        call = FunctionCallFromLLM(
            function_name=DELEGATE_TOOL_NAME,
            tool_call_id=f"early_delegation_{uuid.uuid4().hex[:16]}",
            arguments={},
            context=frame.context,
        )
        await self.push_frame(_EarlyDelegationFrame(turn=frame, function_calls=[call]))

    def _frontend_settings(self) -> LLMSettings:
        assert self._frontend is not None, "predictive delegation not bound"
        settings = self._frontend.settings
        assert isinstance(settings, LLMSettings)
        return settings

    async def _cancel_classification(self) -> None:
        if self._classify_task is not None:
            await self.cancel_task(self._classify_task)
            self._classify_task = None

    async def _on_classifier_metrics(self, classifier: BaseClassifier, data: list[MetricsData]):
        await self.push_frame(MetricsFrame(data=data))

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(message)


def _render_for_classifier(messages: list[LLMContextMessage]) -> list[dict[str, str]]:
    """The conversation's latest messages, as speaker and text, for the classifier."""
    rendered: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role, content = message.get("role"), message.get("content")
        if role == "user" and isinstance(content, str):
            rendered.append({"speaker": "user", "text": content})
        elif role == "assistant":
            if isinstance(content, str) and content:
                rendered.append({"speaker": "assistant", "text": content})
            for call in message.get("tool_calls") or []:
                if call.get("function", {}).get("name") == DELEGATE_TOOL_NAME:
                    rendered.append(
                        {"speaker": "assistant", "action": "handed the request to the backend"}
                    )
        elif _is_backend_message(message) and isinstance(content, str):
            rendered.append(
                {
                    "speaker": "backend",
                    "text": content.removesuffix(f"\n\n{BACKEND_MESSAGE_RELAY_INSTRUCTION}"),
                }
            )
    return rendered[-_DELEGATION_PREDICTION_HISTORY_LENGTH:]
