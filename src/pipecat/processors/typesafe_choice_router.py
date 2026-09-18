#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Route a user turn on a TypeSafe ``Choice`` judgment instead of an LLM completion.

Some turns need a decision, not a reply: a yes/no confirmation, a pick from a
short menu. Running the conversation LLM for those costs a full completion
of latency and money. :class:`TypeSafeChoiceRouter` sits between the user
context aggregator and the LLM, asks TypeSafe which option the user's reply
matches, and hands a confident answer to application code. When the code
handles it, the LLM never runs for that turn; otherwise the turn continues to
the LLM as usual.
"""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    StartFrame,
    UserStartedSpeakingFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage, MetricsData
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.typesafe.judge import Choice, JudgeResult, Question, TypeSafeJudge

ChoiceHandler = Callable[[JudgeResult, LLMContext], Awaitable[bool]]
"""Receives a confident judgment. Returns True when the turn was handled and the LLM should not run."""

FallthroughHandler = Callable[[JudgeResult | None, LLMContext], Awaitable[bool]]
"""Receives a turn the routing question did not settle. The result is None when the request failed."""

StateBuilder = Callable[[LLMContext], str | Mapping[str, Any] | list]
"""Builds the TypeSafe state for a turn from the LLM context."""

ROUTE_QUESTION_ID = "route"
"""Question id of the routing ``Choice`` inside every request this processor sends."""


def _message_text(content: Any) -> str:
    """Flatten a standard message's content to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") for part in content if isinstance(part, dict) and "text" in part
        )
    return ""


def default_state_builder(context: LLMContext) -> dict[str, str]:
    """Build the state from the last bot message and the latest user message.

    Args:
        context: The LLM context at the end of the user's turn.

    Returns:
        A dict with ``bot_message`` and ``user_reply`` fields. Either is empty
        when the context has no such message.
    """
    bot_message = ""
    user_reply = ""
    for message in reversed(context.messages):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "user" and not user_reply:
            user_reply = _message_text(message.get("content", ""))
        elif role == "assistant" and not bot_message:
            bot_message = _message_text(message.get("content", ""))
        if user_reply and bot_message:
            break
    return {"bot_message": bot_message, "user_reply": user_reply}


class TypeSafeChoiceRouter(FrameProcessor):
    """Intercepts the end of a user turn and routes it on a TypeSafe ``Choice``.

    Place it between ``context_aggregator.user()`` and the LLM. For each
    non-speculative :class:`~pipecat.frames.frames.LLMContextFrame` while
    :attr:`is_active` returns True, the router holds the frame, asks TypeSafe
    the routing question (plus any ``extra_questions``, answered in the same
    request), and:

    - calls ``on_choice`` when the chosen option is not the fallback and its
      confidence reaches ``confidence_threshold``. If the handler returns True
      the held frame is dropped and the LLM does not run this turn.
    - otherwise calls ``on_fallthrough``, when one is set, with the judgment
      (or None if the request failed). If it returns True the frame is dropped.
    - otherwise pushes the held frame so the LLM answers as usual.

    A pipeline with no LLM at all sets ``on_fallthrough`` and answers every
    turn itself, for example with a canned re-prompt. If the user starts
    speaking again while a judgment is in flight, the judgment is cancelled and
    the held frame dropped; the aggregator will send a new one when the turn
    ends.

    Confidence describes how strongly the model prefers the chosen option. It
    says nothing about whether acting on that option is what the user wants.
    A reply like "yes, but does it cost anything?" is a confident "yes" that
    still needs the LLM. Add a ``Noul`` to ``extra_questions`` for such cases
    and let ``on_choice`` return False when it fires.
    """

    def __init__(
        self,
        *,
        judge: TypeSafeJudge,
        instructions: str,
        criteria: Mapping[str, str],
        on_choice: ChoiceHandler,
        on_fallthrough: FallthroughHandler | None = None,
        is_active: Callable[[], bool] = lambda: True,
        confidence_threshold: float = 0.7,
        fallback_option: str | None = "other",
        fallback_description: str = "Neither of the listed options: unclear, off topic, or a question",
        extra_questions: Mapping[str, Question] | None = None,
        state_builder: StateBuilder = default_state_builder,
        warm_up: bool = True,
        **kwargs,
    ):
        """Initialize the router.

        Args:
            judge: The TypeSafe client. The router starts it on ``StartFrame``
                and closes it on ``EndFrame`` or ``CancelFrame``.
            instructions: The routing question, for example "How did the user
                answer the question in `bot_message`? Judge `user_reply`."
            criteria: Options keyed by option id, each with a description of
                the replies it covers.
            on_choice: Called with the judgment and the LLM context when the
                choice is confident. Return True to consume the turn.
            on_fallthrough: Called when the turn was not consumed by
                ``on_choice``: the fallback option won, confidence was too low,
                the handler returned False, or the request failed (then the
                judgment is None). Return True to consume the turn. Without it
                such turns continue downstream to the LLM.
            is_active: Whether the router should judge the current turn. Turns
                arriving while this returns False pass straight to the LLM.
            confidence_threshold: Minimum confidence for ``on_choice`` to be
                called.
            fallback_option: Option id added to ``criteria`` for replies that
                match none of them. None disables it; then the model must pick
                one of your options even when nothing fits.
            fallback_description: Description of the fallback option.
            extra_questions: Additional questions asked in the same request,
                keyed by your own ids. Read them from the result in ``on_choice``.
            state_builder: Builds the request state from the LLM context.
            warm_up: Whether to open the TypeSafe connection on ``StartFrame``
                so the first judgment does not pay for a TLS handshake.
            **kwargs: Additional arguments passed to :class:`FrameProcessor`.
        """
        super().__init__(**kwargs)
        self._judge = judge
        self._on_choice = on_choice
        self._on_fallthrough = on_fallthrough
        self._is_active = is_active
        self._confidence_threshold = confidence_threshold
        self._fallback_option = fallback_option
        self._state_builder = state_builder
        self._warm_up = warm_up

        options = dict(criteria)
        if fallback_option is not None:
            options[fallback_option] = fallback_description
        self._questions: dict[str, Question] = {
            ROUTE_QUESTION_ID: Choice(instructions=instructions, criteria=options)
        }
        if extra_questions:
            if ROUTE_QUESTION_ID in extra_questions:
                raise ValueError(
                    f"extra_questions may not use the reserved id {ROUTE_QUESTION_ID!r}"
                )
            self._questions.update(extra_questions)

        self._judge_task: asyncio.Task | None = None
        self._held_frame: LLMContextFrame | None = None

        self.set_core_metrics_data(MetricsData(processor=self.name, model=judge.model))

    def can_generate_metrics(self) -> bool:
        """Processing time and token usage are reported per judgment."""
        return True

    @property
    def is_active(self) -> bool:
        """Whether the router judges turns right now."""
        return self._is_active()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Hold the turn-end context frame while TypeSafe judges it; pass everything else.

        Args:
            frame: The frame to process.
            direction: The direction the frame is moving in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._start()
            await self.push_frame(frame, direction)
        elif isinstance(frame, EndFrame):
            await self._release_held_frame()
            await self.push_frame(frame, direction)
            await self._judge.close()
        elif isinstance(frame, CancelFrame):
            await self._drop_held_frame()
            await self.push_frame(frame, direction)
            await self._judge.close()
        elif isinstance(frame, (InterruptionFrame, UserStartedSpeakingFrame)):
            await self._drop_held_frame()
            await self.push_frame(frame, direction)
        elif (
            isinstance(frame, LLMContextFrame)
            and direction == FrameDirection.DOWNSTREAM
            and not frame.speculation
            and self.is_active
        ):
            await self._drop_held_frame()
            self._held_frame = frame
            self._judge_task = self.create_task(self._judge_turn(frame))
        else:
            await self.push_frame(frame, direction)

    async def _start(self):
        self._judge.start()
        if self._warm_up:
            self.create_task(self._judge.warm_up())

    async def _drop_held_frame(self):
        """Cancel a judgment in flight and forget the frame it was about."""
        self._held_frame = None
        if self._judge_task:
            await self.cancel_task(self._judge_task)
            self._judge_task = None

    async def _release_held_frame(self):
        """Cancel a judgment in flight and let its frame continue to the LLM."""
        frame = self._held_frame
        await self._drop_held_frame()
        if frame:
            await self.push_frame(frame)

    async def _judge_turn(self, frame: LLMContextFrame):
        handled = False
        result: JudgeResult | None = None
        try:
            state = self._state_builder(frame.context)
            await self.start_processing_metrics()
            result = await self._judge.ask(state, self._questions)
            await self.stop_processing_metrics()
            await self._report_usage(result)
            logger.debug(
                f"{self}: judged {state!r} in {result.latency_secs * 1000:.0f}ms: "
                f"{ {k: v.probabilities for k, v in result.choices.items()} } "
                f"{ {k: round(v.probability, 2) for k, v in result.nouls.items()} }"
            )
            handled = await self._decide(result, frame.context)
        except asyncio.CancelledError:
            # The user resumed speaking or the pipeline is ending; whoever
            # cancelled us has already decided what happens to the frame.
            raise
        except Exception as e:
            await self.push_error(f"TypeSafe judgment failed: {e}", e)

        if not handled and self._on_fallthrough is not None:
            handled = await self._on_fallthrough(result, frame.context)

        self._judge_task = None
        if self._held_frame is frame:
            self._held_frame = None
            if not handled:
                await self.push_frame(frame)

    async def _decide(self, result: JudgeResult, context: LLMContext) -> bool:
        decision = result.choices.get(ROUTE_QUESTION_ID)
        if decision is None:
            logger.warning(f"{self}: TypeSafe response had no {ROUTE_QUESTION_ID!r} answer")
            return False
        if decision.choice == self._fallback_option:
            logger.debug(f"{self}: no option matched")
            return False
        if decision.confidence < self._confidence_threshold:
            logger.debug(
                f"{self}: {decision.choice!r} at confidence {decision.confidence:.2f} is below "
                f"{self._confidence_threshold:.2f}"
            )
            return False
        return await self._on_choice(result, context)

    async def _report_usage(self, result: JudgeResult):
        if result.input_tokens is None and result.output_tokens is None:
            return
        prompt = result.input_tokens or 0
        completion = result.output_tokens or 0
        await self.start_llm_usage_metrics(
            LLMTokenUsage(
                prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
            )
        )
