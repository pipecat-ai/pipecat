#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""An LLM service that answers each turn with one label, chosen by TypeSafe.

Several Pipecat components take an LLM whose only job is to answer with one
word from a fixed set: the voicemail detector wants "CONVERSATION" or
"VOICEMAIL", a routing prompt wants the name of a branch. Running a text LLM
for that costs a completion and a prompt that begs the model to answer with
exactly one word. :class:`TypeSafeChoiceLLMService` stands in the same place
and asks TypeSafe's Jev one ``Choice`` instead: the chosen option's label goes
out as the LLM response, nothing else is generated, and the probability of
each option is known, so a weak verdict can be answered with silence and left
to the next turn.

Requires the ``typesafe`` extra: ``uv add "pipecat-ai[typesafe]"``.
"""

import asyncio
from collections.abc import Callable, Mapping
from typing import Any

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    StartFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings
from pipecat.services.typesafe.judge import Choice, JudgeResult, TypeSafeError, TypeSafeJudge

QUESTION_ID = "label"
"""Question id of the ``Choice`` in every request this service sends."""

StateBuilder = Callable[[LLMContext], str | Mapping[str, Any] | list]
"""Builds the TypeSafe state for a turn from the LLM context."""


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") for part in content if isinstance(part, dict) and "text" in part
        )
    return ""


def transcript_state_builder(context: LLMContext) -> dict[str, str]:
    """Build the state from every user message in the context.

    The service's own labels land in the context as assistant messages, and
    a classifier prompt may sit there as a system or developer message;
    neither is part of what the user said.

    Args:
        context: The LLM context at the end of the user's turn.

    Returns:
        A dict with one ``transcript`` field: the user's turns so far, one per
        line.
    """
    lines = [
        _message_text(message.get("content", ""))
        for message in context.messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    return {"transcript": "\n".join(line for line in lines if line)}


class TypeSafeChoiceLLMService(LLMService):
    """Answers every context frame with one of a fixed set of labels.

    On each :class:`~pipecat.frames.frames.LLMContextFrame` the service builds
    a state from the context, asks TypeSafe which of the ``criteria`` it
    matches, and emits the chosen label as a complete LLM response:
    ``LLMFullResponseStartFrame``, one ``LLMTextFrame`` with the label, and
    ``LLMFullResponseEndFrame``. Whatever reads the LLM's text downstream sees
    exactly what a well-behaved classifier LLM would have said.

    A verdict below ``confidence_threshold``, or a failed request, produces an
    empty response (start and end frames with no text) so the turn completes
    without a verdict. An interruption cancels the judgment in flight, as it
    would cut off a text LLM's completion.

    Example::

        classifier = TypeSafeChoiceLLMService(
            judge=TypeSafeJudge(),
            instructions="Is `transcript` a person who picked up, or a recording?",
            criteria={"CONVERSATION": "A live person answered", "VOICEMAIL": "A recording"},
        )
        detector = VoicemailDetector(llm=classifier)
    """

    def __init__(
        self,
        *,
        judge: TypeSafeJudge,
        instructions: str,
        criteria: Mapping[str, str],
        confidence_threshold: float = 0.0,
        state_builder: StateBuilder = transcript_state_builder,
        warm_up: bool = True,
        **kwargs,
    ):
        """Initialize the service.

        Args:
            judge: The TypeSafe client wrapper. The service starts it, warms
                it up, and closes it.
            instructions: The question asked about the state. Refer to state
                fields with backticks, for example `transcript`.
            criteria: The labels the service can answer with, each with a
                description of what matches it. The key is the exact text
                emitted.
            confidence_threshold: A verdict below this confidence is answered
                with an empty response. 0 answers with every verdict.
            state_builder: Builds the request state from the LLM context.
            warm_up: Whether to open the TypeSafe connection on start.
            **kwargs: Passed to :class:`~pipecat.services.llm_service.LLMService`.
        """
        super().__init__(
            settings=LLMSettings(
                model=judge.model,
                system_instruction=None,
                temperature=None,
                max_tokens=None,
                top_p=None,
                top_k=None,
                frequency_penalty=None,
                presence_penalty=None,
                seed=None,
                filter_incomplete_user_turns=None,
                user_turn_completion_config=None,
            ),
            **kwargs,
        )
        if not criteria:
            raise ValueError("criteria needs at least one label")
        self._judge = judge
        self._question = Choice(instructions=instructions, criteria=dict(criteria))
        self._confidence_threshold = confidence_threshold
        self._state_builder = state_builder
        self._warm_up = warm_up
        self._task: asyncio.Task | None = None

    def can_generate_metrics(self) -> bool:
        """Whether this service reports processing and usage metrics."""
        return True

    async def start(self, frame: StartFrame):
        """Start the service and open the TypeSafe connection."""
        await super().start(frame)
        self._judge.start()
        if self._warm_up:
            self.create_task(self._judge.warm_up())

    async def stop(self, frame: EndFrame):
        """Stop the service and close the TypeSafe connection."""
        await super().stop(frame)
        await self._cancel_turn()
        await self._judge.close()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service and close the TypeSafe connection."""
        await super().cancel(frame)
        await self._cancel_turn()
        await self._judge.close()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Answer context frames; pass everything else through."""
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            await self._cancel_turn()
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMContextFrame):
            await self._cancel_turn()
            self._task = self.create_task(self._respond(frame))
        else:
            await self.push_frame(frame, direction)

    async def _cancel_turn(self) -> None:
        if self._task is not None:
            task, self._task = self._task, None
            await self.cancel_task(task)
            # A judgment that never arrived has no first byte or processing
            # time to report, and must not be measured against the next turn.
            await self.cancel_ttfb_metrics()
            await self.stop_processing_metrics()

    async def _respond(self, frame: LLMContextFrame) -> None:
        try:
            await self.start_processing_metrics()
            label = await self._choose(frame.context)
            await self.push_frame(LLMFullResponseStartFrame())
            if label is not None:
                await self._push_llm_text(label)
            await self.push_frame(LLMFullResponseEndFrame())
            await self.stop_processing_metrics()
        finally:
            self._task = None

    async def _choose(self, context: LLMContext) -> str | None:
        state = self._state_builder(context)
        try:
            # TTFB runs from the request to the judgment's arrival, which is
            # the whole response.
            await self.start_ttfb_metrics()
            result = await self._judge.ask(state, {QUESTION_ID: self._question})
            await self.stop_ttfb_metrics()
        except (TypeSafeError, TimeoutError, OSError) as e:
            await self.cancel_ttfb_metrics()
            await self.push_error(f"TypeSafe judgment failed: {e}", e)
            return None

        await self._report_usage(result)
        decision = result.choices.get(QUESTION_ID)
        if decision is None:
            logger.warning(f"{self}: TypeSafe response had no {QUESTION_ID!r} answer")
            return None
        probabilities = {k: round(v, 2) for k, v in decision.probabilities.items()}
        logger.debug(
            f"{self}: judged {state!r} in {result.latency_secs * 1000:.0f}ms: "
            f"{decision.choice} at confidence {decision.confidence:.2f} {probabilities}"
        )
        if decision.confidence < self._confidence_threshold:
            logger.debug(f"{self}: below {self._confidence_threshold:.2f}; answering nothing")
            return None
        return decision.choice

    async def _report_usage(self, result: JudgeResult) -> None:
        if result.input_tokens is None and result.output_tokens is None:
            return
        prompt = result.input_tokens or 0
        completion = result.output_tokens or 0
        await self.start_llm_usage_metrics(
            LLMTokenUsage(
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=prompt + completion,
            )
        )
