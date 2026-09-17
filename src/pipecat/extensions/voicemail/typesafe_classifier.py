#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Voicemail classification with a TypeSafe ``Choice`` instead of an LLM.

:class:`TypeSafeClassificationProcessor` takes the place of the LLM and the
text-parsing :class:`~pipecat.extensions.voicemail.voicemail_detector.ClassificationProcessor`
in the classification branch of a
:class:`~pipecat.extensions.voicemail.voicemail_detector.VoicemailDetector`.
At the end of each caller turn it asks TypeSafe's Jev one question about the
transcript so far: did a live person answer, or a recording? The answer
arrives as probabilities, so a verdict can be held back until the model is
sure enough, and nothing has to be generated or parsed.
"""

import asyncio
from typing import Any

from loguru import logger

from pipecat.extensions.voicemail.voicemail_detector import ClassificationProcessor
from pipecat.frames.frames import CancelFrame, EndFrame, Frame, LLMContextFrame, StartFrame
from pipecat.metrics.metrics import LLMTokenUsage, MetricsData
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.typesafe.judge import Choice, JudgeResult, TypeSafeJudge

CONVERSATION = "conversation"
"""Choice option: a live person answered."""

VOICEMAIL = "voicemail"
"""Choice option: a recording or automated system answered."""

QUESTION_ID = "answered_by"
"""Question id of the ``Choice`` in every request this processor sends."""

DEFAULT_INSTRUCTIONS = (
    "A bot placed an outbound phone call and `speech` is a speech-to-text transcript of "
    "everything heard so far from the side that answered. Did a live person answer, or "
    "did the call reach a voicemail or other automated system?"
)

DEFAULT_CRITERIA = {
    CONVERSATION: (
        "A live person answered: a personal greeting such as 'Hello?', 'Hi', 'Yeah?' or "
        "'John speaking'; a question to the caller such as 'Who is this?' or 'Can I help "
        "you?'; or other spontaneous speech that expects a reply"
    ),
    VOICEMAIL: (
        "A recording or automated system: a voicemail greeting such as 'you've reached', "
        "'not available right now', 'leave a message', 'leave your name and number' or "
        "'I'll get back to you'; a carrier message such as 'not in service', 'mailbox is "
        "full' or 'has not been set up'; or a business message such as 'our office is "
        "currently closed'"
    ),
}


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") for part in content if isinstance(part, dict) and "text" in part
        )
    return ""


def caller_transcript(context: LLMContext) -> str:
    """Join every user message in the context into one transcript.

    Args:
        context: The classification branch's context, which holds only what
            the answering side has said.

    Returns:
        The caller side's speech so far, one turn per line.
    """
    lines = [
        _message_text(message.get("content", ""))
        for message in context.messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    return "\n".join(line for line in lines if line)


class TypeSafeClassificationProcessor(ClassificationProcessor):
    """Decides voicemail vs. conversation from a TypeSafe ``Choice`` on the caller transcript.

    Placed directly after the classification branch's user context aggregator.
    Each :class:`~pipecat.frames.frames.LLMContextFrame` the aggregator emits
    ends a caller turn; the processor consumes it, sends the transcript so far
    to TypeSafe, and acts on the answer:

    - ``conversation`` at or above ``confidence_threshold``: the conversation
      proceeds (see :meth:`ClassificationProcessor.conversation_detected`).
    - ``voicemail`` at or above ``confidence_threshold``: voicemail handling
      starts (see :meth:`ClassificationProcessor.voicemail_detected`).
    - a verdict below the threshold, or a failed request: no decision yet.
      The next caller turn is judged with the longer transcript.

    A turn that ends while a judgment is still in flight cancels it; the new
    turn's transcript includes the old one.
    """

    def __init__(
        self,
        *,
        judge: TypeSafeJudge,
        confidence_threshold: float = 0.5,
        instructions: str = DEFAULT_INSTRUCTIONS,
        criteria: dict[str, str] | None = None,
        warm_up: bool = True,
        **kwargs,
    ):
        """Initialize the processor.

        Args:
            judge: The TypeSafe client. Started on ``StartFrame`` and closed on
                ``EndFrame`` or ``CancelFrame``.
            confidence_threshold: Minimum confidence to act on a verdict. Below
                it the processor waits for the caller side to say more. 0 acts
                on every verdict.
            instructions: The question asked about the transcript, which the
                state exposes as `speech`.
            criteria: Descriptions of the two options, keyed ``conversation``
                and ``voicemail``. Defaults to :data:`DEFAULT_CRITERIA`.
            warm_up: Whether to open the TypeSafe connection on ``StartFrame``
                so the first judgment does not pay for a TLS handshake.
            **kwargs: Arguments for :class:`ClassificationProcessor`: the three
                notifiers and ``voicemail_response_delay``.
        """
        super().__init__(**kwargs)
        self._judge = judge
        self._confidence_threshold = confidence_threshold
        self._warm_up = warm_up
        options = dict(criteria) if criteria is not None else dict(DEFAULT_CRITERIA)
        missing = {CONVERSATION, VOICEMAIL} - set(options)
        if missing:
            raise ValueError(f"criteria must describe both options; missing {sorted(missing)}")
        self._question = Choice(instructions=instructions, criteria=options)
        self._judge_task: asyncio.Task | None = None

        self.set_core_metrics_data(MetricsData(processor=self.name, model=judge.model))

    def can_generate_metrics(self) -> bool:
        """Processing time and token usage are reported per judgment."""
        return True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Judge each caller turn; pass every other frame to the base class.

        Args:
            frame: The frame to process.
            direction: The direction the frame is moving in the pipeline.
        """
        if isinstance(frame, StartFrame):
            self._judge.start()
            if self._warm_up:
                self.create_task(self._judge.warm_up())
            await super().process_frame(frame, direction)
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._cancel_judgment()
            await super().process_frame(frame, direction)
            await self._judge.close()
        elif isinstance(frame, LLMContextFrame):
            # The context frame is this branch's turn-end signal. It never
            # continues downstream: the branch's output rejoins the main
            # pipeline, where it would run the conversation LLM on the
            # classifier's context.
            if direction == FrameDirection.DOWNSTREAM and not frame.speculation:
                if not self.decision_made:
                    await self._cancel_judgment()
                    self._judge_task = self.create_task(self._judge_turn(frame.context))
        else:
            await super().process_frame(frame, direction)

    async def cleanup(self):
        """Clean up the processor resources."""
        await self._cancel_judgment()
        await super().cleanup()

    async def _cancel_judgment(self):
        if self._judge_task:
            await self.cancel_task(self._judge_task)
            self._judge_task = None

    async def _judge_turn(self, context: LLMContext):
        state = {"speech": caller_transcript(context)}
        try:
            await self.start_processing_metrics()
            result = await self._judge.ask(state, {QUESTION_ID: self._question})
            await self.stop_processing_metrics()
            await self._report_usage(result)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            await self.push_error(f"TypeSafe voicemail judgment failed: {e}", e)
            return
        finally:
            self._judge_task = None

        decision = result.choices.get(QUESTION_ID)
        if decision is None:
            logger.warning(f"{self}: TypeSafe response had no {QUESTION_ID!r} answer")
            return
        probabilities = {k: round(v, 2) for k, v in decision.probabilities.items()}
        logger.debug(
            f"{self}: judged {state['speech']!r} in {result.latency_secs * 1000:.0f}ms: "
            f"{decision.choice} at confidence {decision.confidence:.2f} {probabilities}"
        )
        if decision.confidence < self._confidence_threshold:
            logger.debug(f"{self}: below {self._confidence_threshold:.2f}; waiting for more speech")
            return
        if decision.choice == VOICEMAIL:
            await self.voicemail_detected()
        elif decision.choice == CONVERSATION:
            await self.conversation_detected()
        else:
            logger.warning(f"{self}: unexpected option {decision.choice!r}")

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
