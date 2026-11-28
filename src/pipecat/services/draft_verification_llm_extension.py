#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The LLM extension that has a second model check each reply and correct it.

A fast model answers the user. A slower, stronger model, the verifier, reads
the same conversation and that answer, and either lets it stand or returns a
corrected one. The extension decides what the correction can still change:
the reply before it is spoken, a short follow-up after it, or nothing, once
the user has moved on.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    SystemFrame,
    UserStartedSpeakingFrame,
    UserTurnResponseCancelFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext, LLMContextMessage
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.base_llm_extension import BaseLLMExtension
from pipecat.services.llm_service import LLMService

DRAFT_VERIFICATION_PROMPT = (
    "You are checking the assistant's last reply in this conversation before the user relies on "
    "it. If the reply is correct and answers what the user asked, set correct to true and repeat "
    "the reply unchanged. If it is wrong, incomplete in a way that matters, or answers the wrong "
    "question, set correct to false and write the reply the assistant should have given: same "
    "voice, same length or shorter, spoken style, no mention of a correction."
)

#: What the extension says after a reply that was already spoken turns out
#: to be wrong. ``{text}`` is the verifier's corrected reply. It is English:
#: a bot that speaks another language passes its own.
LATE_CORRECTION_TEMPLATE = "Sorry, let me correct that. {text}"

# Added after any prompt, so a custom prompt still gets a readable verdict
# from a service that can't enforce the schema.
_VERDICT_FORMAT = (
    'Answer with only a JSON object: {"correct": true or false, "reply": "..."}. No other text.'
)

_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean"},
        "reply": {"type": "string"},
    },
    "required": ["correct", "reply"],
    "additionalProperties": False,
}


class DraftVerifier(ABC):
    """Checks a reply against the conversation it answers."""

    @abstractmethod
    async def verify(self, messages: list[LLMContextMessage], draft: str) -> str | None:
        """Check a reply.

        Args:
            messages: The conversation the reply answers, up to and including
                the user's turn. The reply itself is not in it.
            draft: The reply.

        Returns:
            None if the reply stands, or the reply that should replace it.
        """


class LLMDraftVerifier(DraftVerifier):
    """A verifier that asks an LLM service, out of band, whether the reply is right.

    The service runs a one-shot inference (``run_inference``) on the
    conversation, the reply, and ``prompt``, and answers in JSON. A service
    that supports ``response_schema`` gets the verdict schema; any other is
    asked for the JSON in the prompt, and a verdict wrapped in a code fence or
    in prose is still read. It is not in any pipeline: it only needs to be
    constructed.

    Example::

        verifier = LLMDraftVerifier(llm=AnthropicLLMService(api_key=..., model=...))
    """

    def __init__(self, *, llm: LLMService[Any], prompt: str = DRAFT_VERIFICATION_PROMPT):
        """Initialize the verifier.

        Args:
            llm: The service that checks the reply. Usually a larger model
                than the one that wrote it.
            prompt: What the model is asked, as a developer message after the
                reply. :data:`DRAFT_VERIFICATION_PROMPT` unless given.
        """
        self._llm = llm
        self._prompt = prompt

    async def verify(self, messages: list[LLMContextMessage], draft: str) -> str | None:
        """Ask the model whether the reply stands.

        Args:
            messages: The conversation the reply answers.
            draft: The reply.

        Returns:
            None if the model agrees with the reply, or its corrected reply.
        """
        context = LLMContext(
            messages=[
                *messages,
                {"role": "assistant", "content": draft},
                {"role": "developer", "content": f"{self._prompt}\n\n{_VERDICT_FORMAT}"},
            ]
        )
        schema = _VERDICT_SCHEMA if self._llm.supports_response_schema else None
        raw = await self._llm.run_inference(context, response_schema=schema)
        try:
            verdict = _parse_verdict(raw or "")
            if verdict["correct"]:
                return None
            reply = str(verdict["reply"]).strip()
        except (ValueError, KeyError, TypeError) as e:
            # A verifier that can't be read must not hold up or change speech.
            logger.warning(f"{self}: unreadable verdict, letting the reply stand: {e}")
            return None
        return reply if reply and reply != draft.strip() else None


def _parse_verdict(raw: str) -> Any:
    """Read the first JSON object in ``raw``, ignoring a code fence or prose around it."""
    start = raw.find("{")
    if start < 0:
        raise ValueError("no JSON object in the verdict")
    verdict, _ = json.JSONDecoder().raw_decode(raw, start)
    return verdict


@dataclass
class _Draft:
    """One reply the extension is checking.

    Parameters:
        epoch: The conversation epoch the reply was written in.
        messages: The conversation it answers.
        text: The reply so far.
        held: The frames held back while it waits for its verdict.
        holding: Whether it is held back, or already on its way to the user.
        verifiable: False once it turns out to call tools, whose side effects
            a correction could not undo.
    """

    epoch: int
    messages: list[LLMContextMessage]
    text: str = ""
    held: list[Frame] = field(default_factory=list)
    holding: bool = False
    verifiable: bool = True


class DraftVerificationLLMExtension(BaseLLMExtension):
    """A second model checks each reply, and its correction reaches the user only while it still applies.

    Each text reply the service produces is handed to ``verifier`` once the
    reply ends. What a correction can change depends on where the reply is
    by then:

    - Still held (``hold_secs`` > 0 and the verdict came within it): the
      corrected reply goes out in place of the original, which never reaches
      the TTS or the context.
    - Already on its way to the user: with ``late_corrections`` on, the
      correction goes out as a reply of its own (``late_correction_template``),
      queued behind the original, so the bot never talks over itself. With it
      off, the correction is logged and dropped.

    Either way, a verdict is only applied in the epoch it was asked in. The
    epoch moves on whenever the conversation does: the user starts speaking,
    an interruption, a withdrawn speculative turn, or a new inference. A
    verdict from an earlier epoch is dropped, never spoken. A reply still held
    when the epoch moves on goes out unchecked ahead of a new inference's
    reply, and is dropped if the user started speaking or interrupted. A
    correction that is queued but not yet spoken when the user starts speaking
    is cleared by the interruption like any other bot speech.

    Holding a reply costs what it saves in quality: with ``hold_secs`` > 0
    the user hears nothing until the whole reply is generated and the verifier
    answers or the hold runs out. It suits a fast model with short replies.
    With ``hold_secs`` = 0, the default, the reply is spoken as it streams and
    only a late correction is possible.

    A reply that calls tools is not checked: the calls run inside the service,
    and a correction could not undo them.

    The extension belongs on a text LLM, and works with a speculative turn
    (:class:`~pipecat.turns.user_stop.EagerUserTurnStopStrategy`): it sits
    inside the service's speculation gate, so checking a speculative reply
    starts early, and the reply still waits for the turn to be confirmed.

    Example::

        llm = OpenAILLMService(
            settings=OpenAILLMService.Settings(
                model="gpt-4.1-mini",
                extensions=[
                    DraftVerificationLLMExtension(
                        verifier=LLMDraftVerifier(llm=AnthropicLLMService(...)),
                        hold_secs=0.4,
                    )
                ]
            ),
        )
    """

    # Farthest from the service of the built-in extensions, so the reply it
    # checks is what the user would hear: turn markers and silent replies are
    # already taken out by the extensions "closer" to the service.
    order = 2

    def __init__(
        self,
        *,
        verifier: DraftVerifier,
        hold_secs: float = 0.0,
        late_corrections: bool = True,
        late_correction_template: str = LATE_CORRECTION_TEMPLATE,
        **kwargs,
    ):
        """Initialize the extension.

        Args:
            verifier: What checks each reply.
            hold_secs: How long, after a reply ends, it is held for its
                verdict before it goes out unchecked. 0 sends every reply as
                it streams.
            late_corrections: Whether a correction of a reply already sent is
                spoken, as a reply of its own. Off drops it.
            late_correction_template: The late correction, a template with
                ``{text}``. :data:`LATE_CORRECTION_TEMPLATE` unless given.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._verifier = verifier
        self._hold_secs = hold_secs
        self._late_corrections = late_corrections
        self._late_correction_template = late_correction_template
        self._epoch = 0
        self._messages: list[LLMContextMessage] = []
        # The reply streaming now, and the one waiting for its verdict.
        self._draft: _Draft | None = None
        self._verifying: _Draft | None = None
        self._verification_task: asyncio.Task | None = None

    @property
    def epoch(self) -> int:
        """The conversation epoch: moves on each time the conversation does."""
        return self._epoch

    async def cleanup(self):
        """Stop the verification in flight."""
        await super().cleanup()
        await self._cancel_verification()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Track the conversation the next reply answers, and when it moves on.

        Args:
            frame: The frame the service processes.
            direction: The direction of frame processing.
        """
        if isinstance(frame, LLMContextFrame):
            # A new inference: whatever was being checked answers an earlier
            # point in the conversation. A held reply goes out unchecked
            # rather than being lost, ahead of the new reply.
            await self._advance(release_held=True)
            self._messages = list(frame.context.get_messages())
        elif isinstance(
            frame, (UserStartedSpeakingFrame, InterruptionFrame, UserTurnResponseCancelFrame)
        ):
            # The held reply is void: the user is speaking again, or it
            # answers a turn they had not finished. Sending it now would talk
            # over them with an answer to what they said before.
            await self._advance(release_held=False)
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._advance(release_held=isinstance(frame, EndFrame))

    async def filter_push_frame(self, frame: Frame, direction: FrameDirection) -> list[Frame]:
        """Hold a reply for its verdict, and note what it says.

        Args:
            frame: The frame the service is pushing.
            direction: The direction of frame pushing.

        Returns:
            The frames to push on: nothing while a reply is held.
        """
        if direction != FrameDirection.DOWNSTREAM or isinstance(frame, SystemFrame):
            if isinstance(frame, UserTurnResponseCancelFrame):
                # The service withdrew its own speculative reply (it wanted a
                # tool call).
                await self._advance(release_held=False)
            elif isinstance(frame, (FunctionCallsStartedFrame, FunctionCallInProgressFrame)):
                await self._skip_draft()
            return [frame]

        if isinstance(frame, LLMFullResponseStartFrame):
            self._draft = _Draft(
                epoch=self._epoch, messages=self._messages, holding=self._hold_secs > 0
            )

        draft = self._draft
        if draft is None:
            return [frame]

        if isinstance(frame, (FunctionCallsStartedFrame, FunctionCallInProgressFrame)):
            await self._skip_draft()
            return [frame]

        if isinstance(frame, LLMTextFrame):
            draft.text += frame.text

        if isinstance(frame, LLMFullResponseEndFrame):
            self._draft = None
            if draft.verifiable and draft.text.strip():
                self._start_verification(draft)
            elif draft.holding:
                draft.held.append(frame)
                return self._take_held(draft)

        if draft.holding:
            draft.held.append(frame)
            return []
        return [frame]

    def _start_verification(self, draft: _Draft):
        # One reply is checked at a time: a new one only ends after the
        # epoch moved on, which cancelled the previous check.
        self._verifying = draft
        self._verification_task = self.create_task(self._verify(draft), "verify")

    async def _verify(self, draft: _Draft):
        verification = self.create_task(
            self._verifier.verify(draft.messages, draft.text), "verifier"
        )
        try:
            if draft.holding:
                done, _ = await asyncio.wait({verification}, timeout=self._hold_secs)
                if not done:
                    logger.debug(f"{self}: no verdict within {self._hold_secs}s, sending the reply")
                    await self._push_held(draft)
            correction = await verification
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"{self}: verifier failed, letting the reply stand: {e}")
            correction = None
        finally:
            if not verification.done():
                await self.cancel_task(verification)

        self._verifying = None
        if draft.epoch != self._epoch:
            logger.debug(f"{self}: dropping a verdict from an earlier epoch")
            return

        if correction is None:
            logger.debug(f"{self}: reply verified")
            await self._push_held(draft)
        elif draft.holding:
            logger.debug(f"{self}: replacing the reply before it is spoken")
            await self._push_frames(self._replace_text(draft, correction))
        elif self._late_corrections:
            logger.debug(f"{self}: correcting a reply already sent")
            await self._push_frames(
                [
                    LLMFullResponseStartFrame(),
                    LLMTextFrame(self._late_correction_template.format(text=correction)),
                    LLMFullResponseEndFrame(),
                ]
            )
        else:
            logger.debug(f"{self}: reply already sent, dropping its correction")

    async def _advance(self, *, release_held: bool):
        """Move to a new epoch: no verdict asked for before this applies any more."""
        self._epoch += 1
        await self._cancel_verification()
        # The reply waiting for its verdict came first, so it goes out first.
        drafts = [d for d in (self._verifying, self._draft) if d is not None and d.holding]
        self._verifying = None
        self._draft = None
        for draft in drafts:
            frames = self._take_held(draft)
            if not release_held:
                # Like an interruption, which keeps the frames that must
                # always be delivered.
                frames = [f for f in frames if not f.interruptible]
            await self._push_frames(frames)

    async def _skip_draft(self):
        """Send on a reply that can't be checked."""
        draft = self._draft
        if draft is None:
            return
        draft.verifiable = False
        if draft.holding:
            await self._push_frames(self._take_held(draft))

    async def _push_held(self, draft: _Draft):
        await self._push_frames(self._take_held(draft))

    def _take_held(self, draft: _Draft) -> list[Frame]:
        frames, draft.held = draft.held, []
        draft.holding = False
        return frames

    def _replace_text(self, draft: _Draft, correction: str) -> list[Frame]:
        """The held reply with its text replaced, everything else kept in place."""
        frames: list[Frame] = []
        replaced = False
        for frame in self._take_held(draft):
            if isinstance(frame, LLMTextFrame):
                if not replaced:
                    frames.append(LLMTextFrame(correction))
                    replaced = True
                continue
            frames.append(frame)
        return frames

    async def _push_frames(self, frames: list[Frame]):
        for frame in frames:
            await self.push_frame(frame)

    async def _cancel_verification(self):
        task, self._verification_task = self._verification_task, None
        if task:
            await self.cancel_task(task)
