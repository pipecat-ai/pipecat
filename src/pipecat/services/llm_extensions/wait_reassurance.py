#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The LLM extension that has the model say a brief word while the user waits on background work.

The instruction and the prompt below are what the model reads: the instruction
once, in its system instruction, and the prompt each time the conversation has
gone quiet with background work in progress. A small change in wording can
change what a model says, or whether it stays silent, in ways no unit test
catches. Before and after changing either, run the checks in
``scripts/llm-with-backend/tuning/README.md`` and record their results in
``RESULTS.md`` beside it.
"""

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMFullResponseStartFrame,
    LLMMarkerResponseFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    StartFrame,
    UserWaitingFrame,
    UserWaitingTimeoutUpdateFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_extensions.base_llm_extension import BaseLLMExtension

#: What the model answers with to say nothing.
WAIT_REASSURANCE_SILENT_MARK = "∅"

#: Standing guidance on the waiting message, composed into the LLM's system
#: instruction: what the message is, how to answer it with silence, and what
#: to say otherwise. ``{mark}`` is the silent mark.
WAIT_REASSURANCE_INSTRUCTION = (
    "While background work you started on the user's behalf is in flight and the conversation has "
    'gone quiet, a developer message beginning "The user is waiting on background work you started '
    'on their behalf" will say so. That message asks one thing: to reassure the user, if needed, '
    "that the work is still underway. Answer that message with exactly {mark} to "
    "respond with silence (after any marker your instructions require). Otherwise, say one short "
    "sentence, in your own voice and with wording that varies from previous reassurances about the "
    "same work, that makes clear the work has been going on since the user asked and is "
    'continuing: a "still", not a start. Keep it general, naming at most the kind of step under '
    "way, unless the user asked for detail: then name the step, drawing on the notes the work has "
    "produced. Never report a result or a finding, never say the work is done, and ask no "
    "questions and make no offers or promises, not even to keep the user posted. {mark} is only "
    "ever the answer to that message. Whatever the user says, answer them, and when the work "
    "reports back, tell the user, however often you have chosen silence before."
)

#: The developer message appended to the conversation each time the user has
#: waited quietly on background work: the facts of the moment, the ask, and
#: what to weigh in answering it. ``{quiet_secs}`` is how long nothing has
#: been said. Its opening is what the instruction names it by.
WAIT_REASSURANCE_PROMPT = (
    "The user is waiting on background work you started on their behalf, and nothing has been said "
    "for about {quiet_secs} seconds. Reassure them, if needed, that the work is still underway. "
    "When deciding what to say (or in some cases, whether to say anything at all), consider the "
    "user's preferences (for example, they may have asked for only the final result of the work "
    "and no progress updates along the way) as well as what would now sound natural and "
    "reassuring in the conversation."
)


class WaitReassuranceLLMExtension(BaseLLMExtension):
    """A brief remark from the LLM, in its own words, while the user waits on background work.

    Background work is what a ``BackgroundWorkStartedFrame`` announced, e.g.
    an asynchronous function call or a task delegated to an ``LLMWithBackend``
    backend. Once nothing has been said for ``interval_secs`` with such work
    in progress, the user aggregator broadcasts a ``UserWaitingFrame``, on
    which the extension appends ``prompt`` to the conversation as a developer
    message and runs the LLM. The model decides what to say, with the user's
    wishes and what it knows of the work in view, or that nothing should be
    said, by answering with ``silent_mark``.

    The extension's convention with the model has two halves. ``instruction``
    goes into the service's system instruction and explains the prompt and
    the mark once. A reply opening with the mark is dropped unspoken,
    reaching neither the TTS nor the context, and the decision is broadcast
    as an ``LLMMarkerResponseFrame`` of kind ``"silent"``, on which the user
    aggregator starts the interval over. A spoken remark starts it over
    through the bot's speech, so a long wait gets a fresh decision every
    interval.

    At start, the extension sets the user aggregator's waiting timeout to
    ``interval_secs`` with a ``UserWaitingTimeoutUpdateFrame``, so the
    interval is configured here alone. The extension belongs on a text LLM:
    a speech-to-speech service's audio carries no mark to read. An app that
    wants to answer the wait in its own way (a canned phrase, music) uses the
    user aggregator's ``user_waiting_timeout`` and ``on_user_waiting``
    instead.

    Example::

        llm = OpenAILLMService(..., extensions=[WaitReassuranceLLMExtension()])
    """

    def __init__(
        self,
        *,
        interval_secs: float = 10.0,
        instruction: str = WAIT_REASSURANCE_INSTRUCTION,
        prompt: str = WAIT_REASSURANCE_PROMPT,
        silent_mark: str = WAIT_REASSURANCE_SILENT_MARK,
        **kwargs,
    ):
        """Initialize the extension.

        Args:
            interval_secs: Seconds the conversation stays quiet, with background
                work in progress, before the LLM is asked whether to say something.
            instruction: The standing guidance added to the LLM's system
                instruction, a template with ``{mark}`` (so any other brace in
                it is doubled). :data:`WAIT_REASSURANCE_INSTRUCTION` unless
                given.
            prompt: The developer message appended each time, a template with
                ``{quiet_secs}`` and, if it wants it, ``{mark}``.
                :data:`WAIT_REASSURANCE_PROMPT` unless given.
            silent_mark: What the model answers with to say nothing.
                :data:`WAIT_REASSURANCE_SILENT_MARK` unless given.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._interval_secs = interval_secs
        self._instruction = instruction
        self._prompt = prompt
        self._silent_mark = silent_mark
        # The reply being read for the silent mark: its text so far, held until
        # the first character decides it, and whether the rest is dropped.
        self._reply_opening = ""
        self._reply_decided = False
        self._reply_silent = False

    @property
    def system_instruction(self) -> str:
        """The standing guidance, with the silent mark in place."""
        return self._instruction.format(mark=self._silent_mark)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Configure the wait at start, and ask the model once the user has waited.

        Args:
            frame: The frame the service processes.
            direction: The direction of frame processing.
        """
        if isinstance(frame, StartFrame):
            await self.push_frame(
                UserWaitingTimeoutUpdateFrame(timeout=self._interval_secs), FrameDirection.UPSTREAM
            )
        elif isinstance(frame, UserWaitingFrame):
            quiet_secs = round(frame.quiet_secs)
            logger.debug(
                f"{self}: asking the model whether to reassure the user ({quiet_secs}s quiet)"
            )
            await self.push_frame(
                LLMMessagesAppendFrame(
                    messages=[
                        {
                            "role": "developer",
                            "content": self._prompt.format(
                                quiet_secs=quiet_secs, mark=self._silent_mark
                            ),
                        }
                    ],
                    run_llm=True,
                )
            )
        elif isinstance(frame, InterruptionFrame):
            self._reset_reading()

    async def filter_push(self, frame: Frame, direction: FrameDirection) -> list[Frame]:
        """Drop a reply that opens with the silent mark, and report it.

        Args:
            frame: The frame the service is pushing.
            direction: The direction of frame pushing.

        Returns:
            The frames to push on: the reply's text once its opening shows it
            is not the mark, nothing while the opening is held or the reply
            is being dropped.
        """
        if isinstance(frame, LLMFullResponseStartFrame):
            self._reset_reading()
        elif isinstance(frame, LLMTextFrame) and direction == FrameDirection.DOWNSTREAM:
            return await self._read_opening(frame)
        return [frame]

    async def _read_opening(self, frame: LLMTextFrame) -> list[Frame]:
        """Hold a reply's opening until its first character shows whether the reply is the mark.

        A silent reply is reported as soon as the mark is read, and the rest
        of it is dropped unread. The turn-completion extension runs first, so
        a reply of a turn marker followed by the mark is read as silent too.
        """
        if self._reply_silent:
            return []
        if self._reply_decided:
            return [frame]
        held = self._reply_opening
        self._reply_opening += frame.text
        text = self._reply_opening.lstrip()
        if not text:
            return []
        if text.startswith(self._silent_mark):
            logger.debug(f"{self}: the model chose to say nothing")
            self._reply_decided = True
            self._reply_silent = True
            await self.broadcast_frame(
                LLMMarkerResponseFrame,
                raw=self._reply_opening,
                marker=self._silent_mark,
                kind="silent",
                markers=[self._silent_mark],
            )
            return []
        if self._silent_mark.startswith(text):
            # The mark so far, with more of it possibly to come.
            return []
        self._reply_decided = True
        return [frame if not held else LLMTextFrame(self._reply_opening)]

    def _reset_reading(self):
        self._reply_opening = ""
        self._reply_decided = False
        self._reply_silent = False
