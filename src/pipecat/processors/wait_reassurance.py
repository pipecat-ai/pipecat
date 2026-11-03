#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A brief remark from the LLM while the user waits on work in flight.

The prompt below is what the model reads each time the conversation has gone
quiet with work in flight. A small change in wording can change what a model
says, or whether it stays silent, in ways no unit test catches. Before and
after changing it, run the checks in ``scripts/llm-with-backend/tuning/README.md``
and record their results in ``RESULTS.md`` beside it.
"""

import time

from loguru import logger

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessor,
    FrameProcessorSetup,
)
from pipecat.turns.user_idle_controller import UserIdleController

#: What the model writes, and nothing else, to say nothing this time.
WAIT_REASSURANCE_SILENT_MARK = "∅"

#: The developer message appended to the conversation each time the user has
#: waited quietly on work in flight. ``{quiet_secs}`` is how long nothing has
#: been said; ``{mark}`` is the silent mark.
WAIT_REASSURANCE_PROMPT = (
    "The user is waiting on work you started on their behalf, and nothing has been said for "
    "about {quiet_secs} seconds. This message asks one thing: whether to say a word now, "
    "unprompted. First check what the user asked for: if they said not to keep them posted, "
    "or to just tell them when it's done, or a remark now would not sound natural, answer "
    "this message with exactly {mark} and nothing else (after any marker your instructions "
    "require). Otherwise say one or two short sentences, in your own voice and worded "
    "differently from anything you said before, so the user knows you are still on it: that "
    "you are still working, and at most what kind of step is under way now, with no file "
    "names, commands or code. What the notes say was found or changed is not yours to pass "
    "on; it will be reported when the work is done. If the user asked for detail, name the "
    "step and draw on the notes for it, files included. Never report a result or say the "
    "work is done, and ask no questions, make no offers and no promises. {mark} is only ever "
    "the answer to this message: whatever the user says next, answer them."
)


class WaitReassurance(FrameProcessor):
    """A brief remark from the LLM, in its own words, while the user waits on work in flight.

    Placed after the LLM: any text :class:`~pipecat.services.llm_service.LLMService`,
    or an :class:`~pipecat.pipeline.llm_with_backend.LLMWithBackend`. Work in
    flight is what a ``BackgroundWorkStartedFrame`` announced: an asynchronous
    function call, a backend's delegated task. Once nothing has been said for
    ``interval_secs`` with such work in flight, the processor appends
    ``prompt`` to the conversation as a developer message and runs the LLM.
    The model decides what to say, with the user's wishes and what it knows of
    the work in view, or that nothing should be said: a reply that opens with
    ``silent_mark`` is dropped unspoken. A spoken remark starts the interval
    over through the bot's speech; a silent decision starts it over directly,
    so a long wait gets a fresh decision every interval.

    Every response the LLM streams is read for the mark, with its opening
    held back until the first character shows whether the reply is one, so
    the processor belongs after a text LLM: a speech-to-speech service's
    audio cannot be read. A dropped reply reaches nothing after the processor,
    the TTS, the output transport, the assistant aggregator, but an observer
    of the LLM's own output, such as the RTVI observer's ``bot-llm-text`` and
    ``llm-raw-text`` messages, still sees the mark the model wrote. An app
    that wants to answer the wait in its own way (a canned phrase, music) uses
    the user aggregator's ``user_waiting_timeout`` and ``on_user_waiting``
    instead of this processor.

    Example::

        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                aggregators.user(),
                llm,
                WaitReassurance(),
                tts,
                transport.output(),
                aggregators.assistant(),
            ]
        )
    """

    def __init__(
        self,
        *,
        interval_secs: float = 10.0,
        prompt: str = WAIT_REASSURANCE_PROMPT,
        silent_mark: str = WAIT_REASSURANCE_SILENT_MARK,
        **kwargs,
    ):
        """Initialize the processor.

        Args:
            interval_secs: Seconds the conversation stays quiet, with work in
                flight, before the LLM is asked whether to say something.
            prompt: The developer message appended each time, a template with
                ``{quiet_secs}`` and ``{mark}`` (so any other brace in it is
                doubled). :data:`WAIT_REASSURANCE_PROMPT` unless given.
            silent_mark: What a reply opens with to say nothing.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._interval_secs = interval_secs
        self._prompt = prompt
        self._silent_mark = silent_mark

        self._controller = UserIdleController(user_waiting_timeout=interval_secs)
        self._controller.add_event_handler("on_user_waiting", self._on_user_waiting)

        self._last_spoke_at: float | None = None
        # The response being read: its opening, held until the first character
        # decides it, and whether the rest is being dropped.
        self._held = ""
        self._decided = False
        self._silenced = False

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the processor.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._controller.setup(setup)

    async def cleanup(self):
        """Clean up the processor."""
        await super().cleanup()
        await self._controller.cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame: read the LLM's responses for the mark, and keep the timer.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, EndFrame):
            await self.push_frame(frame, direction)
            await self._controller.stop()
        elif isinstance(frame, CancelFrame):
            await self._controller.stop()
            await self.push_frame(frame, direction)
        elif isinstance(frame, InterruptionFrame):
            self._reset_reading()
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMFullResponseStartFrame):
            self._reset_reading()
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMTextFrame) and direction == FrameDirection.DOWNSTREAM:
            await self._read(frame)
        elif isinstance(frame, LLMFullResponseEndFrame):
            silenced = self._silenced
            self._reset_reading()
            await self.push_frame(frame, direction)
            if silenced:
                # Nothing was said, so no bot speech will start the interval
                # over: the next one gets a fresh decision.
                await self._controller.wait_for_work()
        else:
            if isinstance(frame, (BotStoppedSpeakingFrame, UserStoppedSpeakingFrame)):
                self._last_spoke_at = time.monotonic()
            await self.push_frame(frame, direction)

        await self._controller.process_frame(frame)

    async def _read(self, frame: LLMTextFrame):
        """Pass a response's text on, except a reply that opens with the mark."""
        if self._silenced:
            return
        if self._decided:
            await self.push_frame(frame)
            return
        opening = self._held
        self._held += frame.text
        text = self._held.lstrip()
        if not text:
            return
        self._decided = True
        if text.startswith(self._silent_mark):
            logger.debug(f"{self}: the model chose to say nothing")
            self._silenced = True
            return
        # The frame itself goes on when it opened the reply, keeping whatever
        # else it carries; held whitespace from earlier frames goes with it.
        await self.push_frame(frame if not opening else LLMTextFrame(self._held))

    def _reset_reading(self):
        self._held = ""
        self._decided = False
        self._silenced = False

    async def _on_user_waiting(self, controller: UserIdleController):
        quiet_secs = (
            round(time.monotonic() - self._last_spoke_at)
            if self._last_spoke_at is not None
            else round(self._interval_secs)
        )
        logger.debug(f"{self}: asking the model whether to reassure the user ({quiet_secs}s quiet)")
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
