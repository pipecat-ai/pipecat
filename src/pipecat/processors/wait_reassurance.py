#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A brief remark from the LLM while the user waits on background work.

The prompt below is what the model reads each time the conversation has gone
quiet with background work in progress. A small change in wording can change
what a model says, or whether it stays silent, in ways no unit test catches.
Before and after changing it, run the checks in
``scripts/llm-with-backend/tuning/README.md`` and record their results in
``RESULTS.md`` beside it.
"""

import time

from loguru import logger

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    LLMMarkerResponseFrame,
    LLMMessagesAppendFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessor,
    FrameProcessorSetup,
)
from pipecat.services.llm_service import LLM_SILENT_MARK
from pipecat.turns.user_idle_controller import UserIdleController

#: The developer message appended to the conversation each time the user has
#: waited quietly on background work. ``{quiet_secs}`` is how long nothing has
#: been said; ``{mark}`` is the silent mark.
WAIT_REASSURANCE_PROMPT = (
    "The user is waiting on background work you started on their behalf, and nothing has "
    "been said for about {quiet_secs} seconds. This message asks one thing: whether to say a "
    "few words, now, unprompted, to reassure the user that the work is still underway. When "
    "deciding, consider the user's preferences (for example, they may have asked for only the "
    "final result of the work and no progress updates along the way) as well as whether a "
    "remark now would sound natural. Answer this message with exactly {mark} to respond with "
    "silence (after any marker your instructions require). Otherwise, say one or two short "
    "sentences, in your own voice and with wording that varies from previous reassurances "
    "about the same work. Keep it general, naming at most the kind of step under way, unless "
    "the user asked for detail. Never report a result or a finding, never say the work is "
    "done, and ask no questions and make no offers or promises, not even to keep the user "
    "posted. {mark} is only ever the answer to this message: whatever the user says next, "
    "answer them."
)


class WaitReassurance(FrameProcessor):
    """A brief remark from the LLM, in its own words, while the user waits on background work.

    Placed after the LLM: any text :class:`~pipecat.services.llm_service.LLMService`,
    or an :class:`~pipecat.pipeline.llm_with_backend.LLMWithBackend`.
    Background work is what a ``BackgroundWorkStartedFrame`` announced, e.g.
    an asynchronous function call or a task delegated to an ``LLMWithBackend``
    backend. Once nothing has been said for ``interval_secs`` with such work
    in progress, the processor appends ``prompt`` to the conversation as a
    developer message and runs the LLM. The model decides what to say, with
    the user's wishes and what it knows of the work in view, or that nothing
    should be said, by answering with ``silent_mark``. The LLM service honors
    that mark (its ``honor_silent_mark``, which names the same one): the reply
    is dropped unspoken, reaching neither the TTS nor the context, and the
    decision is reported, on which the processor starts the interval over. A
    spoken remark starts it over through the bot's speech, so a long wait
    gets a fresh decision every interval.

    The processor belongs after a text LLM: a speech-to-speech service's
    audio carries no mark to read. An app that wants to answer the wait in
    its own way (a canned phrase, music) uses the user aggregator's
    ``user_waiting_timeout`` and ``on_user_waiting`` instead of this
    processor.

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
        silent_mark: str = LLM_SILENT_MARK,
        **kwargs,
    ):
        """Initialize the processor.

        Args:
            interval_secs: Seconds the conversation stays quiet, with background
                work in progress, before the LLM is asked whether to say something.
            prompt: The developer message appended each time, a template with
                ``{quiet_secs}`` and ``{mark}`` (so any other brace in it is
                doubled). :data:`WAIT_REASSURANCE_PROMPT` unless given.
            silent_mark: What the prompt tells the model to answer with to say
                nothing. The LLM service's ``honor_silent_mark`` must name the
                same mark.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._interval_secs = interval_secs
        self._prompt = prompt
        self._silent_mark = silent_mark

        self._controller = UserIdleController(user_waiting_timeout=interval_secs)
        self._controller.add_event_handler("on_user_waiting", self._on_user_waiting)

        self._last_spoke_at: float | None = None

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
        """Process a frame: keep the timer, and start it over on a silent decision.

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
        elif isinstance(frame, LLMMarkerResponseFrame) and frame.kind == "silent":
            await self.push_frame(frame, direction)
            # Nothing was said, so no bot speech will start the interval over:
            # the next one gets a fresh decision.
            await self._controller.wait_for_work()
        else:
            if isinstance(frame, (BotStoppedSpeakingFrame, UserStoppedSpeakingFrame)):
                self._last_spoke_at = time.monotonic()
            await self.push_frame(frame, direction)

        await self._controller.process_frame(frame)

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
