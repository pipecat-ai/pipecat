#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Observer that feeds the bot's spoken replies to AssemblyAI for context carryover.

AssemblyAI's Universal-3 Pro streaming model (``u3-rt-pro``) supports context
carryover: telling the model what the agent just said improves transcription of
the user's reply. This observer captures the bot's spoken text and forwards it to
:class:`~pipecat.services.assemblyai.stt.AssemblyAISTTService` after each reply,
so the feature works with a single line of wiring.
"""

from loguru import logger

from pipecat.frames.frames import BotStoppedSpeakingFrame, TTSTextFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.services.assemblyai.stt import AssemblyAISTTService
from pipecat.utils.string import TextPartForConcatenation, concatenate_aggregated_text


class AssemblyAIContextObserver(BaseObserver):
    """Feeds each bot reply to an AssemblyAI STT service as carryover context.

    Accumulates the bot's spoken text from ``TTSTextFrame`` and, when the bot
    stops speaking, sends the concatenated reply to the STT service via
    :meth:`~pipecat.services.assemblyai.stt.AssemblyAISTTService.update_agent_context`.

    Because it keys off actually-spoken text, an interrupted reply carries only
    the portion the user heard. The text is concatenated the same way as the
    assistant context aggregator, so it matches what is stored in the LLM context.

    Event handlers available:

    - on_agent_context: Called with the agent reply text each time it is sent to
      AssemblyAI. Useful as a signal that context carryover is active.

    Example::

        observer = AssemblyAIContextObserver(stt)

        @observer.event_handler("on_agent_context")
        async def on_agent_context(observer, text):
            logger.info(f"Sent agent_context: {text!r}")

        worker = PipelineWorker(pipeline, observers=[observer])
    """

    def __init__(self, stt: AssemblyAISTTService):
        """Initialize the observer.

        Args:
            stt: The AssemblyAI STT service to feed agent context to.
        """
        super().__init__()
        self._stt = stt
        self._parts: list[TextPartForConcatenation] = []
        # Observers see every frame on every hop; track ids to count each once.
        self._seen_ids: set[int] = set()
        self._register_event_handler("on_agent_context")

    async def on_push_frame(self, data: FramePushed):
        """Accumulate spoken text and flush it to the STT when the bot stops.

        Args:
            data: The frame-pushed event.
        """
        frame = data.frame

        if isinstance(frame, TTSTextFrame):
            if frame.id in self._seen_ids or not frame.text:
                return
            self._seen_ids.add(frame.id)
            self._parts.append(
                TextPartForConcatenation(
                    frame.text,
                    includes_inter_part_spaces=frame.includes_inter_frame_spaces,
                )
            )
        elif isinstance(frame, BotStoppedSpeakingFrame):
            # No-op when there is nothing buffered. BotStoppedSpeakingFrame is
            # emitted both upstream and downstream, so clearing the buffer makes
            # the sibling a no-op.
            if not self._parts:
                return
            text = concatenate_aggregated_text(self._parts).strip()
            self._parts = []
            self._seen_ids.clear()
            if text:
                logger.trace(f"{self}: sending agent_context to AssemblyAI: {text!r}")
                await self._stt.update_agent_context(text)
                await self._call_event_handler("on_agent_context", text)
