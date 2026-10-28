#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The processor that tags an LLM's plain replies with Pipecat Tone's tagger."""

import asyncio
from dataclasses import replace

from loguru import logger

from pipecat.expressive.formatter import EXPRESSION_TAGS
from pipecat.expressive.policy import ExpressionPolicy
from pipecat.frames.frames import (
    ExpressionMetadataFrame,
    Frame,
    InterruptionFrame,
    LLMContextAssistantTurnFrame,
    LLMContextUserTurnFrame,
    LLMFullResponseStartFrame,
    StartFrame,
    TextType,
    TTSMetadataFrame,
    UserTurnResponseCancelFrame,
)
from pipecat.processors.aggregators.llm_text_processor import LLMTextProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.processors.frameworks.rtvi.frames import RTVISendTextFrame
from pipecat.processors.tone._models import load_model
from pipecat.processors.tone.tagger._input import Message, TurnContext, fold
from pipecat.processors.tone.tagger._labels import LabelSpace, ReplyRenderer, SentenceLabels
from pipecat.processors.tone.tagger.model import PipecatToneTaggerModel
from pipecat.utils.text.skip_tags_aggregator import SkipTagsAggregator


class PipecatToneTagger(LLMTextProcessor):
    """Tags an LLM's plain replies with Pipecat Tone, a sentence at a time.

    Goes between the LLM and the TTS service. Each sentence is labeled with
    the call so far, how the user sounded and the pipeline's expression
    policy, and sent on with the tags the TTS service performs. The labeling
    runs off the event loop, and the sentence waits for it, about 15
    milliseconds.

    The model loads while the pipeline is set up, once a process, and every
    tagger shares it. The first load downloads it from Hugging Face, which holds
    the pipeline's start, so download it ahead, such as when building the
    image. At start, the tagger broadcasts an
    :class:`~pipecat.frames.frames.ExpressionMetadataFrame`, so the LLM is
    taught no tags and writes plain words, and the context keeps them plain. A
    model that can't be loaded is reported as an error that leaves the tagger
    unusable: the LLM writes the tags itself and replies pass through as
    written.

    The policy's ``ExpressionGuidance`` presets steer the tagger, its
    ``delivery`` and ``sounds`` narrow the names it writes, and a breath is
    written only with ``ExpressionGuidance.BREATHE``. The developer's own
    lines are for the LLM.

    Part of expressive conversation (alpha): turn it on with
    ``PipelineParams(enable_expressive_conversation=True)``. Add a
    :class:`~pipecat.processors.tone.reader.PipecatToneReader` for the tagger
    to hear how the user sounded.

    Example::

        pipeline = Pipeline([..., user_aggregator, llm, PipecatToneTagger(), tts, ...])
    """

    def __init__(
        self,
        *,
        threads: int = 1,
        **kwargs,
    ):
        """Initialize the processor.

        Args:
            threads: The CPU threads for one sentence.
            **kwargs: Additional arguments passed to the parent.
        """
        super().__init__(text_aggregator=SkipTagsAggregator(EXPRESSION_TAGS), **kwargs)
        self.add_text_transformer(self._tag, TextType.SENTENCE)
        self._model: PipecatToneTaggerModel | None = None
        self._threads = threads
        self._policy = ExpressionPolicy()
        self._space: LabelSpace | None = None
        self._messages: list[Message] = []
        self._turn: TurnContext | None = None
        self._renderer: ReplyRenderer | None = None
        self._sentences: list[str] = []
        # The labels of the reply being tagged, which the next reply's model
        # reads; None before the first reply.
        self._labels: list[SentenceLabels] | None = None

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the processor with the pipeline's expression policy, and load its model.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        self._policy = setup.expression_policy or ExpressionPolicy()
        if not setup.enable_expressive_conversation:
            logger.warning(
                f"{self}: expressive conversation is off, so replies pass through untagged; "
                "turn it on with PipelineParams(enable_expressive_conversation=True)"
            )
            return
        self._model = await load_model(
            self,
            ("tagger", self._threads),
            lambda: PipecatToneTaggerModel.download(threads=self._threads),
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Tag the LLM's sentences, and keep the call and the TTS's names to tag them by.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        if not self._model:
            # Replies pass through as the LLM wrote them.
            await super().process_frame(frame, direction)
            return

        if isinstance(frame, TTSMetadataFrame):
            self._on_tts_metadata(frame)
        elif isinstance(frame, LLMContextUserTurnFrame):
            self._on_user_turn(frame)
        elif isinstance(frame, RTVISendTextFrame):
            # Text the user typed is appended to the context with no user turn
            # frame, and with no reading.
            self._messages.append(Message("user", frame.text))
        elif isinstance(frame, LLMContextAssistantTurnFrame):
            self._messages.append(Message("assistant", frame.text))
        elif isinstance(frame, UserTurnResponseCancelFrame):
            # The turn an early reply was to answer isn't over yet.
            self._messages = [m for m in self._messages if not m.speculation]
        elif isinstance(frame, InterruptionFrame):
            self._turn = None

        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.broadcast_frame(ExpressionMetadataFrame, service_name=self.name)
        elif isinstance(frame, LLMFullResponseStartFrame):
            await self._start_reply()

    def _on_tts_metadata(self, frame: TTSMetadataFrame):
        if frame.expression is None:
            logger.warning(
                f"{self}: {frame.service_name} performs no expression, "
                "so replies pass through untagged"
            )
            self._space = None
            return
        self._space = LabelSpace.resolve(frame.expression, self._policy)

    def _on_user_turn(self, frame: LLMContextUserTurnFrame):
        message = Message("user", frame.text, fold(frame.perception), frame.speculation)
        # A turn replaces the speculation before it, where it stands: the early
        # reply that answered it may already follow it.
        for i in range(len(self._messages) - 1, -1, -1):
            if self._messages[i].speculation:
                self._messages[i] = message
                return
        self._messages.append(message)

    async def _start_reply(self):
        # Text left from a reply cut short isn't part of this one.
        await self.reset()
        self._sentences = []
        if not self._space:
            self._turn = None
            return
        self._turn = TurnContext(list(self._messages), self._space, self._policy, self._labels)
        self._renderer = ReplyRenderer(self._space)
        self._labels = []

    async def _tag(self, sentence: str, text_type: TextType | str) -> str:
        """Tag one sentence of the reply; text that isn't spoken stays as written."""
        if (
            not self._turn
            or not self._renderer
            or not self._model
            or self._labels is None
            or self._skip_tts
        ):
            return sentence
        # The labels are offsets into the sentence as the model reads it.
        plain = " ".join(sentence.split())
        earlier = list(self._sentences)
        self._sentences.append(plain)
        try:
            labels = await asyncio.to_thread(self._model.tag, self._turn, earlier, plain)
            labels = self._capped_breaths(labels)
            tagged = self._renderer.render(plain, labels)
        except Exception as e:
            # A transform that raises drops the sentence, so it's spoken untagged.
            await self.push_error(f"{self}: couldn't tag a sentence, spoken untagged: {e}", e)
            return plain
        self._labels.append(labels)
        return tagged

    def _capped_breaths(self, labels: SentenceLabels) -> SentenceLabels:
        """The labels without a breath past the model's limit for a reply."""
        cap = self._model.breaths_per_reply if self._model else None
        if cap is None:
            return labels
        taken = sum(_breaths(sentence) for sentence in self._labels or [])
        if labels.sound_before == "breath":
            if taken >= cap:
                labels = replace(labels, sound_before=None)
            taken += 1
        if labels.sound_after == "breath" and taken >= cap:
            labels = replace(labels, sound_after=None)
        return labels


def _breaths(labels: SentenceLabels) -> int:
    return (labels.sound_before == "breath") + (labels.sound_after == "breath")
