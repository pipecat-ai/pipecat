#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The processor that tags an LLM's plain replies with Pipecat Tone's tagger."""

import asyncio
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Literal

from loguru import logger

from pipecat.expressive.expression import (
    CORE_DELIVERY,
    CORE_SOUNDS,
    ExpressionDelivery,
    ExpressionSound,
)
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
from pipecat.processors.tone.tagger._labels import (
    BREATH_KINDS,
    LabelSpace,
    ReplyRenderer,
    SentenceLabels,
)
from pipecat.processors.tone.tagger.model import TAGGER_REVISION, PipecatToneTaggerModel
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
        size: Literal["17m", "32m"] = "17m",
        revision: str = TAGGER_REVISION,
        model_dir: str | Path | None = None,
        threads: int = 1,
        vocabulary: dict[ExpressionDelivery | ExpressionSound, list[str]] | None = None,
        **kwargs,
    ):
        """Initialize the processor.

        Args:
            size: The model's size: ``"17m"``, or ``"32m"``, which agrees with its
                teacher a little more often and takes nearly three times as long a
                sentence.
            revision: The model's release on Hugging Face, such as ``"tagger-rc1"``.
            model_dir: A folder to load the model from instead of downloading one,
                laid out as a release's ``tagger/17m``, such as for a model you
                trained or keep yourself. ``size`` and ``revision`` are then unused.
            threads: The CPU threads for one sentence.
            vocabulary: Names to write first for a delivery, sound or kind of breath
                (``CORE_DELIVERY``, ``CORE_SOUNDS``, ``BREATH_KINDS``), best first,
                such as ``{ExpressionDelivery.WARM: ["cheerful"]}``. A name is written
                only where the TTS service performs it and the policy offers it;
                elsewhere the label's universal name stands. A name written for
                ``ExpressionSound.BREATH`` is written for every breath, rather than its
                kinds' names.
            **kwargs: Additional arguments passed to the parent.

        Raises:
            ValueError: The vocabulary names a label the tagger doesn't write.
        """
        labels = (*CORE_DELIVERY, *CORE_SOUNDS, *BREATH_KINDS)
        if unknown := sorted(set(vocabulary or {}) - set(labels)):
            raise ValueError(f"the tagger writes none of {unknown}; it writes {list(labels)}")
        super().__init__(text_aggregator=SkipTagsAggregator(EXPRESSION_TAGS), **kwargs)
        self.add_text_transformer(self._tag, TextType.SENTENCE)
        self._model: PipecatToneTaggerModel | None = None
        self._size: Literal["17m", "32m"] = size
        self._revision = revision
        self._model_dir = Path(model_dir) if model_dir else None
        self._threads = threads
        self._vocabulary = vocabulary or {}
        self._policy = ExpressionPolicy()
        self._space: LabelSpace | None = None
        self._messages: list[Message] = []
        self._turn: TurnContext | None = None
        self._renderer: ReplyRenderer | None = None
        self._sentences: list[str] = []
        # The labels of the reply being tagged, which the next reply's model
        # reads; None before the first reply.
        self._labels: list[SentenceLabels] | None = None
        # Draws each breath's kind, seeded by the call so far, so a call replays alike.
        self._draw = random.Random()

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
            ("tagger", self._model_dir or (self._size, self._revision), self._threads),
            self._open_model,
        )

    def _open_model(self) -> PipecatToneTaggerModel:
        if self._model_dir:
            return PipecatToneTaggerModel(self._model_dir, threads=self._threads)
        return PipecatToneTaggerModel.download(
            size=self._size, revision=self._revision, threads=self._threads
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
        self._space = LabelSpace.resolve(
            frame.expression, self._policy, frame.service_name, self._vocabulary
        )

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
        self._draw = random.Random("\n".join(m.text for m in self._turn.messages))

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
        started = time.perf_counter()
        try:
            labels = await asyncio.to_thread(self._model.tag, self._turn, earlier, plain)
            labels = self._capped_breaths(labels)
            tagged = self._renderer.render(plain, self._drawn(labels))
        except Exception as e:
            # A transform that raises drops the sentence, so it's spoken untagged.
            await self.push_error(f"{self}: couldn't tag a sentence, spoken untagged: {e}", e)
            return plain
        logger.debug(f"{self}: tagged in {(time.perf_counter() - started) * 1000:.1f} ms: {tagged}")
        # The next reply's model reads a breath as a breath, whatever kind it was drawn as.
        self._labels.append(labels)
        return tagged

    def _drawn(self, labels: SentenceLabels) -> SentenceLabels:
        """The labels with a breath drawn as a kind the service performs, at the model's shares.

        A breath stays plain when the model or the service has no kinds, or when it's
        written as a name the developer gave for every breath.
        """
        if labels.sound_before != "breath" or not self._model or not self._turn:
            return labels
        if self._turn.space.sounds.get("breath") in self._vocabulary.get(
            ExpressionSound.BREATH, []
        ):
            return labels
        breaths = self._turn.space.breaths
        kinds = {k: share for k, share in self._model.breath_kinds.items() if k in breaths}
        if not kinds:
            return labels
        kind = self._draw.choices(list(kinds), list(kinds.values()))[0]
        return replace(labels, sound_before=kind)

    def _capped_breaths(self, labels: SentenceLabels) -> SentenceLabels:
        """The labels without a breath past the model's limit for a reply.

        The model breathes only before a sentence.
        """
        cap = self._model.breaths_per_reply if self._model else None
        if cap is None or labels.sound_before != "breath":
            return labels
        taken = sum(sentence.sound_before == "breath" for sentence in self._labels or [])
        return replace(labels, sound_before=None) if taken >= cap else labels
