#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The description of a pipeline's text path, and how to build its processors.

A :class:`TextFlowSetup` holds everything downstream of the LLM that touches
text: the optional ``LLMTextProcessor`` and its aggregator and transforms, the
TTS service's aggregation, skipping and transforms, and the ``RTVIObserver``'s
output transforms. It is plain data so it can be saved as YAML, edited in the
page and sent to the server. The ``build_*`` functions turn each part into the
real Pipecat object it describes.
"""

from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from pipecat.utils.text.base_text_aggregator import BaseTextAggregator, TextType
from pipecat.utils.text.pattern_pair_aggregator import MatchAction, PatternPairAggregator
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator
from pipecat.utils.text.skip_tags_aggregator import SkipTagsAggregator
from pipecat.utils.text.transforms.text_transform import TextTransform

from .transforms import build_transform


class PatternSpec(BaseModel):
    """A pattern pair for a ``PatternPairAggregator``.

    Parameters:
        type: The text type the matched content is given.
        start: The text that opens the pattern, such as ``<code>``.
        end: The text that closes it, such as ``</code>``.
        action: What the aggregator does with a match: ``remove`` it from the
            sentence, ``keep`` it in place, or ``aggregate`` it as its own segment.
    """

    type: str
    start: str
    end: str
    action: Literal["remove", "keep", "aggregate"] = "aggregate"


class AggregatorSpec(BaseModel):
    """Which text aggregator a stage uses and how it is configured.

    Parameters:
        kind: ``simple`` splits on sentence boundaries, ``pattern_pair`` also cuts
            tagged spans into their own segments, ``skip_tags`` holds tagged spans
            back until they close so they are never split.
        aggregation_type: What the aggregator produces: a sentence, a word or a token.
        patterns: The pattern pairs a ``pattern_pair`` aggregator detects.
        skip_tags: The ``[start, end]`` tag pairs a ``skip_tags`` aggregator holds back.
    """

    kind: Literal["simple", "pattern_pair", "skip_tags"] = "simple"
    aggregation_type: Literal["sentence", "word", "token"] = "sentence"
    patterns: list[PatternSpec] = Field(default_factory=list)
    skip_tags: list[list[str]] = Field(default_factory=list)


class TransformSpec(BaseModel):
    """A transform registered under a name, applied to one text type.

    Parameters:
        text_type: The text type the transform applies to, or ``*`` for all.
        name: The transform's name in :mod:`pipecat.textflow.transforms`.
        options: Options passed to the transform's factory.
    """

    text_type: str = "*"
    name: str
    options: dict[str, Any] = Field(default_factory=dict)


class LLMTextProcessorSpec(BaseModel):
    """The ``LLMTextProcessor`` between the LLM and the TTS.

    Parameters:
        aggregator: How LLM text is cut into segments.
        transforms: Transforms that rewrite each segment for every consumer.
    """

    aggregator: AggregatorSpec = Field(default_factory=AggregatorSpec)
    transforms: list[TransformSpec] = Field(default_factory=list)


class TokenizationSpec(BaseModel):
    """How the stand-in TTS provider reports the words it spoke.

    Parameters:
        style: ``words`` reports each whitespace-separated word, ``split_punctuation``
            reports trailing punctuation as its own token, ``characters`` reports
            every character (as some providers do for CJK text).
        pre_merge_tokens: Merge punctuation tokens into the word before them,
            as a service does for a provider that splits them off.
        includes_inter_frame_spaces: Report that tokens carry their own spacing, so
            consumers do not add spaces between them.
        drop_words: Words the provider never reports, to show how a frame completes anyway.
        gap_s: Seconds between one word's timestamp and the next.
    """

    style: Literal["words", "split_punctuation", "characters"] = "words"
    pre_merge_tokens: bool = False
    includes_inter_frame_spaces: bool = False
    drop_words: list[str] = Field(default_factory=list)
    gap_s: float = 0.05


class TTSSpec(BaseModel):
    """The TTS service's text handling.

    Parameters:
        word_timestamps: Whether the provider reports word timestamps. With them the
            service emits one ``TTSTextFrame`` per word as it is spoken; without
            them it pushes the whole segment once synthesis is done.
        delivery: ``http`` returns audio from the request, ``websocket`` delivers it
            later over an audio context, as a streaming provider does.
        aggregation_mode: ``sentence`` waits for sentence boundaries, ``token``
            streams tokens straight to the provider.
        aggregator: The service's own aggregator, used when no ``LLMTextProcessor``
            precedes it. None uses the service default.
        skip_text_types: Text types the service never speaks.
        transforms: Transforms that rewrite a segment for the provider only.
        markdown_filter: Add the deprecated ``MarkdownTextFilter`` as a text filter.
        append_trailing_space: Add a trailing space to each segment sent to the provider.
        tokenization: How the stand-in provider reports words.
        live: Run the bot's own TTS service, calling its provider, instead of the
            stand-in. None does so whenever the setup names a ``bot``; False keeps
            the stand-in, with the bot's text handling, for a quick offline run.
            Live needs the bot's keys and the time the provider takes: the run
            plays the audio at its real pace. ``word_timestamps``, ``delivery``
            and ``tokenization`` describe the stand-in and do not apply.
    """

    word_timestamps: bool = True
    delivery: Literal["http", "websocket"] = "http"
    aggregation_mode: Literal["sentence", "token"] = "sentence"
    aggregator: AggregatorSpec | None = None
    skip_text_types: list[str] = Field(default_factory=list)
    transforms: list[TransformSpec] = Field(default_factory=list)
    markdown_filter: bool = False
    append_trailing_space: bool = False
    tokenization: TokenizationSpec = Field(default_factory=TokenizationSpec)
    live: bool | None = None


class RTVISpec(BaseModel):
    """The ``RTVIObserver`` that turns frames into client messages.

    Parameters:
        enabled: Whether to attach the observer at all.
        skip_text_types: Text types the observer never sends to the client.
        transforms: Transforms applied to a segment before it reaches the client.
        bot_llm: Whether to send the bot's LLM messages: ``bot-llm-started``,
            ``bot-llm-stopped`` and the deprecated ``bot-llm-text`` for each raw token.
            Off by default; ``bot-output`` carries the text clients use.
        bot_tts: Whether to send the bot's TTS messages: ``bot-tts-started``,
            ``bot-tts-stopped`` and the deprecated ``bot-tts-text`` for each word as it
            is spoken. Off by default; ``bot-output`` progress reports the same.
    """

    enabled: bool = True
    skip_text_types: list[str] = Field(default_factory=list)
    transforms: list[TransformSpec] = Field(default_factory=list)
    bot_llm: bool = False
    bot_tts: bool = False


class ChunkingSpec(BaseModel):
    """How the LLM's response is streamed as ``LLMTextFrame`` chunks.

    Parameters:
        style: ``whole`` sends the response as one frame, ``words`` one frame per word
            (with its trailing space), ``characters`` one frame per ``size``
            characters, ``custom`` sends ``chunks`` as given.
        size: The chunk length for ``characters``.
        chunks: The chunks for ``custom``.
        interval_s: Seconds between chunks, as a streaming LLM would space them.
            Zero sends them all at once, so every chunk is queued before any is
            processed.
    """

    style: Literal["whole", "words", "characters", "custom"] = "words"
    size: int = 4
    chunks: list[str] = Field(default_factory=list)
    interval_s: float = 0.03


class TextFlowSetup(BaseModel):
    """A pipeline's text path, from the LLM's output to its three consumers.

    Parameters:
        name: A label for the setup.
        description: What the setup shows.
        llm_text_processor: The ``LLMTextProcessor``, or None when the TTS aggregates.
        tts: The TTS service.
        rtvi: The RTVI observer.
        chunking: How the response streams from the LLM.
        sample_text: A response that shows the setup off.
        bot: A bot file whose own processors the run uses. When set, the
            ``LLMTextProcessor``, the TTS service's aggregator, skipped text types,
            filters and transforms, and the RTVI observer's parameters are the bot's,
            and the matching fields here only describe them. The stand-in knobs
            still apply: ``tts.word_timestamps``, ``tts.delivery``,
            ``tts.tokenization``, ``rtvi.enabled`` and ``chunking``.
    """

    name: str = "custom"
    description: str = ""
    bot: str | None = None
    llm_text_processor: LLMTextProcessorSpec | None = Field(default_factory=LLMTextProcessorSpec)
    tts: TTSSpec = Field(default_factory=TTSSpec)
    rtvi: RTVISpec = Field(default_factory=RTVISpec)
    chunking: ChunkingSpec = Field(default_factory=ChunkingSpec)
    sample_text: str = "Hello there. How are you today?"

    @classmethod
    def load(cls, path: str) -> "TextFlowSetup":
        """Read a setup from a YAML or JSON file."""
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f) or {})

    def to_yaml(self) -> str:
        """The setup as YAML, with defaults left out."""
        return yaml.safe_dump(self.model_dump(exclude_defaults=True), sort_keys=False)

    @classmethod
    def preset(cls, name: str) -> "TextFlowSetup":
        """One of the bundled setups, by name.

        Raises:
            KeyError: If no preset has that name.
        """
        for preset in presets():
            if preset.name == name:
                return preset.model_copy(deep=True)
        raise KeyError(f"Unknown preset {name!r}. Presets: {', '.join(p.name for p in presets())}")


def build_aggregator(spec: AggregatorSpec) -> BaseTextAggregator:
    """Build the aggregator a spec describes."""
    aggregation_type = TextType(spec.aggregation_type)
    if spec.kind == "pattern_pair":
        aggregator = PatternPairAggregator(aggregation_type=aggregation_type)
        for pattern in spec.patterns:
            aggregator.add_pattern(
                type=pattern.type,
                start_pattern=pattern.start,
                end_pattern=pattern.end,
                action=MatchAction(pattern.action),
            )
        return aggregator
    if spec.kind == "skip_tags":
        tags = [(start, end) for start, end in spec.skip_tags]
        return SkipTagsAggregator(tags, aggregation_type=aggregation_type)
    return SimpleTextAggregator(aggregation_type=aggregation_type)


def build_transforms(specs: list[TransformSpec]) -> list[tuple[str, TextTransform]]:
    """Build ``(text_type, transform)`` pairs from transform specs."""
    return [(spec.text_type, build_transform(spec.name, spec.options)) for spec in specs]


def _code_helper() -> TextFlowSetup:
    return TextFlowSetup(
        name="code-helper",
        description=(
            "The code-helper example: card numbers are spelled out for the TTS and "
            "redacted for the client, code blocks are never spoken."
        ),
        llm_text_processor=LLMTextProcessorSpec(
            aggregator=AggregatorSpec(
                kind="pattern_pair",
                patterns=[
                    PatternSpec(type="credit_card", start="<card>", end="</card>"),
                    PatternSpec(type="code", start="<code>", end="</code>"),
                ],
            )
        ),
        tts=TTSSpec(
            skip_text_types=["code"],
            transforms=[TransformSpec(text_type="credit_card", name="wrap")],
        ),
        rtvi=RTVISpec(transforms=[TransformSpec(text_type="credit_card", name="redact")]),
        sample_text=(
            "Your card is <card>1234-5678-9012-3456</card>. Run <code>npm install</code> to start."
        ),
    )


def _plain() -> TextFlowSetup:
    return TextFlowSetup(
        name="plain",
        description="A sentence aggregator and nothing else: all three texts are the same.",
        sample_text="Hello there. How are you today?",
    )


def _voice_formatting() -> TextFlowSetup:
    return TextFlowSetup(
        name="voice-formatting",
        description="Markdown and numbers rewritten for speech only; the context keeps them.",
        tts=TTSSpec(
            transforms=[
                TransformSpec(name="strip_markdown"),
                TransformSpec(name="expand_currency"),
                TransformSpec(name="normalize_dates"),
            ]
        ),
        sample_text="**Total:** $42.50, due 2024-01-15. Email a@b.com with questions.",
    )


def _rewrite_for_everyone() -> TextFlowSetup:
    return TextFlowSetup(
        name="rewrite-for-everyone",
        description=(
            "A transform on the LLMTextProcessor reaches the TTS, the context and the client; "
            "the same transform on the TTS reaches the provider only."
        ),
        llm_text_processor=LLMTextProcessorSpec(
            transforms=[TransformSpec(name="strip_emoji")],
        ),
        tts=TTSSpec(transforms=[TransformSpec(name="upper")]),
        sample_text="Great choice 🎉. Your order ships tomorrow 🚚.",
    )


def _tts_aggregates() -> TextFlowSetup:
    return TextFlowSetup(
        name="tts-aggregates",
        description="No LLMTextProcessor: the TTS service cuts the sentences itself.",
        llm_text_processor=None,
        tts=TTSSpec(transforms=[TransformSpec(name="expand_numbers")]),
        sample_text="Your room is 1994. Breakfast is at 7.",
    )


def _no_word_timestamps() -> TextFlowSetup:
    return TextFlowSetup(
        name="no-word-timestamps",
        description=(
            "A provider without word timestamps: the segment reaches the context and the "
            "client once, after synthesis, with no word progress."
        ),
        tts=TTSSpec(word_timestamps=False, transforms=[TransformSpec(name="strip_markdown")]),
        sample_text="This is **important**. Please read it twice.",
    )


def _streaming_tokens() -> TextFlowSetup:
    return TextFlowSetup(
        name="streaming-tokens",
        description="Token aggregation: text streams to the provider as it arrives.",
        llm_text_processor=None,
        tts=TTSSpec(aggregation_mode="token", delivery="websocket"),
        sample_text="Streaming tokens straight through. Sentence two follows.",
    )


def presets() -> list[TextFlowSetup]:
    """The bundled setups."""
    return [
        _plain(),
        _code_helper(),
        _voice_formatting(),
        _rewrite_for_everyone(),
        _tts_aggregates(),
        _no_word_timestamps(),
        _streaming_tokens(),
    ]
