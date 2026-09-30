#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Sentence language selection from ordered TTS settings in streaming text."""

import pytest

from pipecat.frames.frames import (
    AggregatedTextFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transcriptions.language import Language
from pipecat.utils.context.aggregated_frame_sequencer import (
    AggregatedFrameSequencer,
    _ParallelSentenceAggregator,
)
from pipecat.utils.string import match_endofsentence, resolve_sentence_tokenizer_language
from pipecat.utils.text.base_text_aggregator import AggregationType
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator


class RecordingTTS(TTSService):
    """Capture aggregation results without a provider connection."""

    def __init__(self, language=Language.DE, **kwargs):
        super().__init__(settings=TTSSettings(language=language, voice=None, model=None), **kwargs)
        self.sentences = []

    def language_to_service_language(self, language):
        return f"provider-{language.value}"

    async def run_tts(self, text, context_id):
        yield None

    async def _push_tts_frames(self, frame: AggregatedTextFrame, *args, **kwargs):
        self.sentences.append((frame.text, self.text_aggregation_language))


@pytest.mark.parametrize(
    ("language", "model"),
    [
        (Language.DE, "de"),
        ("de-DE", "de"),
        ("de_AT", "de"),
        ("pt-BR", "pt"),
        ("nb-NO", "nb"),
        ("nn", "nn"),
        ("sl", "sl"),
        ("ml", "ml"),
        ("unknown", "unknown"),
        ("ja", "ja"),
        ("auto", "en"),
        (None, "en"),
    ],
)
def test_language_resolution(language, model):
    assert resolve_sentence_tokenizer_language(language) == model


def test_tts_preserves_language_before_provider_conversion():
    tts = RecordingTTS(language=Language.DE)
    assert tts._settings.language == "provider-de"
    assert tts.text_aggregation_language == "de"


@pytest.mark.asyncio
async def test_streaming_models_are_independent():
    german = SimpleTextAggregator(language="de")
    english = SimpleTextAggregator(language="en")
    assert [a.text async for a in german.aggregate("Das ist bzw. w")] == []
    assert [a.text async for a in english.aggregate("Das ist bzw. w")] == ["Das ist bzw."]
    assert [a.text async for a in german.aggregate("ichtig. Weiter")] == ["Das ist bzw. wichtig."]
    assert match_endofsentence("こんにちは。次", language="ja") == len("こんにちは。")


@pytest.mark.asyncio
async def test_language_update_applies_to_following_text_and_preserves_buffer():
    tts = RecordingTTS()
    frames = [
        LLMFullResponseStartFrame(),
        LLMTextFrame("Das ist bzw."),
        TTSUpdateSettingsFrame(delta=TTSSettings(language=Language.EN)),
        LLMTextFrame(" wichtig."),
        LLMFullResponseEndFrame(),
        LLMFullResponseStartFrame(),
        LLMTextFrame("Das ist bzw. wichtig."),
        LLMFullResponseEndFrame(),
    ]
    await run_test(tts, frames_to_send=frames)
    assert tts.sentences == [
        ("Das ist bzw.", "en"),
        ("wichtig.", "en"),
        ("Das ist bzw.", "en"),
        ("wichtig.", "en"),
    ]


@pytest.mark.asyncio
async def test_unspecified_language_defaults_to_english_and_updates_without_start_frame():
    tts = RecordingTTS(language=None)
    assert tts.text_aggregation_language == "en"
    await run_test(
        tts,
        frames_to_send=[
            LLMTextFrame("Das ist bzw. wichtig."),
            LLMFullResponseEndFrame(),
            TTSUpdateSettingsFrame(delta=TTSSettings(language=Language.DE)),
            LLMTextFrame("Das ist bzw. wichtig."),
            LLMFullResponseEndFrame(),
            TTSUpdateSettingsFrame(delta=TTSSettings(language=None)),
            LLMTextFrame("Das ist bzw. wichtig."),
            LLMFullResponseEndFrame(),
        ],
    )
    assert tts.sentences == [
        ("Das ist bzw.", "en"),
        ("wichtig.", "en"),
        ("Das ist bzw. wichtig.", "de"),
        ("Das ist bzw.", "en"),
        ("wichtig.", "en"),
    ]


@pytest.mark.asyncio
async def test_parallel_channels_use_the_same_language_for_slicing():
    aggregator = _ParallelSentenceAggregator(language="de")
    text = "Das ist bzw. wichtig. Weiter geht es. N"
    result = [a async for a in aggregator.aggregate(text, text, text)]
    assert [a.tts_text.strip() for a in result] == [
        "Das ist bzw. wichtig.",
        "Weiter geht es.",
    ]
    assert all(a.tts_text == a.llm_text == a.user_facing_text for a in result)


@pytest.mark.asyncio
async def test_targeted_update_does_not_change_other_service():
    tts = RecordingTTS()
    other = RecordingTTS(language=Language.EN)
    await run_test(
        tts,
        frames_to_send=[
            TTSUpdateSettingsFrame(service=other, delta=TTSSettings(language=Language.EN)),
            LLMFullResponseStartFrame(),
            LLMTextFrame("Das ist bzw. wichtig."),
            LLMFullResponseEndFrame(),
        ],
    )
    assert tts.sentences == [("Das ist bzw. wichtig.", "de")]


@pytest.mark.asyncio
async def test_interruption_discards_buffer_and_allows_new_language():
    tts = RecordingTTS()
    await run_test(
        tts,
        frames_to_send=[
            LLMFullResponseStartFrame(),
            LLMTextFrame("Das ist bzw."),
            SleepFrame(sleep=0.05),
            InterruptionFrame(),
            SleepFrame(sleep=0.05),
            TTSUpdateSettingsFrame(delta=TTSSettings(language=Language.EN)),
            LLMFullResponseStartFrame(),
            LLMTextFrame("Das ist bzw. wichtig."),
            LLMFullResponseEndFrame(),
        ],
    )
    assert tts.sentences == [("Das ist bzw.", "en"), ("wichtig.", "en")]


@pytest.mark.asyncio
async def test_streaming_context_retains_language_after_another_context_starts():
    sequencer = AggregatedFrameSequencer(streaming=True)

    async def feed(context, text, language):
        await sequencer.register_spoken(
            AggregatedTextFrame(text, AggregationType.TOKEN),
            context,
            text,
            append_to_context=True,
            language=language,
        )

    await feed("de", "Das ist bzw.", "de")
    await feed("en", "Hello. Next", "en")
    await feed("de", " wichtig. Weiter", "de")
    assert [slot.frame.text.strip() for slot in sequencer._slots] == [
        "Hello.",
        "Das ist bzw. wichtig.",
    ]


@pytest.mark.asyncio
async def test_streaming_context_language_updates_with_incoming_text():
    sequencer = AggregatedFrameSequencer(streaming=True)
    for text, language in [("Das ist bzw.", "de"), (" wichtig. Weiter", "en")]:
        await sequencer.register_spoken(
            AggregatedTextFrame(text, AggregationType.TOKEN),
            "context",
            text,
            append_to_context=True,
            language=language,
        )
    assert [slot.frame.text.strip() for slot in sequencer._slots] == ["Das ist bzw.", "wichtig."]
