#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Sentence language selection across streaming generations and observers."""

import pytest

from pipecat.frames.frames import (
    AggregatedTextFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.observers.base_observer import FrameProcessed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi.observer import RTVIObserver
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
        self.sentences.append((frame.text, self._generation_text_aggregation_language))


@pytest.mark.parametrize(
    ("language", "model"),
    [
        (Language.DE, "german"),
        ("de-DE", "german"),
        ("de_AT", "german"),
        ("pt-BR", "portuguese"),
        ("nb-NO", "norwegian"),
        ("nn", "norwegian"),
        ("sl", "slovene"),
        ("ml", "malayalam"),
        ("german", "german"),
        ("ja", "english"),
        ("auto", "english"),
        (None, "english"),
    ],
)
def test_language_resolution(language, model):
    assert resolve_sentence_tokenizer_language(language) == model


def test_tts_preserves_language_before_provider_conversion():
    tts = RecordingTTS(language=Language.DE)
    assert tts._settings.language == "provider-de"
    assert tts.text_aggregation_language == "german"


@pytest.mark.asyncio
async def test_streaming_models_are_independent():
    german = SimpleTextAggregator(language="de")
    english = SimpleTextAggregator(language="en")
    assert [a.text async for a in german.aggregate("Das ist z.B. w")] == []
    assert [a.text async for a in english.aggregate("Das ist z.B. w")] == ["Das ist z.B."]
    assert [a.text async for a in german.aggregate("ichtig. Weiter")] == ["Das ist z.B. wichtig."]
    assert match_endofsentence("こんにちは。次", language="ja") == len("こんにちは。")


@pytest.mark.asyncio
async def test_language_update_applies_to_next_generation():
    tts = RecordingTTS()
    frames = [
        LLMFullResponseStartFrame(),
        LLMTextFrame("Das ist z.B."),
        TTSUpdateSettingsFrame(delta=TTSSettings(language=Language.EN)),
        LLMTextFrame(" wichtig."),
        LLMFullResponseEndFrame(),
        LLMFullResponseStartFrame(),
        LLMTextFrame("Das ist z.B. wichtig."),
        LLMFullResponseEndFrame(),
    ]
    await run_test(tts, frames_to_send=frames)
    assert tts.sentences == [
        ("Das ist z.B. wichtig.", "german"),
        ("Das ist z.B.", "english"),
        ("wichtig.", "english"),
    ]
    assert tts.get_text_aggregation_language(frames[0]) == "german"
    assert tts.get_text_aggregation_language(frames[3]) == "german"
    assert tts.get_text_aggregation_language(frames[5]) == "english"


@pytest.mark.asyncio
async def test_explicit_override_survives_tts_language_updates():
    tts = RecordingTTS(language=None, text_aggregation_language="de")
    await run_test(
        tts,
        frames_to_send=[
            TTSUpdateSettingsFrame(delta=TTSSettings(language=Language.EN)),
            LLMFullResponseStartFrame(),
            LLMTextFrame("Das ist z.B. wichtig."),
            LLMFullResponseEndFrame(),
        ],
    )
    assert tts.sentences == [("Das ist z.B. wichtig.", "german")]


@pytest.mark.asyncio
async def test_parallel_channels_use_the_same_language_for_slicing():
    aggregator = _ParallelSentenceAggregator(language="de")
    text = "Das ist z.B. wichtig. Weiter geht es. N"
    result = [a async for a in aggregator.aggregate(text, text, text)]
    assert [a.tts_text.strip() for a in result] == [
        "Das ist z.B. wichtig.",
        "Weiter geht es.",
    ]
    assert all(a.tts_text == a.llm_text == a.user_facing_text for a in result)


@pytest.mark.asyncio
async def test_observer_uses_recorded_generation_not_current_settings():
    tts = RecordingTTS()
    frames = [
        LLMFullResponseStartFrame(),
        LLMTextFrame("Das ist z.B. w"),
        LLMTextFrame("ichtig."),
        LLMFullResponseEndFrame(),
        TTSUpdateSettingsFrame(delta=TTSSettings(language=Language.EN)),
    ]
    await run_test(tts, frames_to_send=frames)
    assert tts.text_aggregation_language == "english"

    messages = []
    observer = RTVIObserver(tts_service=tts)

    async def record(message, **kwargs):
        messages.append(message)

    observer.send_rtvi_message = record
    for frame in frames[:2]:
        await observer.on_process_frame(FrameProcessed(tts, frame, FrameDirection.DOWNSTREAM, 0))
    assert messages == []
    await observer.on_process_frame(FrameProcessed(tts, frames[2], FrameDirection.DOWNSTREAM, 0))
    assert len(messages) == 1
    assert messages[0].data.text == "Das ist z.B. wichtig."

    other_tts = RecordingTTS(language=Language.EN)
    await observer.on_process_frame(
        FrameProcessed(other_tts, LLMTextFrame("Unrelated."), FrameDirection.DOWNSTREAM, 0)
    )
    assert len(messages) == 1


@pytest.mark.asyncio
async def test_targeted_update_does_not_change_other_service():
    tts = RecordingTTS()
    other = RecordingTTS(language=Language.EN)
    await run_test(
        tts,
        frames_to_send=[
            TTSUpdateSettingsFrame(service=other, delta=TTSSettings(language=Language.EN)),
            LLMFullResponseStartFrame(),
            LLMTextFrame("Das ist z.B. wichtig."),
            LLMFullResponseEndFrame(),
        ],
    )
    assert tts.sentences == [("Das ist z.B. wichtig.", "german")]


@pytest.mark.asyncio
async def test_interruption_discards_buffer_and_allows_new_language():
    tts = RecordingTTS()
    await run_test(
        tts,
        frames_to_send=[
            LLMFullResponseStartFrame(),
            LLMTextFrame("Das ist z.B."),
            SleepFrame(sleep=0.05),
            InterruptionFrame(),
            SleepFrame(sleep=0.05),
            TTSUpdateSettingsFrame(delta=TTSSettings(language=Language.EN)),
            LLMFullResponseStartFrame(),
            LLMTextFrame("Das ist z.B. wichtig."),
            LLMFullResponseEndFrame(),
        ],
    )
    assert tts.sentences == [("Das ist z.B.", "english"), ("wichtig.", "english")]


@pytest.mark.asyncio
async def test_service_snapshots_do_not_overwrite_each_other():
    german = RecordingTTS()
    english = RecordingTTS(language=Language.EN)
    start = LLMFullResponseStartFrame()
    text = LLMTextFrame("Hello.")
    for tts in (german, english):
        await run_test(tts, frames_to_send=[start, text, LLMFullResponseEndFrame()])
    assert german.get_text_aggregation_language(start) == "german"
    assert english.get_text_aggregation_language(start) == "english"
    assert german.get_text_aggregation_language(text) == "german"
    assert english.get_text_aggregation_language(text) == "english"


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

    await feed("german", "Das ist z.B.", "de")
    await feed("english", "Hello. Next", "en")
    await feed("german", " wichtig. Weiter", "en")
    assert [slot.frame.text.strip() for slot in sequencer._slots] == [
        "Hello.",
        "Das ist z.B. wichtig.",
    ]
