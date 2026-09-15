"""Sentencex adapter offset and incremental aggregation contracts."""

import pytest

from pipecat.utils.string import match_endofsentence
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator


@pytest.mark.parametrize(
    ("text", "language", "prefix"),
    [
        ("Hello.   Next", "en", "Hello."),
        ("Hello.\n\nNext", "en", "Hello."),
        ("\n\nHello. Next", "en", "\n\nHello."),
        ("👋 Hello. 😀 Next", "en", "👋 Hello."),
        ("Cafe\u0301 is open. Next", "en", "Cafe\u0301 is open."),
        ("こんにちは。次", "ja", "こんにちは。"),
        ("你好。下一句", "zh", "你好。"),
        ("नमस्ते। यह", "hi", "नमस्ते।"),
        ("هل أنت بخير؟ نعم", "ar", "هل أنت بخير؟"),
        ("Hello. Next", "unrecognized", "Hello."),
    ],
)
def test_source_character_offsets(text, language, prefix):
    boundary = match_endofsentence(text, language=language)
    assert boundary == len(prefix)
    assert text[:boundary] == prefix


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk_size", [1, 3, 1000])
@pytest.mark.parametrize(
    ("language", "sentences"),
    [
        ("en", ["Dr. Smith is here.", "It costs $29.95."]),
        ("de-DE", ["Das ist z.B. wichtig.", "Weiter geht es."]),
        ("pt-BR", ["O Sr. Silva chegou.", "Depois saiu."]),
        ("fr", ["Voir p. 12 pour les détails.", "Ensuite continuez."]),
        ("pl", ["Prof. Kowalski przyszedł.", "Potem wyszedł."]),
        ("it", ["Il dott. Rossi arriva.", "Poi parte."]),
        ("nl", ["Dr. Jansen komt.", "Daarna vertrekt hij."]),
        ("ja", ["こんにちは。", "次の文です。"]),
    ],
)
async def test_chunk_boundaries_do_not_change_sentences(chunk_size, language, sentences):
    text = " ".join(sentences)
    aggregator = SimpleTextAggregator(language=language)
    actual = []
    for start in range(0, len(text), chunk_size):
        actual.extend(
            [a.text async for a in aggregator.aggregate(text[start : start + chunk_size])]
        )
    tail = await aggregator.flush()
    if tail:
        actual.append(tail.text)
    assert actual == sentences


@pytest.mark.asyncio
async def test_pronoun_boundary_with_streamed_sentence_starter():
    aggregator = SimpleTextAggregator(language="en")
    actual = [
        a.text
        async for a in aggregator.aggregate("We make a good team, you and I. Did you see him?")
    ]
    tail = await aggregator.flush()
    if tail:
        actual.append(tail.text)
    assert actual == ["We make a good team, you and I.", "Did you see him?"]


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk_size", [1, 3, 1000])
@pytest.mark.parametrize(
    "sentences",
    [
        ["We make a good team, you and I.", "Did you see him?"],
        ["Albert I. Jones is here.", "Hello."],
        ["Albert I. Douglas is here.", "Hello."],
        ['He said "Hello."', "Then he left."],
        ["This is the U.S. Department of State.", "Next."],
        ["Hello!?", "Next."],
        ["We make a good team, you and I.", "Did?"],
    ],
)
async def test_lookahead_retry_preserves_sentences(chunk_size, sentences):
    await test_chunk_boundaries_do_not_change_sentences(chunk_size, "en", sentences)


@pytest.mark.asyncio
async def test_retry_emits_on_word_delimiter_without_splitting_partial_starters():
    aggregator = SimpleTextAggregator()
    assert [a.text async for a in aggregator.aggregate("We make a good team, you and I. D")] == []
    assert [a.text async for a in aggregator.aggregate("id")] == []
    assert [a.text async for a in aggregator.aggregate(" ")] == ["We make a good team, you and I."]
    assert aggregator.text.text == "Did"

    initials = SimpleTextAggregator()
    assert [a.text async for a in initials.aggregate("Albert I. Do")] == []
    assert [a.text async for a in initials.aggregate("uglas ")] == []
    assert (await initials.flush()).text == "Albert I. Douglas"


@pytest.mark.asyncio
async def test_first_character_fast_path_still_emits_immediately():
    aggregator = SimpleTextAggregator()
    assert [a.text async for a in aggregator.aggregate("Hello. ")] == []
    assert [a.text async for a in aggregator.aggregate("N")] == ["Hello."]
    assert aggregator.text.text == "N"


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["reset", "handle_interruption", "flush"])
async def test_pending_retry_is_cleared_between_generations(ending):
    aggregator = SimpleTextAggregator()
    text = "We make a good team, you and I. Did"
    assert [a.text async for a in aggregator.aggregate(text)] == []
    result = await getattr(aggregator, ending)()
    if ending == "flush":
        assert result.text == text
    assert [a.text async for a in aggregator.aggregate("Dr. Smith is here. Next")] == [
        "Dr. Smith is here."
    ]
    assert (await aggregator.flush()).text == "Next"


@pytest.mark.asyncio
async def test_unbroken_lookahead_has_bounded_tokenizer_work(monkeypatch):
    from pipecat.utils.text import simple_text_aggregator

    calls = []

    def unresolved(text, *, language):
        calls.append(text)
        return 0

    monkeypatch.setattr(simple_text_aggregator, "match_endofsentence", unresolved)
    aggregator = SimpleTextAggregator()
    text = "I. " + "x" * 10_000 + " more words without punctuation"
    assert [a.text async for a in aggregator.aggregate(text)] == []
    assert calls == ["I. x", "I. " + "x" * 10_000 + " "]
    assert (await aggregator.flush()).text == text


@pytest.mark.asyncio
@pytest.mark.parametrize("quote", ['"', "'", "“", "‘", "«", "「"])
@pytest.mark.parametrize("chunk_size", [1, 3, 1000])
async def test_open_quote_preserves_abbreviations_without_waiting_for_close(quote, chunk_size):
    text = f"She said, {quote}Dr. Smith is here. Next sentence"
    aggregator = SimpleTextAggregator()
    actual = []
    for offset in range(0, len(text), chunk_size):
        actual.extend(
            [a.text async for a in aggregator.aggregate(text[offset : offset + chunk_size])]
        )
    assert actual == [f"She said, {quote}Dr. Smith is here."]
    assert (await aggregator.flush()).text == "Next sentence"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sentences",
    [
        ['She said, "Dr. Smith is here."', "Then she left."],
        ["She said, “Dr. Smith is here.”", "Then she left."],
        ['She said, "Hello.', 'Next sentence."'],
        ["Don't call Dr. Smith.", "He's busy."],
        ["The doctor's here.", "Let's go."],
    ],
)
async def test_quotation_probe_preserves_other_boundaries(sentences):
    await test_chunk_boundaries_do_not_change_sentences(1, "en", sentences)


@pytest.mark.parametrize("quote", ['"', "“", "'", "«"])
def test_quoted_abbreviation_offsets(quote):
    text = f"👋 She said, {quote}Dr. Smith is here. Next"
    assert match_endofsentence(text) == len(f"👋 She said, {quote}Dr. Smith is here.")
    assert match_endofsentence(f"She said, {quote}Dr. S") == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("word_length", [64, 65, 10_000])
async def test_long_lookahead_word_retries_at_delimiter(monkeypatch, word_length):
    from pipecat.utils.text import simple_text_aggregator

    calls = []

    def boundary_when_word_ends(text, **kwargs):
        calls.append(text)
        return len("I.") if text.endswith(" ") else 0

    monkeypatch.setattr(simple_text_aggregator, "match_endofsentence", boundary_when_word_ends)
    aggregator = SimpleTextAggregator()
    word = "x" * word_length
    assert [a.text async for a in aggregator.aggregate("I. " + word)] == []
    assert calls == ["I. x"]
    assert [a.text async for a in aggregator.aggregate(" ")] == ["I."]
    assert calls == ["I. x", "I. " + word + " "]
    assert (await aggregator.flush()).text == word


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ['"', "“", "—", "👋 "])
async def test_non_word_lookahead_retries_only_after_following_word(monkeypatch, prefix):
    from pipecat.utils.text import simple_text_aggregator

    calls = []

    def unresolved(text, **kwargs):
        calls.append(text)
        return 0

    monkeypatch.setattr(simple_text_aggregator, "match_endofsentence", unresolved)
    aggregator = SimpleTextAggregator()
    assert [a.text async for a in aggregator.aggregate("I. " + prefix)] == []
    assert calls == ["I. " + prefix[0]]
    assert [a.text async for a in aggregator.aggregate("Douglas")] == []
    assert len(calls) == 1
    assert [a.text async for a in aggregator.aggregate(" ")] == []
    assert calls == ["I. " + prefix[0], "I. " + prefix + "Douglas "]
    assert (await aggregator.flush()).text == "I. " + prefix + "Douglas"
