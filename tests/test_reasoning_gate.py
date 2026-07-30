"""ReasoningTagGate must keep leaked model reasoning out of the visible stream.

Run 2002 (wf15): Gemma-4 emitted a PIPE-LESS ``<thought`` pseudo-tag followed by
an English chain-of-thought into ``delta.content``. The server-side reasoning
parser keys on the ``<|thought|>`` special token and the downstream sanitizers
are stateless per-sentence, so the CoT streamed straight to TTS and the caller
heard ~46 seconds of "Analyze the user input…" read aloud. The gate is the
stateful choke point in the LLM stream loop: only text outside a reasoning
block may come back from ``feed``/``flush``.

Pinned behaviors:
- normal text streams through immediately (no added latency);
- every tag variant gates: piped/pipe-less, missing ``>``, think/thinking, any
  casing, tags split across chunks;
- a closed block drops the reasoning and releases the reply after it;
- an unclosed block with a CoT signature ("Thinking Process…") suppresses to
  end of stream (the caller then treats the completion as empty and retries);
- an unclosed lone opener followed by a direct answer (real Gemma-4
  no-thinking-mode artifact) releases the answer at flush — a gate that ate
  those replies would kill every healthy turn.
"""

import pytest

from pipecat.services.openai.base_llm import ReasoningTagGate


def run_stream(chunks):
    """Feed chunks through a fresh gate; return (visible, suppressed_chars)."""
    gate = ReasoningTagGate()
    out = "".join(gate.feed(c) for c in chunks)
    out += gate.flush()
    assert out == gate.visible_text
    return out, gate.suppressed_chars


class TestPassThrough:
    def test_plain_hebrew_streams_unchanged(self):
        out, suppressed = run_stream(["שלום, ", "מה שלומך היום?"])
        assert out == "שלום, מה שלומך היום?"
        assert suppressed == 0

    def test_no_holding_latency_on_normal_text(self):
        gate = ReasoningTagGate()
        assert gate.feed("היי, מדבר עידן ") == "היי, מדבר עידן "

    def test_legit_angle_bracket_untouched(self):
        out, suppressed = run_stream(["ציון 5 <-- ", "טוב מאוד"])
        assert out == "ציון 5 <-- טוב מאוד"
        assert suppressed == 0

    def test_partial_tag_at_stream_end_is_released(self):
        out, _ = run_stream(["נתראה <thou"])
        assert out == "נתראה <thou"


class TestClosedBlocks:
    def test_run_2002_pipeless_block_then_reply(self):
        out, suppressed = run_stream(
            [
                "<thought\nThinking Process:\n\n1.  **Analyze",
                " the user input:** The user confirmed identity.\n</thought>\n",
                "מעולה. מתכוון להצביע בבחירות הקרובות?",
            ]
        )
        assert out.strip() == "מעולה. מתכוון להצביע בבחירות הקרובות?"
        assert suppressed > 0

    def test_piped_block_split_across_chunks(self):
        out, _ = run_stream(["<|thou", "ght|>secret reasoning<|/thou", "ght|>הבנתי, תודה"])
        assert out == "הבנתי, תודה"

    def test_think_variant(self):
        out, _ = run_stream(["<think>should ask about the party</think>", "למי אתה מתכוון להצביע?"])
        assert out == "למי אתה מתכוון להצביע?"

    def test_mixed_case_thinking_variant(self):
        out, _ = run_stream(["<Thinking>plan</Thinking>", "בסדר גמור."])
        assert out == "בסדר גמור."

    def test_midstream_block_after_real_text(self):
        out, _ = run_stream(["בסדר גמור. ", "<thought>\nwhat next?</thought>", " יום טוב!"])
        assert out == "בסדר גמור.  יום טוב!"

    def test_whitespace_before_opener(self):
        out, _ = run_stream(["  \n<think>x</think>", "שלום"])
        assert out == "שלום"


class TestUnclosedBlocks:
    def test_run_to_limit_cot_fully_suppressed(self):
        out, suppressed = run_stream(
            [
                "<thought\nThinking Process:\n\n1. **Analyze the user",
                " input** and keep going without ever closing",
            ]
        )
        assert out == ""
        assert suppressed > 0

    def test_markdown_wrapped_cot_signature_suppressed(self):
        out, _ = run_stream(["<|thought|>\n**Thinking Process:** step one"])
        assert out == ""

    def test_lone_opener_with_direct_reply_releases_reply(self):
        # Gemma-4 in no-thinking mode emits a lone opener then answers directly.
        out, _ = run_stream(["<|thought|>\nזה בהחלט מובן. מתכוון להצביע?"])
        assert out.strip() == "זה בהחלט מובן. מתכוון להצביע?"

    def test_lone_pipeless_opener_with_direct_reply(self):
        out, _ = run_stream(["<thinking>\nהבנתי, תודה רבה. מה הסיבות?"])
        assert out.strip() == "הבנתי, תודה רבה. מה הסיבות?"


class TestChunkBoundaryResidue:
    """Per-token streaming splits tags at arbitrary positions; no split may
    change the visible output (review of the initial gate found ">"/"ing>"
    residue leaking whenever a closer or opener split at the buffer edge)."""

    def test_per_token_closed_block_leaks_nothing(self):
        # The exact per-token shape vLLM streams for a textual pseudo-tag.
        out, _ = run_stream(
            [
                "<",
                "thought",
                "\n",
                "Thinking",
                " Process",
                ":",
                " analyze",
                "\n",
                "</",
                "thought",
                ">",
                "\n",
                "Great",
                ", will you vote?",
            ]
        )
        assert out == "\nGreat, will you vote?"

    def test_think_thinking_prefix_collision_on_closer(self):
        # "</think" must not half-match while "</thinking>" is still arriving.
        out, _ = run_stream(["<thinking>plan the reply</think", "ing>very well."])
        assert out == "very well."

    def test_piped_closer_split_before_pipe(self):
        out, _ = run_stream(["<|thought|>secret<|/thought", "|>הבנתי."])
        assert out == "הבנתי."

    def test_open_tag_split_before_gt_lone_opener_reply(self):
        # Opener "<thought" + ">" split: the ">" must not prefix the reply.
        out, _ = run_stream(["<thought", ">\nשלום, מה שלומך?"])
        assert out.strip() == "שלום, מה שלומך?"

    def test_piped_opener_split_before_pipe_gt(self):
        out, _ = run_stream(["<|thought", "|>\nזה מובן. מתכוון להצביע?"])
        assert out.strip() == "זה מובן. מתכוון להצביע?"

    def test_thinking_opener_split_after_think(self):
        out, _ = run_stream(["<think", "ing>\nהבנתי, תודה. מה הסיבות?"])
        assert out.strip() == "הבנתי, תודה. מה הסיבות?"

    def test_per_token_unclosed_cot_fully_suppressed(self):
        out, _ = run_stream(["<", "thought", ">", "\nThinking Process: analyze the input. Step 1"])
        assert out == ""

    def test_legit_tag_prefix_word_survives_any_split(self):
        # "<thoughts" is prose, not a tag — no split may eat characters.
        for cut in range(len("I have <thoughts about it")):
            s = "I have <thoughts about it"
            out, suppressed = run_stream([s[:cut], s[cut:]])
            assert out == s, (cut, out)
            assert suppressed == 0

    def test_gtless_opener_glued_to_cot_uppercase(self):
        # ">"-less opener glued straight onto the CoT preamble: IGNORECASE must
        # not fold the lowercase-lookahead (a real bug caught by fuzzing) —
        # "<thoughtThinking Process:" is a tag + CoT, not prose.
        out, suppressed = run_stream(["<thought", "Thinking Process: secret analysis"])
        assert out == "" and suppressed > 0
        out, _ = run_stream(["<thoughtThinking Process: secret</thought>שלום"])
        assert out == "שלום"

    def test_prose_tag_prefix_words_stay_prose(self):
        for s in ["we have <thoughts on this", "I thought about it.", "a <thoughtful answer"]:
            out, suppressed = run_stream([s])
            assert out == s and suppressed == 0, s

    def test_closer_mention_inside_cot_does_not_unhold(self):
        # CoT prose that MENTIONS a ">"-less closer must not end the hold and
        # stream the rest of the reasoning.
        out, _ = run_stream(
            [
                "<thought\nThinking Process: if we emit </think here the rest",
                " leaks\nmore secret reasoning",
            ]
        )
        assert out == ""

    def test_split_invariance_fuzz(self):
        # Every 2-way split of every corpus stream must equal the unsplit run.
        corpus = [
            "שלום, מה שלומך היום? הכול טוב.",
            "<thought\nThinking Process:\n1. **Analyze** it\n</thought>\nמעולה, רשמתי.",
            "<|thought|>secret<|/thought|>הבנתי, תודה.",
            "<think>plan</think>למי אתה מתכוון להצביע?",
            "<thinking>plan the reply</thinking>very well.",
            "<|thought|>\nזה בהחלט מובן. מתכוון להצביע?",
            "בסדר גמור. <thought>hmm</thought> יום טוב!",
            "ציון 5 <-- טוב מאוד, נתראה <thou",
            "<thought\nThinking Process: run to the limit without closing",
        ]
        for s in corpus:
            reference, _ = run_stream([s])
            for cut in range(1, len(s)):
                out, _ = run_stream([s[:cut], s[cut:]])
                assert out == reference, (s, cut, out, reference)


class TestWholeStringMode:
    """run_inference passes the complete completion through one feed+flush."""

    def test_thought_containing_json_does_not_poison_extraction(self):
        # The CoT contains a JSON *example*; only the real JSON after the block
        # may survive, or brace-matching parsers extract the wrong object.
        out, _ = run_stream(
            [
                '<thought>\nThe schema example is {"party_name": "הליכוד"} but '
                'the user refused.</thought>\n{"party_name": "סירב"}'
            ]
        )
        assert out.strip() == '{"party_name": "סירב"}'

    def test_plain_json_untouched(self):
        out, suppressed = run_stream(['{"intends_to_vote": "כן"}'])
        assert out == '{"intends_to_vote": "כן"}'
        assert suppressed == 0


class TestAccounting:
    def test_suppressed_preview_captures_block_start(self):
        gate = ReasoningTagGate()
        gate.feed("<thought\nThinking Process: analyze everything at length")
        gate.flush()
        assert gate.suppressed_chars > 0
        assert "Thinking Process" in gate.suppressed_preview

    def test_empty_feed_is_noop(self):
        gate = ReasoningTagGate()
        assert gate.feed("") == ""
        assert gate.visible_text == ""


class TestBoundedHold:
    """`hold` must not batch an entire reply behind a lone opener.

    Gemma-4's no-thinking-mode artifact is an opener with no closer followed by
    the real answer. Waiting for `flush()` to release that means downstream TTS
    receives nothing until the generation finishes, so time-to-first-audio
    becomes full generation time rather than first-token time — on the model
    that carried 100% of production traffic.
    """

    def test_long_reply_behind_a_lone_opener_starts_streaming_early(self):
        reply = "".join(f"sentence {i} of the answer. " for i in range(40))
        assert len(reply) > ReasoningTagGate.REASONING_HOLD_MAX_CHARS

        gate = ReasoningTagGate()
        gate.feed("<thought>")
        deltas = [reply[i : i + 20] for i in range(0, len(reply), 20)]
        emitted = [gate.feed(d) for d in deltas]

        # Emission began before the last delta arrived...
        assert any(chunk for chunk in emitted[:-1]), "hold released nothing mid-stream"
        # ...and nothing was lost or duplicated.
        assert "".join(emitted) + gate.flush() == reply
        assert gate.bounded_releases == 1

    def test_long_chain_of_thought_still_held_and_suppressed(self):
        cot = "Thinking Process:\n\n" + "".join(
            f"{i}. analyze the user input carefully. " for i in range(40)
        )
        assert len(cot) > ReasoningTagGate.REASONING_HOLD_MAX_CHARS

        gate = ReasoningTagGate()
        gate.feed("<thought>")
        emitted = [gate.feed(cot[i : i + 20]) for i in range(0, len(cot), 20)]

        assert "".join(emitted) == ""
        assert gate.flush() == ""
        assert gate.bounded_releases == 0
        assert gate.suppressed_chars > 0

    def test_short_reply_behind_a_lone_opener_is_unchanged(self):
        # Under the bound: still released at flush, exactly as before.
        gate = ReasoningTagGate()
        assert gate.feed("<thought>שלום, איך אפשר לעזור?") == ""
        assert gate.flush().strip() == "שלום, איך אפשר לעזור?"
        assert gate.bounded_releases == 0

    def test_closer_arriving_after_a_bounded_release_is_not_spoken(self):
        reply = "".join(f"word{i} " for i in range(80))
        assert len(reply) > ReasoningTagGate.REASONING_HOLD_MAX_CHARS

        gate = ReasoningTagGate()
        gate.feed("<thought>")
        out = "".join(gate.feed(reply[i : i + 20]) for i in range(0, len(reply), 20))
        # The closer finally turns up — it belongs to the released block.
        out += "".join(gate.feed(c) for c in ["</thou", "ght>", " tail."])
        out += gate.flush()

        assert "</thought>" not in out
        assert "</thou" not in out
        assert out == reply + " tail."
