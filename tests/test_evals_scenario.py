#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import tempfile
import unittest
from pathlib import Path

from pipecat.evals.scenario import (
    EvalExpectation,
    EvalFunctionCall,
    EvalScenario,
    EvalSendAfter,
    EvalTurn,
)


def _write(yaml_text: str) -> Path:
    """Write yaml_text to a temp file and return the path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
    f.write(yaml_text)
    f.close()
    return Path(f.name)


class TestEvalsScenarioParser(unittest.TestCase):
    def test_minimal_valid(self):
        s = EvalScenario.load(
            _write(
                """
                name: minimal
                turns:
                  - user: "hello"
                    expect:
                      - event: user_started_speaking
                """
            )
        )
        self.assertEqual(s.name, "minimal")
        self.assertEqual(len(s.turns), 1)
        self.assertEqual(s.turns[0].user, "hello")
        self.assertEqual(s.turns[0].expect[0].event, "user_started_speaking")
        self.assertIsNone(s.turns[0].send_after)
        # bot_audio defaults to False: evals are text/silent unless they opt in.
        self.assertFalse(s.bot_audio)
        self.assertIsNone(s.transcriber)

    def test_judge_audio_modality_enables_transcriber(self):
        s = EvalScenario.load(
            _write(
                "name: a\n"
                "judge:\n"
                "  modality: audio\n"
                "  transcription: {service: whisper, model: base}\n"
                "turns: [{user: hi, expect: [{event: tts_response, eval: ok}]}]\n"
            )
        )
        self.assertTrue(s.bot_audio)
        self.assertEqual(s.transcriber, {"service": "whisper", "model": "base"})

    def test_judge_audio_requires_transcription(self):
        with self.assertRaises(ValueError) as cm:
            EvalScenario.load(
                _write(
                    "name: a\njudge: {modality: audio}\n"
                    "turns: [{user: hi, expect: [{event: tts_response, eval: ok}]}]\n"
                )
            )
        self.assertIn("transcription", str(cm.exception))

    def test_judge_invalid_modality_rejected(self):
        with self.assertRaises(ValueError):
            EvalScenario.load(
                _write(
                    "name: a\njudge: {modality: bogus}\n"
                    "turns: [{user: hi, expect: [{event: llm_started}]}]\n"
                )
            )

    def test_user_audio_modality(self):
        s = EvalScenario.load(
            _write(
                "name: a\n"
                "user:\n"
                "  modality: audio\n"
                "  speech: {service: cartesia, voice: v1}\n"
                "turns: [{user: hi, expect: [{event: llm_started}]}]\n"
            )
        )
        self.assertEqual(s.user_audio, {"service": "cartesia", "voice": "v1"})

    def test_user_audio_requires_speech(self):
        with self.assertRaises(ValueError) as cm:
            EvalScenario.load(
                _write(
                    "name: a\nuser: {modality: audio}\n"
                    "turns: [{user: hi, expect: [{event: llm_started}]}]\n"
                )
            )
        self.assertIn("speech", str(cm.exception))

    def test_response_event_resolves_to_modality(self):
        # judge.modality audio -> response stays response (the audio transcription)
        audio = EvalScenario.load(
            _write(
                "name: a\n"
                "judge: {modality: audio, transcription: {service: whisper}}\n"
                "turns: [{user: hi, expect: [{event: response, eval: ok}]}]\n"
            )
        )
        self.assertEqual(audio.turns[0].expect[0].event, "response")
        # text (default) -> response falls back to llm_response (no audio)
        text = EvalScenario.load(
            _write("name: a\nturns: [{user: hi, expect: [{event: response, eval: ok}]}]\n")
        )
        self.assertEqual(text.turns[0].expect[0].event, "llm_response")

    def test_tts_response_in_text_modality_rejected(self):
        with self.assertRaises(ValueError) as cm:
            EvalScenario.load(
                _write("name: a\nturns: [{user: hi, expect: [{event: tts_response, eval: ok}]}]\n")
            )
        self.assertIn("tts_response", str(cm.exception))

    def test_all_expectation_fields(self):
        s = EvalScenario.load(
            _write(
                """
                name: all_fields
                turns:
                  - user: "x"
                    expect:
                      - event: llm_response
                        within_ms: 500
                        text_contains: "bar"
                        eval: "is friendly"
                """
            )
        )
        exp = s.turns[0].expect[0]
        self.assertEqual(exp.within_ms, 500)
        self.assertEqual(exp.text_contains, "bar")
        self.assertEqual(exp.eval, "is friendly")
        self.assertIsNone(exp.calls)

    def test_function_call_name_args_shorthand(self):
        """A single function_call uses the ``name:``/``args:`` shorthand."""
        s = EvalScenario.load(
            _write(
                """
                name: one_call
                turns:
                  - user: "weather?"
                    expect:
                      - event: function_call
                        name: get_weather
                        args: {city: Paris}
                """
            )
        )
        self.assertEqual(
            s.turns[0].expect[0].calls,
            [EvalFunctionCall(name="get_weather", args={"city": "Paris"})],
        )

    def test_function_call_calls_list_any_order(self):
        """Multiple calls in a turn go under ``calls:`` (matched in any order)."""
        s = EvalScenario.load(
            _write(
                """
                name: two_calls
                turns:
                  - user: "weather and food?"
                    expect:
                      - event: function_call
                        calls:
                          - get_weather
                          - {name: get_restaurants, args: {city: Paris}}
                """
            )
        )
        self.assertEqual(
            s.turns[0].expect[0].calls,
            [
                EvalFunctionCall(name="get_weather"),
                EvalFunctionCall(name="get_restaurants", args={"city": "Paris"}),
            ],
        )

    def test_bare_function_call_matches_any(self):
        """A bare function_call (no name/calls) matches any single call."""
        s = EvalScenario.load(
            _write("name: bare\nturns: [{user: hi, expect: [{event: function_call}]}]\n")
        )
        self.assertEqual(s.turns[0].expect[0].calls, [EvalFunctionCall(name=None)])

    def test_send_after_parsed(self):
        s = EvalScenario.load(
            _write(
                """
                name: with_send_after
                turns:
                  - user: "first"
                    expect: [{event: user_stopped_speaking}]
                  - send_after: {event: user_stopped_speaking, delay_ms: 200}
                    user: "second"
                    expect: [{event: user_stopped_speaking}]
                """
            )
        )
        self.assertIsNone(s.turns[0].send_after)
        self.assertIsInstance(s.turns[1].send_after, EvalSendAfter)
        assert s.turns[1].send_after is not None  # for the type checker
        self.assertEqual(s.turns[1].send_after.event, "user_stopped_speaking")
        self.assertEqual(s.turns[1].send_after.delay_ms, 200)

    def test_expect_only_turn(self):
        """A turn without `user:` is observation-only (bot-first scenarios)."""
        s = EvalScenario.load(
            _write(
                """
                name: bot_first
                turns:
                  - expect:
                      - event: llm_response
                """
            )
        )
        self.assertIsNone(s.turns[0].user)
        self.assertEqual(s.turns[0].expect[0].event, "llm_response")

    def test_judge_eval_preserved(self):
        s = EvalScenario.load(
            _write(
                """
                name: with_judge
                judge:
                  eval:
                    service: openai
                    model: gpt-4o-mini
                    endpoint: http://custom-endpoint
                turns:
                  - user: "hi"
                    expect: [{event: user_stopped_speaking}]
                """
            )
        )
        self.assertEqual(
            s.judge,
            {"service": "openai", "model": "gpt-4o-mini", "endpoint": "http://custom-endpoint"},
        )

    def test_judge_block_defaults_to_ollama(self):
        s = EvalScenario.load(
            _write("name: e\nturns: [{user: hi, expect: [{event: user_stopped_speaking}]}]\n")
        )
        self.assertEqual(s.judge, {"service": "ollama", "model": "gemma2:9b"})

    def test_judge_block_non_mapping_rejected(self):
        with self.assertRaises(ValueError) as cm:
            EvalScenario.load(
                _write(
                    "name: bad\njudge: not_a_mapping\nturns: [{user: hi, expect: [{event: x}]}]\n"
                )
            )
        self.assertIn("'judge:'", str(cm.exception))

    def test_missing_name_field(self):
        with self.assertRaises(ValueError) as cm:
            EvalScenario.load(_write("turns: []\n"))
        self.assertIn("'name:'", str(cm.exception))

    def test_send_after_without_user_rejected(self):
        with self.assertRaises(ValueError) as cm:
            EvalScenario.load(
                _write(
                    """
                    name: bad
                    turns:
                      - send_after: {event: x, delay_ms: 100}
                        expect: [{event: y}]
                    """
                )
            )
        self.assertIn("send_after", str(cm.exception))
        self.assertIn("no 'user:'", str(cm.exception))

    def test_invalid_send_after_delay_rejected(self):
        with self.assertRaises(ValueError) as cm:
            EvalScenario.load(
                _write(
                    """
                    name: bad
                    turns:
                      - send_after: {event: x, delay_ms: -1}
                        user: "y"
                        expect: [{event: z}]
                    """
                )
            )
        self.assertIn("non-negative", str(cm.exception))

    def test_missing_expect_rejected(self):
        with self.assertRaises(ValueError) as cm:
            EvalScenario.load(_write("name: bad\nturns: [{user: hi}]\n"))
        self.assertIn("expect", str(cm.exception))

    def test_expectation_missing_event_rejected(self):
        with self.assertRaises(ValueError) as cm:
            EvalScenario.load(
                _write("name: bad\nturns: [{user: hi, expect: [{within_ms: 100}]}]\n")
            )
        self.assertIn("event", str(cm.exception))

    def test_expectation_dataclass_defaults(self):
        """Construct EvalExpectation directly to lock its defaults."""
        e = EvalExpectation(event="foo")
        self.assertIsNone(e.within_ms)
        self.assertIsNone(e.text_contains)
        self.assertIsNone(e.eval)

    def test_eval_on_non_bot_event_warns(self):
        """eval: on user-side events should produce a parser warning
        (the user transcript is deterministic, so judging it adds cost without
        signal). We can't easily assert on loguru output, but the parse
        should succeed and the field should be preserved."""
        s = EvalScenario.load(
            _write(
                """
                name: misused_eval
                turns:
                  - user: "hi"
                    expect:
                      - event: user_stopped_speaking
                        eval: "is a greeting"
                """
            )
        )
        self.assertEqual(s.turns[0].expect[0].eval, "is a greeting")

    def test_turn_dataclass_construction(self):
        """Direct construction (used by tests / programmatic eval generation)."""
        t = EvalTurn(user="hi", expect=[EvalExpectation(event="x")])
        self.assertEqual(t.user, "hi")
        self.assertIsNone(t.send_after)
        self.assertIsNone(t.image)

    def test_turn_image_resolved_relative_to_scenario(self):
        p = _write(
            "name: t\nturns: [{user: hi, image: pics/cat.jpg, expect: [{event: llm_response}]}]\n"
        )
        s = EvalScenario.load(p)
        self.assertEqual(s.turns[0].image, str((p.parent / "pics/cat.jpg").resolve()))

    def test_turn_image_non_string_rejected(self):
        with self.assertRaises(ValueError) as cm:
            EvalScenario.load(
                _write("name: t\nturns: [{user: hi, image: 5, expect: [{event: x}]}]\n")
            )
        self.assertIn("image", str(cm.exception))

    def test_context_defaults_to_empty(self):
        s = EvalScenario.load(
            _write("name: e\nturns: [{user: hi, expect: [{event: user_stopped_speaking}]}]\n")
        )
        self.assertEqual(s.context, [])

    def test_context_parsed_as_list(self):
        s = EvalScenario.load(
            _write(
                """
                name: with_context
                context:
                  - role: system
                    content: "You are a helpful assistant."
                turns:
                  - user: hi
                    expect: [{event: user_stopped_speaking}]
                """
            )
        )
        self.assertEqual(len(s.context), 1)
        self.assertEqual(s.context[0]["role"], "system")
        self.assertEqual(s.context[0]["content"], "You are a helpful assistant.")

    def test_context_non_list_rejected(self):
        with self.assertRaises(ValueError) as cm:
            EvalScenario.load(
                _write(
                    "name: bad\ncontext: not_a_list\nturns: [{user: hi, expect: [{event: x}]}]\n"
                )
            )
        self.assertIn("'context:'", str(cm.exception))

    def test_include_resolves_judge_and_user_blocks(self):
        # `judge: !include ...` / `user: !include ...` let scenarios share config.
        # Includes resolve relative to the scenario file's directory, so the
        # fragments are written alongside the scenario.
        d = Path(tempfile.mkdtemp())
        (d / "judge_audio.yaml").write_text(
            "modality: audio\n"
            "eval: {service: ollama, model: llama3:latest}\n"
            "transcription: {service: whisper, model: base}\n",
            encoding="utf-8",
        )
        (d / "user_audio.yaml").write_text(
            "modality: audio\nspeech: {service: kokoro, voice: af_heart}\n", encoding="utf-8"
        )
        scenario = d / "math.yaml"
        scenario.write_text(
            "name: math\n"
            "user: !include user_audio.yaml\n"
            "judge: !include judge_audio.yaml\n"
            "turns: [{user: hi, expect: [{event: response, eval: ok}]}]\n",
            encoding="utf-8",
        )

        s = EvalScenario.load(scenario)
        self.assertTrue(s.bot_audio)
        self.assertEqual(s.transcriber, {"service": "whisper", "model": "base"})
        self.assertEqual(s.judge, {"service": "ollama", "model": "llama3:latest"})
        self.assertEqual(s.user_audio, {"service": "kokoro", "voice": "af_heart"})


if __name__ == "__main__":
    unittest.main()
