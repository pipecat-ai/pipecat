#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import tempfile
import unittest
from pathlib import Path

from pipecat.evals.scenario import (
    Expectation,
    SendAfter,
    Turn,
    load_scenario,
)


def _write(yaml_text: str) -> Path:
    """Write yaml_text to a temp file and return the path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
    f.write(yaml_text)
    f.close()
    return Path(f.name)


class TestEvalsScenarioParser(unittest.TestCase):
    def test_minimal_valid(self):
        s = load_scenario(
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

    def test_all_expectation_fields(self):
        s = load_scenario(
            _write(
                """
                name: all_fields
                turns:
                  - user: "x"
                    expect:
                      - event: llm_response
                        within_ms: 500
                        transcript_contains: "foo"
                        text_contains: "bar"
                        name: my_tool
                        args: {a: 1}
                        judge: "is friendly"
                """
            )
        )
        exp = s.turns[0].expect[0]
        self.assertEqual(exp.within_ms, 500)
        self.assertEqual(exp.transcript_contains, "foo")
        self.assertEqual(exp.text_contains, "bar")
        self.assertEqual(exp.name, "my_tool")
        self.assertEqual(exp.args, {"a": 1})
        self.assertEqual(exp.judge, "is friendly")

    def test_send_after_parsed(self):
        s = load_scenario(
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
        self.assertIsInstance(s.turns[1].send_after, SendAfter)
        assert s.turns[1].send_after is not None  # for the type checker
        self.assertEqual(s.turns[1].send_after.event, "user_stopped_speaking")
        self.assertEqual(s.turns[1].send_after.delay_ms, 200)

    def test_expect_only_turn(self):
        """A turn without `user:` is observation-only (bot-first scenarios)."""
        s = load_scenario(
            _write(
                """
                name: bot_first
                turns:
                  - expect:
                      - event: bot_stopped_speaking
                """
            )
        )
        self.assertIsNone(s.turns[0].user)
        self.assertEqual(s.turns[0].expect[0].event, "bot_stopped_speaking")

    def test_judge_fields_and_fixtures_preserved(self):
        s = load_scenario(
            _write(
                """
                name: with_judge
                judge_service: openai
                judge_model: gpt-4o-mini
                judge_endpoint: http://custom-endpoint
                fixtures:
                  bot_url: ws://localhost:9000
                turns:
                  - user: "hi"
                    expect: [{event: user_stopped_speaking}]
                """
            )
        )
        self.assertEqual(s.judge_service, "openai")
        self.assertEqual(s.judge_model, "gpt-4o-mini")
        self.assertEqual(s.judge_endpoint, "http://custom-endpoint")
        self.assertEqual(s.fixtures, {"bot_url": "ws://localhost:9000"})

    def test_judge_fields_default_to_ollama(self):
        s = load_scenario(
            _write("name: e\nturns: [{user: hi, expect: [{event: user_stopped_speaking}]}]\n")
        )
        self.assertEqual(s.judge_service, "ollama")
        self.assertEqual(s.judge_model, "qwen2.5:3b")
        self.assertIsNone(s.judge_endpoint)

    def test_missing_name_field(self):
        with self.assertRaises(ValueError) as cm:
            load_scenario(_write("turns: []\n"))
        self.assertIn("'name:'", str(cm.exception))

    def test_send_after_without_user_rejected(self):
        with self.assertRaises(ValueError) as cm:
            load_scenario(
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
            load_scenario(
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
            load_scenario(_write("name: bad\nturns: [{user: hi}]\n"))
        self.assertIn("expect", str(cm.exception))

    def test_expectation_missing_event_rejected(self):
        with self.assertRaises(ValueError) as cm:
            load_scenario(_write("name: bad\nturns: [{user: hi, expect: [{within_ms: 100}]}]\n"))
        self.assertIn("event", str(cm.exception))

    def test_expectation_dataclass_defaults(self):
        """Construct Expectation directly to lock its defaults."""
        e = Expectation(event="foo")
        self.assertIsNone(e.within_ms)
        self.assertIsNone(e.transcript_contains)
        self.assertIsNone(e.judge)

    def test_judge_on_non_bot_event_warns(self):
        """judge: on user-side events should produce a parser warning
        (the user transcript is deterministic, so judging it adds cost without
        signal). We can't easily assert on loguru output, but the parse
        should succeed and the field should be preserved."""
        s = load_scenario(
            _write(
                """
                name: misused_judge
                turns:
                  - user: "hi"
                    expect:
                      - event: user_stopped_speaking
                        judge: "is a greeting"
                """
            )
        )
        self.assertEqual(s.turns[0].expect[0].judge, "is a greeting")

    def test_turn_dataclass_construction(self):
        """Direct construction (used by tests / programmatic eval generation)."""
        t = Turn(user="hi", expect=[Expectation(event="x")])
        self.assertEqual(t.user, "hi")
        self.assertIsNone(t.send_after)

    def test_reset_defaults_to_empty(self):
        s = load_scenario(
            _write("name: e\nturns: [{user: hi, expect: [{event: user_stopped_speaking}]}]\n")
        )
        self.assertEqual(s.reset, [])

    def test_reset_parsed_as_list(self):
        s = load_scenario(
            _write(
                """
                name: with_reset
                reset:
                  - role: system
                    content: "You are a helpful assistant."
                turns:
                  - user: hi
                    expect: [{event: user_stopped_speaking}]
                """
            )
        )
        self.assertEqual(len(s.reset), 1)
        self.assertEqual(s.reset[0]["role"], "system")
        self.assertEqual(s.reset[0]["content"], "You are a helpful assistant.")

    def test_reset_non_list_rejected(self):
        with self.assertRaises(ValueError) as cm:
            load_scenario(
                _write("name: bad\nreset: not_a_list\nturns: [{user: hi, expect: [{event: x}]}]\n")
            )
        self.assertIn("'reset:'", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
