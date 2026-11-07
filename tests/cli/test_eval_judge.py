#
# Copyright (c) 2025-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for `pipecat eval judge`, over a judge that answers without an LLM."""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from loguru import logger
from typer.testing import CliRunner

from pipecat.cli.commands.eval import eval_app
from pipecat.evals.judge import JudgeVerdict, RunVerdicts

SCENARIO = """\
name: book
persona: A caller.
goal: Book a table.
success: the bot booked a table
metrics:
  - name: politeness
    criterion: stays polite
    min_score: 1
  - measure: turns
    max_value: 10
scenarios:
  - name: available
  - name: impossible
"""

CONVERSATION = [
    {"role": "user", "content": "A table at six."},
    {"role": "tool_call", "name": "book", "arguments": {"time": "6"}, "cancelled": False},
    {"role": "assistant", "content": "Booked for six."},
]


class _FakeJudge:
    """Passes the goal and every turn, each with a reason, and records how it was built."""

    configs: list[dict] = []

    def __init__(self, config):
        self.configs.append(config)
        self.bot_turns = 0
        self.closed = False

    def add_user_message(self, text):
        pass

    def add_assistant_message(self, text):
        self.bot_turns += 1

    def add_tool_call(self, text):
        pass

    async def evaluate_run(self, criteria, success):
        return RunVerdicts(
            goal=JudgeVerdict(
                verdict="yes",
                reason="booked",
                raw_response="",
                confidence=0.8,
                probabilities={"yes": 0.8, "no": 0.2},
            ),
            turns={
                name: [
                    JudgeVerdict(
                        verdict="yes",
                        reason="polite",
                        raw_response="",
                        confidence=0.9,
                        probabilities={"meets": 0.9, "fails": 0.1},
                    )
                    for _ in range(self.bot_turns)
                ]
                for name in criteria
            },
        )

    async def close(self):
        self.closed = True


class TestEvalJudge(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.scenario = self.base / "book.yaml"
        self.scenario.write_text(SCENARIO)
        self.conversation = self.base / "run.conversation.json"
        self.conversation.write_text(json.dumps(CONVERSATION))
        self.output = self.base / "judged.jsonl"
        _FakeJudge.configs = []

    def tearDown(self):
        self._tmp.cleanup()
        # The command drops every log sink; put loguru's default back.
        logger.remove()
        logger.add(sys.stderr)

    def _judge(self, *args: str):
        with patch("pipecat.evals.simulation_judge.EvalJudge.from_config", _FakeJudge):
            result = CliRunner().invoke(eval_app, ["judge", *args, "-o", str(self.output)])
        records = (
            [json.loads(line) for line in self.output.read_text().splitlines()]
            if self.output.exists()
            else []
        )
        return result, records

    def test_a_conversation_is_judged_with_every_verdict_explained(self):
        result, records = self._judge(str(self.scenario), "-s", "available", str(self.conversation))
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(_FakeJudge.configs[0]["explain_below"], math.inf)
        [record] = records
        self.assertEqual(record["conversation"], str(self.conversation))
        self.assertEqual(record["scenario"], "book/available")
        self.assertEqual(record["success"], "the bot booked a table")
        self.assertTrue(record["passed"])
        self.assertEqual(record["reason"], "booked")
        self.assertEqual(record["confidence"], 0.8)
        self.assertEqual(record["probabilities"], {"yes": 0.8, "no": 0.2})
        # The measured metric is left out: a conversation has nothing to measure.
        [politeness] = record["metrics"]
        self.assertEqual(politeness["name"], "politeness")
        self.assertEqual(politeness["criterion"], "stays polite")
        self.assertEqual(
            politeness["verdicts"],
            [
                {
                    "turn": 1,
                    "passed": True,
                    "verdict": "yes",
                    "confidence": 0.9,
                    "probabilities": {"meets": 0.9, "fails": 0.1},
                    "reason": "polite",
                    "text": "Booked for six.",
                }
            ],
        )

    def test_a_file_that_cannot_be_read_is_its_own_error_and_fails_the_command(self):
        broken = self.base / "broken.conversation.json"
        broken.write_text("not json")
        result, records = self._judge(
            str(self.scenario), "-s", "book/available", str(broken), str(self.conversation)
        )
        self.assertEqual(result.exit_code, 1, result.output)
        by_file = {Path(r["conversation"]).name: r for r in records}
        self.assertTrue(by_file["run.conversation.json"]["passed"])
        self.assertFalse(by_file["broken.conversation.json"]["passed"])
        self.assertIn("JSONDecodeError", by_file["broken.conversation.json"]["error"])

    def test_a_file_with_several_simulations_needs_one_picked(self):
        for args in ([], ["-s", "nope"]):
            with self.subTest(args=args):
                result, records = self._judge(str(self.scenario), *args, str(self.conversation))
                self.assertEqual(result.exit_code, 2)
                self.assertIn("book/impossible", result.output)
                self.assertEqual(records, [])


if __name__ == "__main__":
    unittest.main()
