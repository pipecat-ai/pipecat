#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the simulation file format and the persona."""

import tempfile
import unittest
from pathlib import Path

from pipecat.evals.persona import END_CALL_FUNCTION, Persona
from pipecat.evals.results import SimulationRunResult
from pipecat.evals.simulation import EvalSimulation, describe_simulation

MINIMAL = """
name: capital_curious
persona: "A curious traveler."
goal: "Learn the capital of Germany."
simulator: {service: openai, model: gpt-4o-mini}
success: "the bot said the capital of Germany is Berlin"
"""


def _write(yaml_text: str) -> Path:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
    f.write(yaml_text)
    f.close()
    return Path(f.name)


class TestSimulationLoader(unittest.TestCase):
    def test_minimal_has_defaults(self):
        s = EvalSimulation.load(_write(MINIMAL))
        self.assertEqual(s.name, "capital_curious")
        self.assertEqual(s.persona, "A curious traveler.")
        self.assertEqual(s.goal, "Learn the capital of Germany.")
        self.assertEqual(s.simulator, {"service": "openai", "model": "gpt-4o-mini"})
        self.assertEqual(s.success, "the bot said the capital of Germany is Berlin")
        self.assertEqual(s.metrics, [])
        self.assertFalse(s.bot_audio)
        self.assertFalse(s.user_audio)
        self.assertEqual(s.max_turns, 20)
        self.assertEqual(s.max_duration_s, 300.0)
        self.assertEqual(s.runs, 1)
        self.assertEqual(s.pass_threshold, 1.0)
        self.assertEqual(s.judge["service"], "ollama")

    def test_metrics_and_caps(self):
        s = EvalSimulation.load(
            _write(
                MINIMAL
                + """
metrics:
  - name: politeness
    criterion: "stayed courteous"
  - {name: accuracy, criterion: "no invented facts", weight: 2}
max_turns: 5
max_duration_s: 42
runs: 3
pass_threshold: 0.8
"""
            )
        )
        self.assertEqual([m.name for m in s.metrics], ["politeness", "accuracy"])
        self.assertEqual([m.weight for m in s.metrics], [1.0, 2.0])
        self.assertEqual(s.max_turns, 5)
        self.assertEqual(s.max_duration_s, 42.0)
        self.assertEqual(s.runs, 3)
        self.assertEqual(s.pass_threshold, 0.8)

    def test_audio_modalities(self):
        s = EvalSimulation.load(
            _write(
                MINIMAL
                + """
user:
  modality: audio
  speech: {service: kokoro, voice: af_heart}
judge:
  modality: audio
  transcription: {service: moonshine}
"""
            )
        )
        self.assertTrue(s.user_audio)
        self.assertEqual(s.user_speech, {"service": "kokoro", "voice": "af_heart"})
        self.assertTrue(s.bot_audio)
        self.assertEqual(s.transcriber, {"service": "moonshine"})

    def test_user_audio_requires_speech(self):
        with self.assertRaises(ValueError) as cm:
            EvalSimulation.load(_write(MINIMAL + "user: {modality: audio}\n"))
        self.assertIn("user.speech", str(cm.exception))

    def test_required_fields(self):
        for missing in ("persona", "goal", "success", "simulator"):
            text = "\n".join(line for line in MINIMAL.splitlines() if not line.startswith(missing))
            with self.assertRaises(ValueError, msg=missing) as cm:
                EvalSimulation.load(_write(text))
            self.assertIn(missing, str(cm.exception))

    def test_metric_needs_a_criterion(self):
        with self.assertRaises(ValueError) as cm:
            EvalSimulation.load(_write(MINIMAL + "metrics: [{name: politeness}]\n"))
        self.assertIn("criterion", str(cm.exception))

    def test_caps_must_be_positive(self):
        with self.assertRaises(ValueError):
            EvalSimulation.load(_write(MINIMAL + "max_turns: 0\n"))
        with self.assertRaises(ValueError):
            EvalSimulation.load(_write(MINIMAL + "max_duration_s: -1\n"))

    def test_describe(self):
        text = describe_simulation(EvalSimulation.load(_write(MINIMAL)))
        self.assertIn("user  -> modality: text", text)
        self.assertIn("judge -> modality: text | eval: ollama/", text)
        self.assertIn("llm   -> service: openai/gpt-4o-mini | max_turns: 20", text)


class TestPersona(unittest.TestCase):
    def test_context_carries_instruction_and_end_call(self):
        persona = Persona("A curious traveler.", "Learn the capital of Germany.")
        context = persona.context()
        messages = context.get_messages()
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("A curious traveler.", messages[0]["content"])
        self.assertIn("Learn the capital of Germany.", messages[0]["content"])
        self.assertIn(END_CALL_FUNCTION, messages[0]["content"])
        tools = context.tools
        assert not isinstance(tools, type(None))
        self.assertEqual([t.name for t in tools.standard_tools], [END_CALL_FUNCTION])  # type: ignore[union-attr]

    def test_each_context_is_fresh(self):
        persona = Persona("x", "y")
        a, b = persona.context(), persona.context()
        a.add_message({"role": "user", "content": "hi"})
        self.assertEqual(len(b.get_messages()), 1)


class TestSimulationRunResult(unittest.TestCase):
    def test_passed_needs_success_and_no_error(self):
        self.assertTrue(SimulationRunResult("s", succeeded=True).passed)
        self.assertFalse(SimulationRunResult("s", succeeded=False).passed)
        self.assertFalse(SimulationRunResult("s", succeeded=True, error="boom").passed)
