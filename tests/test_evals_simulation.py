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
from pipecat.evals.scenario import EvalScenario
from pipecat.evals.simulation import EvalSimulation, describe_simulation, load_scenario_file

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
        self.assertIn(
            "user  -> modality: text | persona: openai/gpt-4o-mini | max_turns: 20 | "
            "max_duration_s: 300",
            text,
        )
        self.assertIn("judge -> modality: text | eval: ollama/", text)
        self.assertIn("goal  -> Learn the capital of Germany.", text)
        self.assertEqual(len(text.splitlines()), 3)


class TestLoadScenarioFile(unittest.TestCase):
    def test_a_persona_makes_a_simulation(self):
        self.assertIsInstance(load_scenario_file(_write(MINIMAL)), EvalSimulation)

    def test_turns_make_a_scripted_scenario(self):
        self.assertIsInstance(load_scenario_file(_write("name: greet\nturns: []\n")), EvalScenario)

    def test_a_file_is_one_kind_or_the_other(self):
        with self.assertRaises(ValueError) as cm:
            load_scenario_file(_write("name: nothing\n"))
        self.assertIn("'turns:'", str(cm.exception))
        self.assertIn("'persona:'", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            load_scenario_file(_write(MINIMAL + "turns: []\n"))
        self.assertIn("not both", str(cm.exception))


class TestPersona(unittest.TestCase):
    def test_instruction_and_context(self):
        persona = Persona("A curious traveler.", "Learn the capital of Germany.")
        self.assertIn("A curious traveler.", persona.instruction)
        self.assertIn("Learn the capital of Germany.", persona.instruction)
        self.assertIn(END_CALL_FUNCTION, persona.instruction)
        context = persona.context()
        self.assertEqual(context.get_messages(), [])
        tools = context.tools
        assert not isinstance(tools, type(None))
        self.assertEqual([t.name for t in tools.standard_tools], [END_CALL_FUNCTION])  # type: ignore[union-attr]

    def test_each_context_is_fresh(self):
        persona = Persona("x", "y")
        a, b = persona.context(), persona.context()
        a.add_message({"role": "user", "content": "hi"})
        self.assertEqual(b.get_messages(), [])


class TestSimulationRunResult(unittest.TestCase):
    def test_passed_needs_success_and_no_error(self):
        self.assertTrue(SimulationRunResult("s", succeeded=True).passed)
        self.assertFalse(SimulationRunResult("s", succeeded=False).passed)
        self.assertFalse(SimulationRunResult("s", succeeded=True, error="boom").passed)


# ---------------------------------------------------------------------------
# The simulation driver, over fakes.
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from pipecat.evals.client import BOT_ENDED_EVENT, PERSONA_TURN_EVENT  # noqa: E402
from pipecat.evals.events import EvalEventStream  # noqa: E402
from pipecat.evals.judge import JudgeVerdict  # noqa: E402
from pipecat.evals.results import EvalAssertionFailure, EvalTrace  # noqa: E402
from pipecat.evals.simulation import EvalSimulationMetric  # noqa: E402
from pipecat.evals.simulation_driver import END_CALL_EVENT, SimulationDriver  # noqa: E402
from pipecat.processors.aggregators.llm_context import LLMContext  # noqa: E402
from pipecat.services.llm_service import FunctionCallParams  # noqa: E402


class _FakeConversationJudge:
    """Answers evaluate_conversation from a script and records the conversation it saw."""

    def __init__(self, verdicts: list[str]):
        self.verdicts = list(verdicts)
        self.messages: list[dict] = []
        self.criteria: list[str] = []

    def add_user_message(self, text):
        self.messages.append({"role": "user", "content": text})

    def add_assistant_message(self, text):
        self.messages.append({"role": "assistant", "content": text})

    async def evaluate_conversation(self, criterion: str, *, evidence=()) -> JudgeVerdict:
        self.criteria.append(criterion)
        self.evidence = list(evidence)
        verdict = self.verdicts.pop(0)
        return JudgeVerdict(verdict=verdict, reason=f"because {criterion}", raw_response="")


class _FakePersonaLLM:
    def __init__(self):
        self.handlers: dict = {}

    def register_function(self, name, handler, **kwargs):
        self.handlers[name] = handler


class _FakeClient:
    def __init__(self):
        self.instruction: str | None = None
        self.hung_up = False

    async def configure_persona(self, instruction: str):
        self.instruction = instruction

    async def hang_up(self):
        self.hung_up = True


def _simulation(**overrides) -> EvalSimulation:
    fields = dict(
        name="capital",
        persona="A traveler.",
        goal="Learn the capital of Germany.",
        simulator={"service": "openai"},
        success="the bot said Berlin",
        max_turns=10,
        max_duration_s=5.0,
    )
    fields.update(overrides)
    return EvalSimulation(**fields)


def _driver(simulation: EvalSimulation, judge, context: LLMContext | None = None):
    trace = EvalTrace()
    stream = EvalEventStream(bot_audio=simulation.bot_audio, trace=trace)
    llm = _FakePersonaLLM()
    client = _FakeClient()

    async def progress(_record):
        pass

    driver = SimulationDriver(
        simulation=simulation,
        persona=Persona(simulation.persona, simulation.goal),
        persona_llm=llm,  # type: ignore[arg-type]
        persona_context=context or LLMContext(),
        client=client,  # type: ignore[arg-type]
        stream=stream,
        judge=judge,
        trace=trace,
        progress=progress,
    )
    return driver, stream, llm, client


async def _end_call(llm: _FakePersonaLLM, **arguments):
    """Invoke the registered end_call the way the LLM service would."""
    results: list = []

    async def result_callback(result, *, properties=None):
        results.append((result, properties))

    params = FunctionCallParams(
        function_name="end_call",
        tool_call_id="c1",
        arguments=arguments,
        llm=llm,  # type: ignore[arg-type]
        pipeline_worker=SimpleNamespace(),  # type: ignore[arg-type]
        context=LLMContext(),
        result_callback=result_callback,
    )
    await llm.handlers["end_call"](params)
    return results


class TestSimulationDriver(unittest.IsolatedAsyncioTestCase):
    async def test_end_call_ends_the_run_and_the_judge_sees_the_swapped_conversation(self):
        # The persona's context: the bot is its "user", the persona the "assistant".
        context = LLMContext(
            messages=[
                {"role": "system", "content": "instructions"},
                {"role": "user", "content": "Hi! How can I help?"},
                {"role": "assistant", "content": "What is the capital of Germany?"},
                {"role": "user", "content": "Berlin."},
            ]
        )
        judge = _FakeConversationJudge(["yes", "yes", "no"])
        metrics = [
            EvalSimulationMetric("politeness", "stayed polite", weight=1.0),
            EvalSimulationMetric("brevity", "kept it short", weight=3.0),
        ]
        driver, stream, llm, client = _driver(_simulation(metrics=metrics), judge, context)

        async def conversation():
            await stream.append({"type": "llm_response", "text": "Hi! How can I help?"})
            await stream.append({"type": PERSONA_TURN_EVENT})
            await stream.append({"type": "llm_response", "text": "Berlin."})
            await stream.append({"type": PERSONA_TURN_EVENT})
            results = await _end_call(llm, success=True, reason="I got my answer")
            self.assertEqual(results[0][0], {"status": "call ended"})
            self.assertFalse(results[0][1].run_llm)

        task = asyncio.create_task(conversation())
        failures = await driver.run()
        await task

        self.assertEqual(failures, [])
        self.assertIn("A traveler.", client.instruction or "")
        self.assertTrue(client.hung_up)
        self.assertEqual(
            judge.messages,
            [
                {"role": "assistant", "content": "Hi! How can I help?"},
                {"role": "user", "content": "What is the capital of Germany?"},
                {"role": "assistant", "content": "Berlin."},
            ],
        )
        self.assertEqual(judge.criteria, ["the bot said Berlin", "stayed polite", "kept it short"])

        result = driver.result(
            failures=[], duration_ms=10, events_seen=stream.events_seen, debug_log=[]
        )
        self.assertTrue(result.succeeded)
        self.assertTrue(result.passed)
        self.assertEqual(result.reason, "because the bot said Berlin")
        self.assertEqual(result.ended_by, "end_call")
        self.assertEqual(result.turns, 2)
        self.assertEqual(result.end_call, {"success": True, "reason": "I got my answer"})
        self.assertEqual([m.score for m in result.metrics], [1.0, 0.0])
        self.assertAlmostEqual(result.quality or -1, 1.0 / 4.0)
        self.assertEqual(result.messages, judge.messages)
        self.assertIn(END_CALL_EVENT, [e["type"] for e in stream.events_seen])

    async def test_bot_turn_cap_ends_the_run(self):
        driver, stream, _, _ = _driver(_simulation(max_turns=2), _FakeConversationJudge(["no"]))

        async def conversation():
            for text in ("one", "two", "three"):
                await stream.append({"type": "llm_response", "text": text})
                await stream.append({"type": PERSONA_TURN_EVENT})

        task = asyncio.create_task(conversation())
        await driver.run()
        await task
        result = driver.result(failures=[], duration_ms=0, events_seen=[], debug_log=[])
        self.assertEqual(result.ended_by, "max_turns")
        self.assertEqual(result.turns, 2)
        self.assertFalse(result.succeeded)
        self.assertIsNone(result.quality)  # no metrics

    async def test_the_bot_hanging_up_ends_the_run(self):
        judge = _FakeConversationJudge(["yes"])
        driver, stream, _, client = _driver(_simulation(), judge)

        async def conversation():
            await stream.append({"type": "llm_response", "text": "Bye!"})
            await stream.append({"type": BOT_ENDED_EVENT})

        task = asyncio.create_task(conversation())
        await driver.run()
        await task
        result = driver.result(failures=[], duration_ms=0, events_seen=[], debug_log=[])
        self.assertEqual(result.ended_by, "bot")
        self.assertTrue(client.hung_up)

    async def test_the_bots_tool_calls_are_the_judges_evidence(self):
        judge = _FakeConversationJudge(["yes"])
        driver, stream, llm, _ = _driver(_simulation(), judge)

        async def conversation():
            await stream.append(
                {"type": "function_call", "name": "check_availability", "args": {"time": "6:00 PM"}}
            )
            await stream.append(
                {
                    "type": "function_call_stopped",
                    "name": "check_availability",
                    "args": {"cancelled": False},
                }
            )
            await stream.append({"type": "function_call", "name": "end_conversation", "args": {}})
            await stream.append(
                {
                    "type": "function_call_stopped",
                    "name": "end_conversation",
                    "args": {"cancelled": True},
                }
            )
            await _end_call(llm, success=True, reason="booked")

        task = asyncio.create_task(conversation())
        await driver.run()
        await task
        self.assertEqual(
            judge.evidence,
            [
                'check_availability({"time": "6:00 PM"})',
                "end_conversation()",
                "end_conversation was cancelled",
            ],
        )

    async def test_wall_clock_cap_ends_the_run(self):
        driver, _, _, _ = _driver(_simulation(max_duration_s=0.05), _FakeConversationJudge(["no"]))
        await driver.run()
        result = driver.result(failures=[], duration_ms=0, events_seen=[], debug_log=[])
        self.assertEqual(result.ended_by, "max_duration")

    async def test_only_persona_turns_count(self):
        driver, stream, llm, _ = _driver(_simulation(), _FakeConversationJudge(["yes"]))

        async def conversation():
            await stream.append({"type": "llm_response", "text": "the bot's turn"})
            await stream.append({"type": "response", "text": "its transcription"})
            await stream.append({"type": PERSONA_TURN_EVENT})
            await _end_call(llm, success=True, reason="done")

        task = asyncio.create_task(conversation())
        await driver.run()
        await task
        result = driver.result(failures=[], duration_ms=0, events_seen=[], debug_log=[])
        self.assertEqual(result.turns, 1)

    async def test_a_run_level_failure_is_an_error_not_a_goal_failure(self):
        driver, _, _, _ = _driver(_simulation(), _FakeConversationJudge([]))
        failure = EvalAssertionFailure(
            turn_index=-1,
            expectation_index=-1,
            event_name="<connect>",
            reason="refused",
            kind="connect_failed",
        )
        result = driver.result(failures=[failure], duration_ms=0, events_seen=[], debug_log=[])
        self.assertEqual(result.error, "refused")
        self.assertFalse(result.succeeded)
        self.assertFalse(result.passed)
        self.assertEqual(result.ended_by, "error")
        self.assertEqual(result.reason, "refused")
