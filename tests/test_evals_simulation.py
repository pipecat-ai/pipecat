#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the simulation file format and the persona."""

import tempfile
import unittest
from pathlib import Path

from pipecat.evals.persona import END_CALL_FUNCTION, EvalPersona
from pipecat.evals.results import EvalSimulationResult
from pipecat.evals.scenario import (
    EvalScriptScenario,
    EvalSimulationScenario,
    describe_simulation,
    load_scenario_file,
)
from pipecat.evals.script import EvalFunctionCall

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
        s = EvalSimulationScenario.load(_write(MINIMAL))
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
        self.assertEqual(s.judge["service"], "ollama")

    def test_metrics_and_caps(self):
        s = EvalSimulationScenario.load(
            _write(
                MINIMAL
                + """
metrics:
  - name: politeness
    criterion: "stayed courteous"
  - {name: accuracy, criterion: "no invented facts", min_score: 0.8}
max_turns: 5
max_duration_s: 42
runs: 3
"""
            )
        )
        self.assertEqual([m.name for m in s.metrics], ["politeness", "accuracy"])
        self.assertEqual([m.min_score for m in s.metrics], [None, 0.8])
        self.assertEqual(s.max_turns, 5)
        self.assertEqual(s.max_duration_s, 42.0)
        self.assertEqual(s.runs, 3)

    def test_a_measured_metric_takes_a_measure_and_a_range(self):
        s = EvalSimulationScenario.load(
            _write(
                MINIMAL
                + """
metrics:
  - measure: latency
    max_value: 2
  - name: quick
    measure: turns
    min_value: 1
    max_value: 6
"""
            )
        )
        latency, quick = s.metrics
        self.assertEqual(
            (latency.name, latency.measure, latency.max_value), ("latency", "latency", 2.0)
        )
        self.assertIsNone(latency.criterion)
        self.assertEqual((quick.measure, quick.min_value, quick.max_value), ("turns", 1.0, 6.0))
        for bad, message in (
            ("  - measure: mood\n    max_value: 1\n", "must be one of"),
            ("  - measure: turns\n", "needs a 'min_value:' or a 'max_value:'"),
            ("  - measure: turns\n    max_value: 3\n    min_score: 1\n", "takes a range"),
            (
                "  - name: both\n    criterion: x\n    measure: turns\n    max_value: 3\n",
                "one of the two",
            ),
        ):
            with self.assertRaises(ValueError, msg=bad) as cm:
                EvalSimulationScenario.load(_write(MINIMAL + "metrics:\n" + bad))
            self.assertIn(message, str(cm.exception))

    def test_min_score_is_a_share_and_names_are_unique(self):
        with self.assertRaises(ValueError) as cm:
            EvalSimulationScenario.load(
                _write(MINIMAL + "metrics:\n  - {name: a, criterion: x, min_score: 2}\n")
            )
        self.assertIn("0..1", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            EvalSimulationScenario.load(
                _write(
                    MINIMAL + "metrics:\n  - {name: a, criterion: x}\n  - {name: a, criterion: y}\n"
                )
            )
        self.assertIn("twice", str(cm.exception))

    def test_audio_modalities(self):
        s = EvalSimulationScenario.load(
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
            EvalSimulationScenario.load(_write(MINIMAL + "user: {modality: audio}\n"))
        self.assertIn("user.speech", str(cm.exception))

    def test_required_fields(self):
        for missing in ("persona", "goal", "success"):
            text = "\n".join(line for line in MINIMAL.splitlines() if not line.startswith(missing))
            with self.assertRaises(ValueError, msg=missing) as cm:
                EvalSimulationScenario.load(_write(text))
            self.assertIn(missing, str(cm.exception))

    def test_a_function_calls_measure_takes_the_calls_the_bot_should_make(self):
        s = EvalSimulationScenario.load(
            _write(
                MINIMAL
                + """
metrics:
  - measure: function_calls
    calls:
      - book_table
      - name: send_confirmation
        args: { channel: sms }
  - name: hands_off
    measure: function_calls
    calls: []
"""
            )
        )
        booked, hands_off = s.metrics
        self.assertEqual(booked.name, "function_calls")
        self.assertEqual(
            [(c.name, c.args) for c in booked.calls or []],
            [("book_table", None), ("send_confirmation", {"channel": "sms"})],
        )
        self.assertEqual((hands_off.measure, hands_off.calls), ("function_calls", []))
        for bad, message in (
            ("  - measure: function_calls\n", "needs a 'calls:' list"),
            (
                "  - measure: function_calls\n    calls: [book_table]\n    max_value: 1\n",
                "not a range",
            ),
            (
                "  - measure: function_calls\n    calls: [{args: {a: 1}}]\n",
                "entry #0 must be a name",
            ),
            ("  - measure: turns\n    max_value: 3\n    calls: []\n", "'calls:' belongs to"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                EvalSimulationScenario.load(_write(MINIMAL + "metrics:\n" + bad))

    def test_metric_needs_a_criterion(self):
        with self.assertRaises(ValueError) as cm:
            EvalSimulationScenario.load(_write(MINIMAL + "metrics: [{name: politeness}]\n"))
        self.assertIn("criterion", str(cm.exception))

    def test_caps_must_be_positive(self):
        with self.assertRaises(ValueError):
            EvalSimulationScenario.load(_write(MINIMAL + "max_turns: 0\n"))
        with self.assertRaises(ValueError):
            EvalSimulationScenario.load(_write(MINIMAL + "max_duration_s: -1\n"))

    def test_describe(self):
        text = describe_simulation(EvalSimulationScenario.load(_write(MINIMAL)))
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
        self.assertIsInstance(load_scenario_file(_write(MINIMAL)), EvalSimulationScenario)

    def test_turns_make_a_scripted_scenario(self):
        self.assertIsInstance(
            load_scenario_file(_write("name: greet\nturns: []\n")), EvalScriptScenario
        )

    def test_a_file_is_one_kind_or_the_other(self):
        with self.assertRaises(ValueError) as cm:
            load_scenario_file(_write("name: nothing\n"))
        self.assertIn("'turns:'", str(cm.exception))
        self.assertIn("'persona:'", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            load_scenario_file(_write(MINIMAL + "turns: []\n"))
        self.assertIn("not both", str(cm.exception))


class TestPersona(unittest.TestCase):
    def test_instruction_llm_and_context(self):
        llm = _FakePersonaLLM()
        persona = EvalPersona("A curious traveler.", "Learn the capital of Germany.", llm)  # type: ignore[arg-type]
        self.assertIn("A curious traveler.", persona.instruction)
        self.assertIn("Learn the capital of Germany.", persona.instruction)
        self.assertIn(END_CALL_FUNCTION, persona.instruction)
        self.assertIs(persona.llm, llm)
        context = persona.context
        self.assertEqual(context.get_messages(), [])
        tools = context.tools
        assert not isinstance(tools, type(None))
        self.assertEqual([t.name for t in tools.standard_tools], [END_CALL_FUNCTION])  # type: ignore[union-attr]


class TestSimulationRunResult(unittest.TestCase):
    def test_passed_needs_success_and_no_error(self):
        self.assertTrue(EvalSimulationResult("s", succeeded=True).passed)
        self.assertFalse(EvalSimulationResult("s", succeeded=False).passed)
        self.assertFalse(EvalSimulationResult("s", succeeded=True, error="boom").passed)


# ---------------------------------------------------------------------------
# The simulation driver, over fakes.
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from pipecat.evals.client import (  # noqa: E402
    BOT_ENDED_EVENT,
    HARNESS_ERROR_EVENT,  # noqa: E402
    PERSONA_TURN_EVENT,
)
from pipecat.evals.events import EvalEventStream  # noqa: E402
from pipecat.evals.judge import JudgeVerdict, RunVerdicts  # noqa: E402
from pipecat.evals.results import EvalAssertionFailure, EvalTrace  # noqa: E402
from pipecat.evals.scenario import EvalSimulationMetric  # noqa: E402
from pipecat.evals.simulation_driver import END_CALL_EVENT, EvalSimulationDriver  # noqa: E402
from pipecat.processors.aggregators.llm_context import LLMContext  # noqa: E402
from pipecat.services.llm_service import FunctionCallParams  # noqa: E402


class _FakeConversationJudge:
    """Answers the judge's one run question from a script and records what it saw.

    ``verdicts`` feed the goal in order; ``turn_verdicts`` are one dict per bot
    turn, a metric left out of a dict passing that turn.
    """

    def __init__(self, verdicts: list[str], turn_verdicts: list[dict[str, str]] | None = None):
        self.verdicts = list(verdicts)
        self.turn_verdicts = list(turn_verdicts or [])
        self.transcript: list[dict] = []
        self.criteria: list[str] = []
        self.run_criteria: dict[str, str] = {}

    async def evaluate_run(self, transcript, criteria: dict[str, str], success: str):
        self.transcript = list(transcript)
        self.criteria.append(success)
        self.run_criteria = dict(criteria)
        turns = sum(1 for e in transcript if e["role"] == "assistant")
        goal = JudgeVerdict(
            verdict=self.verdicts.pop(0), reason=f"because {success}", raw_response=""
        )
        by_name = {}
        for name, criterion in criteria.items():
            by_name[name] = []
            for index in range(turns):
                scripted = self.turn_verdicts[index] if index < len(self.turn_verdicts) else {}
                by_name[name].append(
                    JudgeVerdict(
                        verdict=scripted.get(name, "yes"),
                        reason=f"because {criterion}",
                        raw_response="",
                    )
                )
        return RunVerdicts(goal=goal, turns=by_name)


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


def _simulation(**overrides) -> EvalSimulationScenario:
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
    return EvalSimulationScenario(**fields)


def _driver(
    simulation: EvalSimulationScenario,
    judge,
    progress_records: list | None = None,
):
    trace = EvalTrace()
    stream = EvalEventStream(bot_audio=simulation.bot_audio, trace=trace)
    llm = _FakePersonaLLM()
    client = _FakeClient()

    async def progress(record):
        if progress_records is not None:
            progress_records.append(record)

    driver = EvalSimulationDriver(
        simulation=simulation,
        persona=EvalPersona(simulation.persona, simulation.goal, llm),  # type: ignore[arg-type]
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
        # The second bot turn is short but the judge finds it curt.
        judge = _FakeConversationJudge(["yes"], [{}, {"brevity": "no"}])
        metrics = [
            EvalSimulationMetric("politeness", "stayed polite", min_score=1.0),
            EvalSimulationMetric("brevity", "kept it short"),
        ]
        driver, stream, llm, client = _driver(_simulation(metrics=metrics), judge)

        async def conversation():
            await stream.append({"type": "llm_response", "text": "Hi! How can I help?"})
            await stream.append(
                {"type": PERSONA_TURN_EVENT, "text": "What is the capital of Germany?"}
            )
            await stream.append({"type": "llm_response", "text": "Berlin."})
            await stream.append({"type": PERSONA_TURN_EVENT, "text": "Thanks!"})
            results = await _end_call(llm, success=True, reason="I got my answer")
            self.assertEqual(results[0][0], {"status": "call ended"})
            self.assertFalse(results[0][1].run_llm)

        task = asyncio.create_task(conversation())
        failures = await driver.run()
        await task

        self.assertEqual(failures, [])
        self.assertIn("A traveler.", client.instruction or "")
        self.assertTrue(client.hung_up)
        # One judge call saw the whole conversation, the criteria, and the goal.
        self.assertEqual(
            judge.transcript,
            [
                {"role": "assistant", "content": "Hi! How can I help?"},
                {"role": "user", "content": "What is the capital of Germany?"},
                {"role": "assistant", "content": "Berlin."},
                {"role": "user", "content": "Thanks!"},
            ],
        )
        self.assertEqual(
            judge.run_criteria, {"politeness": "stayed polite", "brevity": "kept it short"}
        )
        self.assertEqual(judge.criteria, ["the bot said Berlin"])

        result = driver.result(
            failures=[], duration_ms=10, events_seen=stream.events_seen, debug_log=[]
        )
        self.assertTrue(result.succeeded)
        self.assertTrue(result.passed)
        self.assertEqual(result.reason, "because the bot said Berlin")
        self.assertEqual(result.ended_by, "end_call")
        self.assertEqual(result.turns, 2)
        self.assertEqual(result.end_call, {"success": True, "reason": "I got my answer"})
        self.assertEqual([m.score for m in result.metrics], [1.0, 0.5])
        # brevity scored 0.5 but gates nothing, so the run still passes.
        self.assertEqual([m.passed for m in result.metrics], [True, True])
        self.assertEqual(result.metrics[0].reason, "all 2 turn(s)")
        self.assertEqual(result.metrics[1].reason, "turn 2: because kept it short")
        self.assertEqual([v.turn for v in result.metrics[1].verdicts if not v.passed], [2])
        self.assertIsNone(result.failure)
        self.assertEqual(result.messages, judge.transcript)
        self.assertIn(END_CALL_EVENT, [e["type"] for e in stream.events_seen])

    async def test_a_metric_below_its_min_score_fails_the_run(self):
        judge = _FakeConversationJudge(["yes"], [{"politeness": "no"}])
        metrics = [EvalSimulationMetric("politeness", "stayed polite", min_score=1.0)]
        records: list = []
        driver, stream, llm, _ = _driver(_simulation(metrics=metrics), judge, records)

        async def conversation():
            await stream.append({"type": "llm_response", "text": "What do you want."})
            await stream.append({"type": PERSONA_TURN_EVENT, "text": "The capital of Germany?"})
            await _end_call(llm, success=True, reason="rude but answered")

        task = asyncio.create_task(conversation())
        await driver.run()
        await task
        result = driver.result(
            failures=[], duration_ms=10, events_seen=stream.events_seen, debug_log=[]
        )
        self.assertTrue(result.succeeded)
        self.assertFalse(result.passed)
        self.assertEqual(
            result.failure, "politeness 0.00 below 1.00: turn 1: because stayed polite"
        )
        self.assertEqual(
            [(r.status, r.text, r.turn) for r in records if r.status != "bot"],
            [("user", "The capital of Germany?", 1), ("ended", "end_call", 1)],
        )

    async def test_measures_come_from_the_run_and_a_value_out_of_range_fails_it(self):
        metrics = [
            EvalSimulationMetric("latency", measure="latency", max_value=1.0),
            EvalSimulationMetric("words", measure="words", max_value=3),
            EvalSimulationMetric("turns", measure="turns", max_value=5),
            EvalSimulationMetric("duration", measure="duration", min_value=0),
        ]
        driver, stream, llm, _ = _driver(
            _simulation(metrics=metrics), _FakeConversationJudge(["yes"])
        )

        async def conversation():
            # Times are seconds on the stream's clock; a reply's first token is started_at.
            await stream.append(
                {"type": "llm_response", "text": "Hi there", "at": 0.5, "started_at": 0.2}
            )
            await stream.append(
                {"type": PERSONA_TURN_EVENT, "text": "Capital of Germany?", "at": 1.0}
            )
            await stream.append({"type": "bot_interrupted", "at": 1.1})
            await stream.append(
                {
                    "type": "llm_response",
                    "text": "It is Berlin, of course.",
                    "at": 3.0,
                    "started_at": 2.4,
                }
            )
            await stream.append({"type": PERSONA_TURN_EVENT, "text": "Thanks", "at": 3.5})
            await stream.append(
                {"type": "llm_response", "text": "Bye!", "at": 3.9, "started_at": 3.8}
            )
            await _end_call(llm, success=True, reason="done")

        task = asyncio.create_task(conversation())
        await driver.run()
        await task
        result = driver.result(
            failures=[], duration_ms=10, events_seen=stream.events_seen, debug_log=[]
        )
        by_name = {m.name: m for m in result.metrics}
        # The slowest reply took 1.4 s, over the 1 s bound; the greeting had no send before it.
        self.assertEqual((by_name["latency"].value, by_name["latency"].passed), (1.4, False))
        self.assertEqual(
            by_name["latency"].reason, "slowest reply 1.40 s to the first token, at most 1"
        )
        self.assertEqual((by_name["words"].value, by_name["words"].passed), (5.0, False))
        self.assertEqual((by_name["turns"].value, by_name["turns"].passed), (2.0, True))
        self.assertTrue(by_name["duration"].passed)
        self.assertTrue(result.succeeded)
        self.assertFalse(result.passed)
        self.assertEqual(
            result.failure, "latency: slowest reply 1.40 s to the first token, at most 1"
        )

    async def test_function_calls_are_checked_against_the_list_the_bot_should_make(self):
        metrics = [
            EvalSimulationMetric(
                "booking",
                measure="function_calls",
                calls=[EvalFunctionCall(name="book_table", args={"party_size": 2})],
            ),
            EvalSimulationMetric("hands_off", measure="function_calls", calls=[]),
            EvalSimulationMetric(
                "lookup",
                measure="function_calls",
                calls=[
                    EvalFunctionCall(name="check_availability"),
                    EvalFunctionCall(name="book_table"),
                ],
            ),
        ]
        driver, stream, llm, _ = _driver(
            _simulation(metrics=metrics), _FakeConversationJudge(["yes"])
        )

        async def conversation():
            await stream.append({"type": "llm_response", "text": "Hello!"})
            await stream.append({"type": PERSONA_TURN_EVENT, "text": "A table for two at six."})
            await stream.append(
                {"type": "function_call", "name": "check_availability", "args": {"time": "6"}}
            )
            # A cancelled call did not happen.
            await stream.append({"type": "function_call", "name": "send_confirmation", "args": {}})
            await stream.append(
                {
                    "type": "function_call_stopped",
                    "name": "send_confirmation",
                    "args": {"tool_call_id": "1", "cancelled": True},
                }
            )
            await stream.append(
                {
                    "type": "function_call",
                    "name": "book_table",
                    "args": {"party_size": 2, "time": "6"},
                }
            )
            await stream.append({"type": "llm_response", "text": "Booked."})
            await _end_call(llm, success=True, reason="booked")

        task = asyncio.create_task(conversation())
        await driver.run()
        await task
        result = driver.result(
            failures=[], duration_ms=10, events_seen=stream.events_seen, debug_log=[]
        )
        by_name = {m.name: m for m in result.metrics}
        made = "check_availability(time='6'), book_table(party_size=2, time='6')"
        # The list is the whole set: the lookup is not on the booking list, so it fails.
        self.assertEqual((by_name["booking"].value, by_name["booking"].passed), (2.0, False))
        self.assertEqual(by_name["booking"].reason, f"{made}, expected book_table(party_size=2)")
        self.assertFalse(by_name["hands_off"].passed)
        self.assertEqual(by_name["hands_off"].reason, f"{made}, expected none")
        # Names alone match, in any order, and the cancelled call is not counted.
        self.assertTrue(by_name["lookup"].passed)
        self.assertEqual(
            by_name["lookup"].reason, f"{made}, expected check_availability, book_table"
        )
        self.assertEqual(result.failure, f"booking: {made}, expected book_table(party_size=2)")

    async def test_a_bot_that_should_call_nothing_passes_when_it_calls_nothing(self):
        metrics = [EvalSimulationMetric("hands_off", measure="function_calls", calls=[])]
        driver, stream, llm, _ = _driver(
            _simulation(metrics=metrics), _FakeConversationJudge(["yes"])
        )

        async def conversation():
            await stream.append({"type": "llm_response", "text": "I can't do that for you."})
            await _end_call(llm, success=True, reason="turned down")

        task = asyncio.create_task(conversation())
        await driver.run()
        await task
        result = driver.result(
            failures=[], duration_ms=10, events_seen=stream.events_seen, debug_log=[]
        )
        (hands_off,) = result.metrics
        self.assertEqual(
            (hands_off.value, hands_off.passed, hands_off.reason),
            (0.0, True, "no calls, expected none"),
        )
        self.assertTrue(result.passed)

    async def test_a_bot_turn_is_what_it_said_between_persona_turns_with_the_calls_by_then(self):
        judge = _FakeConversationJudge(["yes"])
        metrics = [EvalSimulationMetric("honesty", "claims only what a call backs")]
        driver, stream, llm, _ = _driver(_simulation(metrics=metrics), judge)

        async def conversation():
            await stream.append({"type": "llm_response", "text": "Hello!"})
            await stream.append({"type": PERSONA_TURN_EVENT, "text": "A table at six, please."})
            # A function call splits the reply in two; both halves are one turn,
            # and the call is that turn's evidence, not the greeting's.
            await stream.append({"type": "llm_response", "text": "Let me check."})
            await stream.append(
                {"type": "function_call", "name": "check_availability", "args": {"time": "6"}}
            )
            await stream.append({"type": "llm_response", "text": "Six is free, booked."})
            await stream.append({"type": PERSONA_TURN_EVENT, "text": ""})
            await _end_call(llm, success=True, reason="booked")

        task = asyncio.create_task(conversation())
        await driver.run()
        await task
        # The call sits in the transcript where it happened, before the turn it split.
        self.assertEqual(
            judge.transcript,
            [
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "A table at six, please."},
                {"role": "tool", "content": 'check_availability({"time": "6"})'},
                {"role": "assistant", "content": "Let me check. Six is free, booked."},
            ],
        )
        result = driver.result(
            failures=[], duration_ms=10, events_seen=stream.events_seen, debug_log=[]
        )
        self.assertEqual(
            result.messages,
            [
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "A table at six, please."},
                {"role": "assistant", "content": "Let me check. Six is free, booked."},
            ],
        )

    async def test_the_conversation_is_reported_as_it_happens(self):
        records: list = []
        driver, stream, llm, _ = _driver(_simulation(), _FakeConversationJudge(["yes"]), records)

        async def conversation():
            await stream.append({"type": "llm_response", "text": "Hi! How can I help?"})
            await stream.append({"type": PERSONA_TURN_EVENT, "text": "What is the capital?"})
            await stream.append({"type": "llm_response", "text": "Berlin."})
            # A response without text (a function call's own) is not a line.
            await stream.append({"type": "llm_response", "text": ""})
            await _end_call(llm, success=True, reason="done")

        task = asyncio.create_task(conversation())
        await driver.run()
        await task

        self.assertEqual(
            [(r.status, r.text, r.turn) for r in records],
            [
                ("bot", "Hi! How can I help?", 0),
                ("user", "What is the capital?", 1),
                ("bot", "Berlin.", 1),
                ("ended", "end_call", 1),
            ],
        )

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
            [e["content"] for e in judge.transcript if e["role"] == "tool"],
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


class TestSimulationEarlyEndings(unittest.IsolatedAsyncioTestCase):
    """A lull, a harness failure, or a judge with no verdict ends the run without waiting out the caps."""

    async def test_a_lull_ends_the_run_as_silence(self):
        driver, _, _, _ = _driver(
            _simulation(max_silence_s=0.05, max_duration_s=5.0), _FakeConversationJudge(["no"])
        )
        await driver.run()
        result = driver.result(failures=[], duration_ms=0, events_seen=[], debug_log=[])
        self.assertEqual(result.ended_by, "silence")
        self.assertFalse(result.passed)

    async def test_activity_keeps_a_lull_from_ending_the_run(self):
        driver, stream, llm, _ = _driver(
            _simulation(max_silence_s=0.2, max_duration_s=5.0), _FakeConversationJudge(["yes"])
        )

        async def conversation():
            # Three events spaced inside the lull cap, spanning more than the cap.
            for _ in range(3):
                await asyncio.sleep(0.1)
                await stream.append({"type": "llm_started"})
            await _end_call(llm, success=True, reason="done")

        task = asyncio.create_task(conversation())
        await driver.run()
        await task
        result = driver.result(failures=[], duration_ms=0, events_seen=[], debug_log=[])
        self.assertEqual(result.ended_by, "end_call")

    async def test_a_harness_pipeline_error_is_the_runs_error(self):
        driver, stream, _, client = _driver(_simulation(), _FakeConversationJudge(["yes"]))

        async def failing():
            await stream.append({"type": HARNESS_ERROR_EVENT, "text": "OLLamaLLMService: 400"})

        task = asyncio.create_task(failing())
        failures = await driver.run()
        await task
        self.assertEqual(
            [(f.kind, f.reason) for f in failures], [("error", "OLLamaLLMService: 400")]
        )
        self.assertTrue(client.hung_up)
        result = driver.result(failures=failures, duration_ms=0, events_seen=[], debug_log=[])
        self.assertEqual((result.ended_by, result.error), ("error", "OLLamaLLMService: 400"))
        self.assertFalse(result.passed)

    async def test_a_judge_with_no_goal_verdict_is_the_runs_error(self):
        driver, stream, llm, _ = _driver(_simulation(), _FakeConversationJudge(["none"]))

        async def conversation():
            await stream.append({"type": "llm_response", "text": "Berlin."})
            await _end_call(llm, success=True, reason="done")

        task = asyncio.create_task(conversation())
        failures = await driver.run()
        await task
        self.assertEqual([f.kind for f in failures], ["judge_no_verdict"])
        result = driver.result(failures=failures, duration_ms=0, events_seen=[], debug_log=[])
        self.assertEqual(result.ended_by, "end_call")
        self.assertFalse(result.succeeded)
        self.assertIn("because", result.error or "")


class TestSimulationMetricKinds(unittest.IsolatedAsyncioTestCase):
    """A failed metric's failure kind says whether the judge rejected a turn or left it unanswered."""

    async def _score(self, turn_verdicts: list[dict[str, str]]):
        metrics = [EvalSimulationMetric("brevity", "short", min_score=1.0)]
        driver, stream, llm, _ = _driver(
            _simulation(metrics=metrics), _FakeConversationJudge(["yes"], turn_verdicts)
        )

        async def conversation():
            await stream.append({"type": "llm_response", "text": "one"})
            await stream.append({"type": PERSONA_TURN_EVENT, "text": "and?"})
            await stream.append({"type": "llm_response", "text": "two"})
            await _end_call(llm, success=True, reason="done")

        task = asyncio.create_task(conversation())
        await driver.run()
        await task
        result = driver.result(failures=[], duration_ms=0, events_seen=[], debug_log=[])
        return result.metrics[0]

    async def test_a_rejected_turn_is_judge_no(self):
        metric = await self._score([{}, {"brevity": "no"}])
        self.assertEqual((metric.passed, metric.failure_kind), (False, "judge_no"))
        self.assertEqual([v.verdict for v in metric.verdicts], ["yes", "no"])

    async def test_an_unanswered_turn_alone_is_judge_no_verdict(self):
        metric = await self._score([{}, {"brevity": "none"}])
        self.assertEqual((metric.passed, metric.failure_kind), (False, "judge_no_verdict"))
        self.assertEqual([v.verdict for v in metric.verdicts], ["yes", "none"])

    async def test_a_passing_metric_has_no_kind(self):
        metric = await self._score([{}, {}])
        self.assertEqual((metric.passed, metric.failure_kind), (True, None))


class TestSimulatorDefault(unittest.TestCase):
    def test_simulator_is_optional_and_describes_as_the_default_judge_model(self):
        text = "\n".join(line for line in MINIMAL.splitlines() if not line.startswith("simulator"))
        s = EvalSimulationScenario.load(_write(text))
        self.assertEqual(s.simulator, {})
        self.assertEqual(s.max_silence_s, 30.0)
        described = describe_simulation(s)
        self.assertIn("persona: ollama/gemma4:12b", described)
        self.assertIn("max_silence_s: 30", described)

    def test_a_partial_simulator_keeps_its_own_values(self):
        s = EvalSimulationScenario.load(_write(MINIMAL + "max_silence_s: 7\n"))
        self.assertEqual(s.max_silence_s, 7.0)
        self.assertIn("persona: openai/gpt-4o-mini", describe_simulation(s))
