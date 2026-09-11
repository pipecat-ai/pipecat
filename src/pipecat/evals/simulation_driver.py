#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The simulation driver: lets the persona hold the conversation, then judges the whole of it."""

import json
import time
from collections.abc import Awaitable, Callable

from pipecat.evals.base_driver import BaseEvalDriver
from pipecat.evals.client import (
    BOT_ENDED_EVENT,
    HARNESS_ERROR_EVENT,
    PERSONA_TURN_EVENT,
    EvalClient,
)
from pipecat.evals.events import EvalEventStream
from pipecat.evals.judge import EvalJudge
from pipecat.evals.persona import EvalPersona
from pipecat.evals.results import (
    EvalAssertionFailure,
    EvalProgress,
    EvalSimulationMetricScore,
    EvalSimulationProgress,
    EvalSimulationResult,
    EvalSimulationTurnVerdict,
    EvalTrace,
)
from pipecat.evals.script import EvalFunctionCall
from pipecat.evals.simulation import EvalSimulationMetric, EvalSimulationScenario
from pipecat.frames.frames import FunctionCallResultProperties
from pipecat.services.llm_service import FunctionCallParams

# The event the driver appends when the persona calls end_call.
END_CALL_EVENT = "end_call"
# How long the run waits, after the persona hangs up on its own last line, for
# the bot's reply to it. A flow makes its final tool calls in that turn, and a
# caller who hangs up mid-sentence is not what the simulation tests.
CLOSING_REPLY_WAIT_S = 15.0


def _call_matches(spec: EvalFunctionCall, call: EvalFunctionCall) -> bool:
    """Whether a call the bot made is the one a ``calls:`` entry describes.

    Same name, and the entry's ``args``, when given, all present in the call's
    arguments with the same values.
    """
    if spec.name != call.name:
        return False
    actual = call.args or {}
    return all(actual.get(key) == value for key, value in (spec.args or {}).items())


class EvalSimulationDriver(BaseEvalDriver[EvalSimulationResult]):
    """Lets the persona LLM hold the conversation, then judges the whole of it.

    The persona answers the bot on its own inside the client's pipeline. The
    driver reports each line as progress and watches for the end: the
    persona's ``end_call``, the bot hanging up, the turn cap, the time cap, a
    lull neither side breaks, or a failure of the harness's own pipeline. Then
    one judge call over the whole transcript scores every bot turn on every
    criterion and decides the goal.
    """

    def __init__(
        self,
        *,
        simulation: EvalSimulationScenario,
        persona: EvalPersona,
        client: EvalClient,
        stream: EvalEventStream,
        judge: EvalJudge | None,
        trace: EvalTrace,
        progress: Callable[[EvalProgress], Awaitable[None]],
    ):
        """Initialize the driver.

        Args:
            simulation: The simulation being run.
            persona: The simulated caller: its instruction, its LLM in the
                pipeline, and its context.
            client: The connection to the bot.
            stream: The bot's output as events.
            judge: The judge for the goal and the quality criteria.
            trace: The run's trace.
            progress: Awaited with an :class:`EvalSimulationProgress` for each
                line of the conversation, and once when it ends.
        """
        super().__init__(client=client, stream=stream, judge=judge, trace=trace, progress=progress)
        self._simulation = simulation
        self._persona = persona
        self._turns = 0
        self._ended_by: str | None = None
        self._duration_s = 0.0
        self._end_call: dict | None = None
        self._succeeded = False
        self._reason = ""
        self._metrics: list[EvalSimulationMetricScore] = []
        # The conversation as the events tell it, built as they arrive (see
        # :meth:`timeline`): the lines so far, the bot's words since the last
        # persona turn, the tool calls made so far, and how far into the stream's
        # events the build has read.
        self._bot_said = "response" if simulation.bot_audio else "llm_response"
        self._lines: list[dict] = []
        self._pending: list[str] = []
        self._evidence: list[str] = []
        self._observed = 0
        self._criteria = {m.name: m.criterion for m in simulation.metrics if m.criterion}

    async def run(self) -> list[EvalAssertionFailure]:
        """Watch the conversation until it ends, then judge it."""
        self._persona.on_end_call(self._on_end_call)
        await self._client.configure_persona(self._persona.instruction)
        simulation = self._simulation
        # The bot's finished response: in audio mode the harness's transcription
        # of what it said, in text mode its LLM text.
        started = time.monotonic()
        deadline = started + simulation.max_duration_s
        # A lull is measured from the stream's last activity, which a token or
        # a spoken sentence refreshes without making an event, so a reply in
        # progress on either side is never silence, only nothing happening is.
        self._stream.touch()
        failures: list[EvalAssertionFailure] = []
        self._trace.log(
            f"persona: listening (up to {simulation.max_turns} turn(s), "
            f"{simulation.max_duration_s:g}s, {simulation.max_silence_s:g}s of silence)"
        )
        # Set when the persona hangs up on its own last line: the run then
        # waits for the bot's reply to it, or this long, before ending.
        closing_ends: float | None = None
        while self._ended_by is None:
            lull_ends = self._stream.last_activity + simulation.max_silence_s
            waits = [deadline, lull_ends] + ([closing_ends] if closing_ends is not None else [])
            try:
                event = await self._stream.next_any(min(waits))
            except TimeoutError:
                now = time.monotonic()
                if closing_ends is not None:
                    # The persona already hung up; the bot's reply was a courtesy.
                    self._trace.log("persona: no closing reply from the bot; ending the call")
                    self._ended_by = "end_call"
                    break
                if now >= deadline:
                    self._ended_by = "max_duration"
                    break
                if self._stream.last_activity + simulation.max_silence_s > now:
                    continue
                self._ended_by = "silence"
                break
            self._observe_new_events()
            if event["type"] == END_CALL_EVENT:
                # The persona says nothing more from here. The bot still gets
                # its turn if the persona hung up on its own last line.
                await self._client.hang_up()
                if self._pending or not self._lines or self._lines[-1]["role"] != "user":
                    self._ended_by = "end_call"
                else:
                    self._trace.log("persona: hung up; waiting for the bot's closing turn")
                    closing_ends = time.monotonic() + CLOSING_REPLY_WAIT_S
            elif event["type"] == BOT_ENDED_EVENT:
                self._ended_by = "bot"
            elif event["type"] == HARNESS_ERROR_EVENT:
                self._ended_by = "error"
                failures.append(self._harness_failure(event.get("text", "")))
            elif event["type"] == PERSONA_TURN_EVENT:
                self._turns += 1
                await self._report("user", event.get("text", ""))
                if self._turns >= simulation.max_turns:
                    self._ended_by = "max_turns"
            elif event["type"] == self._bot_said and event.get("text"):
                await self._report("bot", event["text"])
                if closing_ends is not None:
                    self._ended_by = "end_call"
        # The persona has said its last word either way: nothing the bot says
        # from here on gets an answer.
        await self._client.hang_up()
        self._duration_s = round(time.monotonic() - started, 3)
        self._trace.log(f"persona: ended by {self._ended_by} after {self._turns} turn(s)")
        # Whatever the bot said last closes the conversation.
        self._observe_new_events()
        self._close_bot_turn()
        await self._report("ended", self._ended_by)
        if failures:
            return failures
        return await self._judge_conversation()

    def result(
        self,
        *,
        failures: list[EvalAssertionFailure],
        duration_ms: int,
        events_seen: list[dict],
        debug_log: list[str],
        skipped: str | None = None,
    ) -> EvalSimulationResult:
        """The run's result; a run-level failure makes it an error, not a goal failure."""
        error = skipped or ("; ".join(f.reason for f in failures) if failures else None)
        return EvalSimulationResult(
            simulation_name=self._simulation.name,
            succeeded=self._succeeded and error is None,
            reason=error or self._reason,
            error=error,
            metrics=self._metrics,
            messages=self.conversation(),
            turns=self._turns,
            ended_by=self._ended_by or "error",
            end_call=self._end_call,
            duration_ms=duration_ms,
            events_seen=events_seen,
            debug_log=debug_log,
        )

    def timeline(self) -> list[dict]:
        """The conversation as the events told it, each line with the tool calls made by then.

        A bot turn is everything the bot said between two persona turns, however
        turn detection or a function call split it; a turn in which the bot said
        nothing is not a turn. Built as the events arrive.
        """
        return list(self._lines)

    def transcript(self) -> list[dict]:
        """The conversation for the judge: the lines, each tool call in place before the line it preceded."""
        entries: list[dict] = []
        placed = 0
        for line in self._lines:
            for call in line["evidence"][placed:]:
                entries.append({"role": "tool", "content": call})
            placed = len(line["evidence"])
            entries.append({"role": line["role"], "content": line["content"]})
        for call in self._evidence[placed:]:
            entries.append({"role": "tool", "content": call})
        return entries

    def conversation(self) -> list[dict]:
        """The conversation with the persona as ``user`` and the bot as ``assistant``, without the tool calls."""
        return [{"role": line["role"], "content": line["content"]} for line in self.timeline()]

    def tool_calls(self) -> list[str]:
        """The bot's function calls in order, one line each; a cancelled call is listed as cancelled."""
        lines = []
        for event in self._stream.events_seen:
            lines.extend(self._evidence_line(event))
        return lines

    async def _report(self, status: str, text: str) -> None:
        """Emit one line of the conversation, or its end, as progress."""
        await self._progress(EvalSimulationProgress(status=status, text=text, turn=self._turns))

    def _harness_failure(self, text: str) -> EvalAssertionFailure:
        """The failure a harness pipeline error becomes: the run's error, scored against the current turn."""
        self._trace.log(f"error: harness pipeline: {text}")
        return EvalAssertionFailure(
            turn_index=self._turns,
            expectation_index=-1,
            event_name="<harness>",
            reason=text,
            kind="error",
        )

    async def _on_end_call(self, params: FunctionCallParams) -> None:
        """The persona's ``end_call``: note its claim and end the conversation."""
        arguments = params.arguments or {}
        self._end_call = {
            "success": bool(arguments.get("success", False)),
            "reason": str(arguments.get("reason", "")),
        }
        # The call is the persona's last word: no follow-up response.
        await params.result_callback(
            {"status": "call ended"}, properties=FunctionCallResultProperties(run_llm=False)
        )
        await self._stream.append(
            {"type": END_CALL_EVENT, "text": self._end_call["reason"], **self._end_call}
        )

    def _observe_new_events(self) -> None:
        """Read the stream's events not yet read into the timeline."""
        events = self._stream.events_seen
        while self._observed < len(events):
            event = events[self._observed]
            self._observed += 1
            kind = event["type"]
            if kind == self._bot_said and event.get("text"):
                self._pending.append(event["text"])
            elif kind == PERSONA_TURN_EVENT:
                self._close_bot_turn()
                text = (event.get("text") or "").strip()
                if text:
                    self._add_line("user", text)
            elif kind in ("function_call", "function_call_stopped"):
                self._evidence.extend(self._evidence_line(event))

    def _close_bot_turn(self) -> None:
        """End the bot's turn: what it said since the persona last spoke becomes a line."""
        text = " ".join(self._pending).strip()
        self._pending.clear()
        if text:
            self._add_line("assistant", text)

    def _add_line(self, role: str, content: str) -> None:
        """Append a line to the timeline."""
        self._lines.append({"role": role, "content": content, "evidence": list(self._evidence)})

    def _evidence_line(self, event: dict) -> list[str]:
        """The evidence line a function-call event contributes, if any."""
        name = event.get("name") or "?"
        if event["type"] == "function_call":
            arguments = event.get("args") or {}
            return [f"{name}({json.dumps(arguments) if arguments else ''})"]
        if event["type"] == "function_call_stopped" and (event.get("args") or {}).get("cancelled"):
            return [f"{name} was cancelled"]
        return []

    async def _judge_conversation(self) -> list[EvalAssertionFailure]:
        """Score the measured metrics, then ask the judge once about the goal and every judged criterion.

        A judge that answers nothing usable about the goal is the run's
        error, not a verdict on the bot.
        """
        measured = {
            metric.name: self._measure(metric)
            for metric in self._simulation.metrics
            if metric.measure is not None
        }
        if self._judge is None:
            self._metrics.extend(measured.values())
            self._reason = "no judge configured"
            return []
        evidence = self.tool_calls()
        if evidence:
            self._trace.log(f"judge: the bot's tool calls: {'; '.join(evidence)}")
        judged = await self._judge.evaluate_run(
            self.transcript(), self._criteria, self._simulation.success
        )
        for metric in self._simulation.metrics:
            if metric.name in measured:
                self._metrics.append(measured[metric.name])
                continue
            verdicts = [
                EvalSimulationTurnVerdict(
                    turn=index,
                    passed=verdict.verdict == "yes",
                    reason=verdict.reason,
                    verdict=verdict.verdict,
                )
                for index, verdict in enumerate(judged.turns.get(metric.name, []), 1)
            ]
            for verdict in verdicts:
                self._trace.log(
                    f"judge: turn {verdict.turn} {metric.name} {verdict.verdict}: {verdict.reason}"
                )
            self._metrics.append(self._score(metric, verdicts))
        self._succeeded = judged.goal.verdict == "yes"
        self._reason = judged.goal.reason
        if judged.goal.verdict == "none":
            self._trace.log(f"judge: no verdict on the goal: {self._reason}")
            return [
                EvalAssertionFailure(
                    turn_index=self._turns,
                    expectation_index=-1,
                    event_name="<judge>",
                    reason=self._reason,
                    kind="judge_no_verdict",
                )
            ]
        self._trace.log(
            f"judge: goal {'achieved' if self._succeeded else 'not achieved'}: {self._reason}"
        )
        return []

    def _score(
        self, metric: EvalSimulationMetric, verdicts: list[EvalSimulationTurnVerdict]
    ) -> EvalSimulationMetricScore:
        """A metric's score over its turn verdicts: the share of turns that passed.

        A metric that fell short only on turns the judge left unanswered is
        judge trouble, and its failure kind says so.
        """
        failure_kind = None
        if not verdicts:
            score, reason, passed = None, "no bot turn to judge", True
        else:
            score = sum(1 for v in verdicts if v.passed) / len(verdicts)
            failed = [v for v in verdicts if not v.passed]
            reason = (
                f"all {len(verdicts)} turn(s)"
                if not failed
                else "; ".join(f"turn {v.turn}: {v.reason}" for v in failed)
            )
            passed = metric.min_score is None or score >= metric.min_score
            if not passed:
                failure_kind = (
                    "judge_no" if any(v.verdict == "no" for v in failed) else "judge_no_verdict"
                )
        self._trace.log(
            f"judge: {metric.name} = {'unscored' if score is None else f'{score:.2f}'}"
            f"{'' if passed else f' (below {metric.min_score:.2f})'}: {reason}"
        )
        return EvalSimulationMetricScore(
            name=metric.name,
            score=score,
            passed=passed,
            reason=reason,
            min_score=metric.min_score,
            verdicts=verdicts,
            failure_kind=failure_kind,
        )

    def _measure(self, metric: EvalSimulationMetric) -> EvalSimulationMetricScore:
        """A measured metric's outcome: its value against its range, or the bot's calls against the list."""
        score: float | None
        if metric.measure == "function_calls":
            value, passed, reason = self._calls_outcome(metric)
            score = 1.0 if passed else 0.0
        else:
            value, described = self._measurement(metric.measure or "")
            low, high = metric.min_value, metric.max_value
            if low is not None and high is not None:
                bound = f"between {low:g} and {high:g}"
            elif high is not None:
                bound = f"at most {high:g}"
            else:
                bound = f"at least {low:g}"
            reason = f"{described}, {bound}"
            if value is None:
                score, passed = None, True
            else:
                inside = (low is None or value >= low) and (high is None or value <= high)
                score, passed = (1.0 if inside else 0.0), inside
        self._trace.log(
            f"measure: {metric.name} = {'unscored' if score is None else f'{score:.2f}'}"
            f"{'' if passed else ' (failed)'}: {reason}"
        )
        return EvalSimulationMetricScore(
            name=metric.name,
            score=score,
            passed=passed,
            reason=reason,
            value=value,
            failure_kind=(
                None if passed else ("out_of_range" if metric.calls is None else "function_calls")
            ),
        )

    def _measurement(self, measure: str) -> tuple[float | None, str]:
        """A measure's value for this run, and the phrase that reports it."""
        if measure == "turns":
            return float(self._turns), f"{self._turns} persona turn(s)"
        if measure == "duration":
            return self._duration_s, f"{self._duration_s:.1f} s"
        if measure == "words":
            longest = max(
                (
                    len(line["content"].split())
                    for line in self.timeline()
                    if line["role"] == "assistant"
                ),
                default=None,
            )
            if longest is None:
                return None, "no reply to measure"
            return float(longest), f"longest reply {longest} words"
        if measure == "latency":
            slowest = max(self._reply_latencies(), default=None)
            if slowest is None:
                return None, "no reply to time"
            timed = (
                "from the persona stopping to the first spoken sentence"
                if self._simulation.bot_audio
                else "to the first token"
            )
            return slowest, f"slowest reply {slowest:.2f} s {timed}"
        raise ValueError(f"unknown measure {measure!r}")

    def _reply_latencies(self) -> list[float]:
        """Seconds from each persona turn to the bot's first word of the reply.

        In text mode, from the send to the first LLM token; in audio mode, from
        the bot noticing the persona stop to its first spoken sentence.
        """
        if self._simulation.bot_audio:
            sent, replied = "user_stopped_speaking", "tts_response"
        else:
            sent, replied = PERSONA_TURN_EVENT, "llm_response"
        latencies: list[float] = []
        sent_at: float | None = None
        for event in self._stream.events_seen:
            if event["type"] == sent:
                sent_at = event.get("at")
            elif event["type"] == replied and sent_at is not None:
                first_word = event.get("started_at", event.get("at"))
                if first_word is not None:
                    latencies.append(round(max(0.0, first_word - sent_at), 3))
                sent_at = None
        return latencies

    def _calls_outcome(self, metric: EvalSimulationMetric) -> tuple[float, bool, str]:
        """The ``function_calls`` measure: the count, whether the set matches, and a reason naming the calls made against the list."""
        made = self._calls_made()
        expected = metric.calls or []
        missing = [spec for spec in expected if not any(_call_matches(spec, c) for c in made)]
        unlisted = [c for c in made if not any(_call_matches(spec, c) for spec in expected)]
        described = ", ".join(c.signature for c in made) or "no calls"
        wanted = ", ".join(spec.signature for spec in expected) or "none"
        return float(len(made)), not missing and not unlisted, f"{described}, expected {wanted}"

    def _calls_made(self) -> list[EvalFunctionCall]:
        """The bot's function calls in order, less the ones it cancelled."""
        made: list[EvalFunctionCall] = []
        for event in self._stream.events_seen:
            if event["type"] == "function_call":
                made.append(EvalFunctionCall(name=event.get("name"), args=event.get("args") or {}))
            elif event["type"] == "function_call_stopped" and (event.get("args") or {}).get(
                "cancelled"
            ):
                for index in range(len(made) - 1, -1, -1):
                    if made[index].name == event.get("name"):
                        del made[index]
                        break
        return made
