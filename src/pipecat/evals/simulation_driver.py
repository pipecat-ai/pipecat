#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Simulation driver: lets the persona hold the conversation, then judges it.

The :class:`EvalSimulationDriver` registers the persona's ``end_call`` on the
persona LLM, watches the event stream until the persona ends the call, the
bot's turns reach the cap, or the wall clock runs out, then has the judge
decide the goal and score each quality criterion over the whole conversation,
assembling the run's :class:`~pipecat.evals.results.EvalSimulationResult`.
"""

import json
import time
from collections.abc import Awaitable, Callable

from pipecat.evals.base_driver import BaseEvalDriver
from pipecat.evals.client import BOT_ENDED_EVENT, PERSONA_TURN_EVENT, EvalClient
from pipecat.evals.events import EvalEventStream
from pipecat.evals.judge import EvalJudge
from pipecat.evals.persona import END_CALL_FUNCTION, EvalPersona
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
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import FunctionCallParams, LLMService

# The event the driver appends when the persona calls end_call.
END_CALL_EVENT = "end_call"


def _call_matches(spec: EvalFunctionCall, call: EvalFunctionCall) -> bool:
    """Whether a call the bot made is the one a ``calls:`` entry describes.

    Names must be equal; the entry's ``args``, when given, must all be present
    in the call's arguments with the same values, extra arguments ignored.
    """
    if spec.name != call.name:
        return False
    actual = call.args or {}
    return all(actual.get(key) == value for key, value in (spec.args or {}).items())


class EvalSimulationDriver(BaseEvalDriver[EvalSimulationResult]):
    """Lets the persona LLM hold the conversation, then judges the whole of it.

    The persona runs inside the client's pipeline and answers the bot on its
    own. This driver watches the conversation, reporting each line as progress,
    for its end: the persona's ``end_call``, after which it hangs up, the bot
    ending the call, the cap on the persona's turns, or the wall-clock cap.
    Then one judge call over the whole transcript, the bot's tool calls in
    place, scores every bot turn on every criterion and decides the goal.
    """

    def __init__(
        self,
        *,
        simulation: EvalSimulationScenario,
        persona: EvalPersona,
        persona_llm: LLMService,
        persona_context: LLMContext,
        client: EvalClient,
        stream: EvalEventStream,
        judge: EvalJudge | None,
        trace: EvalTrace,
        progress: Callable[[EvalProgress], Awaitable[None]],
    ):
        """Initialize the driver.

        Args:
            simulation: The simulation being run.
            persona: The simulated caller; its instruction goes to the persona LLM.
            persona_llm: The persona LLM service in the pipeline; ``end_call``
                is registered on it.
            persona_context: The persona's context, kept up to date with both
                sides of the conversation by the pipeline's aggregators.
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
        self._persona_llm = persona_llm
        self._context = persona_context
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
        self._persona_llm.register_function(END_CALL_FUNCTION, self._on_end_call)
        await self._client.configure_persona(self._persona.instruction)
        simulation = self._simulation
        # The bot's finished response: in audio mode the harness's transcription
        # of what it said, in text mode its LLM text.
        started = time.monotonic()
        deadline = started + simulation.max_duration_s
        self._trace.log(
            f"persona: listening (up to {simulation.max_turns} turn(s), "
            f"{simulation.max_duration_s:g}s)"
        )
        while self._ended_by is None:
            try:
                event = await self._stream.next_any(deadline)
            except TimeoutError:
                self._ended_by = "max_duration"
                break
            self._observe_new_events()
            if event["type"] == END_CALL_EVENT:
                self._ended_by = "end_call"
            elif event["type"] == BOT_ENDED_EVENT:
                self._ended_by = "bot"
            elif event["type"] == PERSONA_TURN_EVENT:
                self._turns += 1
                await self._report("user", event.get("text", ""))
                if self._turns >= simulation.max_turns:
                    self._ended_by = "max_turns"
            elif event["type"] == self._bot_said and event.get("text"):
                await self._report("bot", event["text"])
        # The persona has said its last word either way: nothing the bot says
        # from here on gets an answer.
        await self._client.hang_up()
        self._duration_s = round(time.monotonic() - started, 3)
        self._trace.log(f"persona: ended by {self._ended_by} after {self._turns} turn(s)")
        # Whatever the bot said last closes the conversation.
        self._observe_new_events()
        self._close_bot_turn()
        await self._report("ended", self._ended_by)
        await self._judge_conversation()
        return []

    async def _report(self, status: str, text: str) -> None:
        """Emit one line of the conversation, or its end, as progress."""
        await self._progress(EvalSimulationProgress(status=status, text=text, turn=self._turns))

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

    def timeline(self) -> list[dict]:
        """The conversation as the events told it, each line with the tool calls before it.

        A bot turn is everything the bot said between two persona turns, which
        merges the segments audio-mode turn detection splits a reply into and
        the responses a function call splits it into. Each entry has a ``role``
        (``assistant`` for the bot, ``user`` for the persona), the ``content``,
        and ``evidence``: the bot's tool calls made by then, as
        :meth:`tool_calls` lists them. Turns in which the bot said nothing are
        not turns. Built as the events arrive, so it is complete once the run
        is over.
        """
        return list(self._lines)

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

    def transcript(self) -> list[dict]:
        """The conversation for the judge: the lines, with each tool call in place.

        A ``tool`` entry carries one call as :meth:`tool_calls` lists it,
        placed before the first line it preceded.
        """
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
        """The conversation with the persona as ``user`` and the bot as ``assistant``.

        The lines of :meth:`timeline`, without the evidence: what the judge
        reads and the result records.
        """
        return [{"role": line["role"], "content": line["content"]} for line in self.timeline()]

    def tool_calls(self) -> list[str]:
        """The bot's function calls in order, one line each, as the judge's evidence.

        A call the bot cancelled is listed as such: it is evidence that the
        action did not happen.
        """
        lines = []
        for event in self._stream.events_seen:
            lines.extend(self._evidence_line(event))
        return lines

    def _evidence_line(self, event: dict) -> list[str]:
        """The evidence line a function-call event contributes, if any."""
        name = event.get("name") or "?"
        if event["type"] == "function_call":
            arguments = event.get("args") or {}
            return [f"{name}({json.dumps(arguments) if arguments else ''})"]
        if event["type"] == "function_call_stopped" and (event.get("args") or {}).get("cancelled"):
            return [f"{name} was cancelled"]
        return []

    async def _judge_conversation(self) -> None:
        """Score the measured metrics, then settle the judged ones and the goal in one call.

        The judge reads the whole transcript once, the bot's tool calls in place,
        and answers for every bot turn on every criterion and for the goal.
        Measured metrics need no judge and take the file's order with the rest.
        """
        measured = {
            metric.name: self._measure(metric)
            for metric in self._simulation.metrics
            if metric.measure is not None
        }
        if self._judge is None:
            self._metrics.extend(measured.values())
            self._reason = "no judge configured"
            return
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
                    turn=index, passed=verdict.verdict == "yes", reason=verdict.reason
                )
                for index, verdict in enumerate(judged.turns.get(metric.name, []), 1)
            ]
            for verdict in verdicts:
                self._trace.log(
                    f"judge: turn {verdict.turn} {metric.name} "
                    f"{'yes' if verdict.passed else 'no'}: {verdict.reason}"
                )
            self._metrics.append(self._score(metric, verdicts))
        self._succeeded = judged.goal.verdict == "yes"
        self._reason = judged.goal.reason
        self._trace.log(
            f"judge: goal {'achieved' if self._succeeded else 'not achieved'}: {self._reason}"
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
            name=metric.name, score=score, passed=passed, reason=reason, value=value
        )

    def _calls_outcome(self, metric: EvalSimulationMetric) -> tuple[float, bool, str]:
        """The ``function_calls`` measure: the count, whether the set matches, and the reason.

        The reason reads the calls made against the list, so a failing run says
        which call was missing or unlisted.
        """
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
            return slowest, f"slowest reply {slowest:.2f} s"
        raise ValueError(f"unknown measure {measure!r}")

    def _reply_latencies(self) -> list[float]:
        """Seconds from each persona turn to the bot's first word of the reply after it.

        In text mode the persona's turn is its send and the bot's first word the
        first token of the LLM response that follows; in audio mode they are the
        bot's own report of the persona stopping and its first spoken sentence.
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

    def _score(
        self, metric: EvalSimulationMetric, verdicts: list[EvalSimulationTurnVerdict]
    ) -> EvalSimulationMetricScore:
        """A metric's score over its turn verdicts: the share of turns that passed."""
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
            passed = metric.min_quality is None or score >= metric.min_quality
        self._trace.log(
            f"judge: {metric.name} = {'unscored' if score is None else f'{score:.2f}'}"
            f"{'' if passed else f' (below {metric.min_quality:.2f})'}: {reason}"
        )
        return EvalSimulationMetricScore(
            name=metric.name,
            score=score,
            passed=passed,
            reason=reason,
            min_quality=metric.min_quality,
            verdicts=verdicts,
        )

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
