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
    EvalTrace,
)
from pipecat.evals.simulation import EvalSimulationScenario
from pipecat.frames.frames import FunctionCallResultProperties
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import FunctionCallParams, LLMService

# The event the driver appends when the persona calls end_call.
END_CALL_EVENT = "end_call"


class EvalSimulationDriver(BaseEvalDriver[EvalSimulationResult]):
    """Lets the persona LLM hold the conversation, then judges the whole of it.

    The persona runs inside the client's pipeline and answers the bot on its
    own. This driver watches the conversation, reporting each line as progress,
    for its end: the persona's ``end_call``, after which it hangs up, the bot
    ending the call, the cap on the persona's turns, or the wall-clock cap.
    Then it asks the judge, with the bot's function calls as evidence, whether
    the goal was achieved and how the conversation scored on each quality
    criterion.
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
        self._end_call: dict | None = None
        self._succeeded = False
        self._reason = ""
        self._metrics: list[EvalSimulationMetricScore] = []

    async def run(self) -> list[EvalAssertionFailure]:
        """Watch the conversation until it ends, then judge it."""
        self._persona_llm.register_function(END_CALL_FUNCTION, self._on_end_call)
        await self._client.configure_persona(self._persona.instruction)
        simulation = self._simulation
        # The bot's finished response: in audio mode the harness's transcription
        # of what it said, in text mode its LLM text.
        bot_said = "response" if simulation.bot_audio else "llm_response"
        deadline = time.monotonic() + simulation.max_duration_s
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
            if event["type"] == END_CALL_EVENT:
                self._ended_by = "end_call"
            elif event["type"] == BOT_ENDED_EVENT:
                self._ended_by = "bot"
            elif event["type"] == PERSONA_TURN_EVENT:
                self._turns += 1
                await self._report("user", event.get("text", ""))
                if self._turns >= simulation.max_turns:
                    self._ended_by = "max_turns"
            elif event["type"] == bot_said and event.get("text"):
                await self._report("bot", event["text"])
        # The persona has said its last word either way: nothing the bot says
        # from here on gets an answer.
        await self._client.hang_up()
        self._trace.log(f"persona: ended by {self._ended_by} after {self._turns} turn(s)")
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

    def conversation(self) -> list[dict]:
        """The conversation with the persona as ``user`` and the bot as ``assistant``.

        The persona's context holds it the other way round (the bot is what the
        persona LLM answers), so the roles are swapped for the judge and the
        result. Tool calls and results are left out.
        """
        swapped = {"user": "assistant", "assistant": "user"}
        messages = []
        for message in self._context.get_messages():
            if not isinstance(message, dict):
                continue
            role, content = message.get("role"), message.get("content")
            if role in swapped and isinstance(content, str) and content.strip():
                messages.append({"role": swapped[role], "content": content})
        return messages

    def tool_calls(self) -> list[str]:
        """The bot's function calls in order, one line each, as the judge's evidence.

        A call the bot cancelled is listed as such: it is evidence that the
        action did not happen.
        """
        lines = []
        for event in self._stream.events_seen:
            name = event.get("name") or "?"
            if event["type"] == "function_call":
                arguments = event.get("args") or {}
                lines.append(f"{name}({json.dumps(arguments) if arguments else ''})")
            elif event["type"] == "function_call_stopped" and (event.get("args") or {}).get(
                "cancelled"
            ):
                lines.append(f"{name} was cancelled")
        return lines

    async def _judge_conversation(self) -> None:
        """Decide the goal and score each quality criterion over the conversation."""
        if self._judge is None:
            self._reason = "no judge configured"
            return
        for message in self.conversation():
            if message["role"] == "user":
                self._judge.add_user_message(message["content"])
            else:
                self._judge.add_assistant_message(message["content"])
        evidence = self.tool_calls()
        if evidence:
            self._trace.log(f"judge: the bot's tool calls: {'; '.join(evidence)}")
        verdict = await self._judge.evaluate_conversation(
            self._simulation.success, evidence=evidence
        )
        self._succeeded = verdict.verdict == "yes"
        self._reason = verdict.reason
        self._trace.log(
            f"judge: goal {'achieved' if self._succeeded else 'not achieved'}: {verdict.reason}"
        )
        for metric in self._simulation.metrics:
            verdict = await self._judge.evaluate_conversation(metric.criterion, evidence=evidence)
            score = 1.0 if verdict.verdict == "yes" else 0.0
            self._metrics.append(
                EvalSimulationMetricScore(
                    name=metric.name, score=score, reason=verdict.reason, weight=metric.weight
                )
            )
            self._trace.log(f"judge: {metric.name} = {score:g}: {verdict.reason}")

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
        weights = sum(m.weight for m in self._metrics)
        quality = sum(m.score * m.weight for m in self._metrics) / weights if weights else None
        return EvalSimulationResult(
            simulation_name=self._simulation.name,
            succeeded=self._succeeded and error is None,
            reason=error or self._reason,
            error=error,
            quality=quality,
            metrics=self._metrics,
            messages=self.conversation(),
            turns=self._turns,
            ended_by=self._ended_by or "error",
            end_call=self._end_call,
            duration_ms=duration_ms,
            events_seen=events_seen,
            debug_log=debug_log,
        )
