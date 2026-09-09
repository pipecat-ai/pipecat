#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The judge: an LLM that decides whether the bot did what a scenario asks.

It answers a natural-language criterion, an ``eval:`` on a scripted turn or
a simulation's ``success:`` and metrics, with a one-shot inference outside
the pipeline, so any Pipecat LLM service with ``run_inference()`` works:
OpenAI, Ollama, Together, and others.

The judge keeps the conversation, so a terse reply ("That's four") is judged
in context. Verdicts are cached by criterion and conversation, so re-runs
are stable and a scenario never pays twice for the same question.

Example::

    from pipecat.services.ollama.llm import OLLamaLLMService

    service = OLLamaLLMService(settings=OLLamaLLMService.Settings(model="gemma4:12b"))
    judge = EvalJudge(service)
    judge.add_user_message("What can you help me with?")
    judge.add_assistant_message("I can answer questions, set reminders, and look things up.")
    verdict = await judge.evaluate("describes the bot's capabilities")
    if not verdict.passed:
        print(f"judge said no: {verdict.reason}")
"""

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from loguru import logger

from pipecat.evals.services import llm_service_from_config
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import LLMService

JUDGE_SYSTEM_INSTRUCTION = (
    "You are a strict but fair judge evaluating a conversation between a user and a "
    "bot under test. The 'user' messages are the user; the 'assistant' messages are "
    "the bot's replies. Judge only the bot's most recent reply — which may have "
    "arrived as several consecutive 'assistant' messages — against the given "
    "criterion, using the earlier turns only as context. The reply may still be "
    "streaming in. "
    "When the bot spoke its reply, the 'assistant' text is an automatic speech-to-text "
    "transcription, so it may contain homophones, misspellings, split or merged words, and "
    "missing punctuation. Always judge it by the intended spoken meaning, never by its exact "
    "spelling. In particular, treat a number as the same value whether it is spelled out, "
    "written as a digit, or transcribed as a homophone: 'for' and 'fore' mean 'four' (4), and "
    "'to' and 'too' mean 'two' (2). Never answer 'no' solely because of a transcription error "
    "when the intended spoken meaning satisfies the criterion. "
    "Respond ONLY with a JSON object on a single line containing two fields: "
    '{"verdict": "yes" | "no" | "continue", "reason": "<one short sentence>"}. '
    'Use "yes" if the reply satisfies the criterion. '
    'Use "continue" if the bot has not given its answer yet: it says it is checking, '
    "looking something up, fetching, working on something, or that it will report back. "
    "The answer is still coming, so there is nothing to judge yet. This holds however "
    'long and however fluent the reply is: "The system is checking the current '
    'conditions for you right now." is waiting, not answering. A greeting or an '
    'obviously incomplete fragment is also "continue". '
    'Use "no" only when the bot has given its answer and that answer fails the '
    'criterion. If the bot has not answered yet, always use "continue", never "no". '
    "Do not include any other text, explanation, or markdown."
)

# Transient final user message appended for the judge call. The conversation it
# refers to ("the bot's most recent reply") is the LLMContext built up by the
# harness; this just poses the question and is never stored in that context.
JUDGE_ASK_TEMPLATE = (
    "Does the bot's most recent reply satisfy this criterion?\n\n"
    "Criterion: {criterion}\n\n"
    "Answer yes, no, or continue."
)


RUN_JUDGE_SYSTEM_INSTRUCTION = (
    "You are a strict but fair judge evaluating a complete conversation between a user "
    "and a bot under test, given as a transcript. The bot's replies are numbered 'Bot "
    "turn 1', 'Bot turn 2', and so on; the user's lines are marked 'User'; a line "
    "marked '[tool call]' is a function the bot called at that point, and a completed "
    "call is stronger evidence of an action (a booking, a lookup) than the bot saying "
    "it did it. "
    "You are given criteria, each with a name, that every bot reply is judged against "
    "on its own, in the light of the conversation before it, and a goal that the "
    "conversation as a whole is judged against. A criterion that forbids something "
    "('never ...', 'does not ...') or that applies only in a situation ('when ...', "
    "'if ...') is satisfied by a reply that does not do the forbidden thing or is not "
    "in that situation; do not fault a reply for something the criterion does not ask "
    "of it. "
    "When the bot spoke its replies, its text is an automatic speech-to-text "
    "transcription, so it may contain homophones, misspellings, split or merged words, "
    "and missing punctuation. Always judge it by the intended spoken meaning, never by "
    "its exact spelling. "
    "Respond ONLY with a JSON object on a single line of the form "
    '{"goal": {"verdict": "yes" | "no", "reason": "<one short sentence>"}, '
    '"turns": {"<criterion name>": ["yes" | "no", ...]}, '
    '"reasons": {"<criterion name>": {"<bot turn number>": "<one short sentence>"}}}. '
    'Under "turns", give every criterion an array with exactly one entry per bot '
    'turn, in order. Under "reasons", give a reason only for the turns you '
    'answered "no". Do not include any other text, explanation, or markdown.'
)

RUN_JUDGE_ASK_TEMPLATE = (
    "Transcript:\n{transcript}\n\n"
    "Criteria for every bot reply:\n{criteria}\n\n"
    "Goal for the conversation as a whole: {success}\n\n"
    "Answer with the JSON described, one array entry per bot turn: there are "
    "{turn_count} bot turns."
)


@dataclass
class JudgeVerdict:
    """Outcome of a single judge call.

    Parameters:
        verdict: ``"yes"`` (satisfies), ``"no"`` (substantive answer that fails),
            or ``"continue"`` (interim/filler/incomplete — re-judge once more text
            arrives).
        reason: One-sentence justification.
        raw_response: The judge LLM's raw text, for diagnostics.
    """

    verdict: str
    reason: str
    raw_response: str

    @property
    def passed(self) -> bool:
        """True only when the verdict is a definite ``"yes"``."""
        return self.verdict == "yes"


@dataclass
class RunVerdicts:
    """A whole simulation run's verdicts, from one judge call.

    Parameters:
        goal: The verdict on the goal, over the whole conversation.
        turns: Per criterion name, a verdict per bot turn, in order.
    """

    goal: JudgeVerdict
    turns: dict[str, list[JudgeVerdict]]


class EvalJudge:
    """Wraps a pipecat LLM service and runs single-shot evaluations.

    Args:
        service: A pipecat LLM service with a ``run_inference()`` method
            (i.e. ``BaseOpenAILLMService`` or any subclass: OpenAI, Ollama, etc.).
        max_tokens: Cap on the judge's response length. Default 200 — enough
            for a JSON verdict + short reason.
    """

    def __init__(self, service: LLMService[Any], *, max_tokens: int = 200):
        """Initialize the judge with a configured pipecat LLM service.

        Args:
            service: A pipecat LLM service exposing ``run_inference()``.
            max_tokens: Cap on the judge's response length.
        """
        self._service = service
        self._max_tokens = max_tokens
        # The conversation the judge evaluates against, grown by the harness over
        # the scenario (one EvalJudge per scenario, so this starts empty).
        self._context = LLMContext()
        self._cache: dict[str, JudgeVerdict] = {}
        self._run_cache: dict[str, RunVerdicts] = {}

    @classmethod
    def from_config(cls, judge_config: dict | None) -> "EvalJudge":
        """Build a judge from a scenario's ``judge.eval:`` block.

        A ``factory`` (a dotted path to a callable taking the config) builds the
        LLM service; otherwise the ``service`` name picks a provider, ``ollama``
        by default. For a fully custom judge, construct ``EvalJudge`` directly
        and pass it to the session.

        Args:
            judge_config: Mapping with keys ``service`` (default ``"ollama"``),
                ``model`` (default ``"gemma4:12b"``), optional ``endpoint``
                (service-specific default if omitted), and an optional ``extra``
                mapping forwarded to the model as top-level request parameters.
                ``None`` uses all defaults.

        Returns:
            A configured EvalJudge.

        Raises:
            ValueError: If ``service`` is unknown (matching
                :func:`pipecat.evals.services.tts_service_from_config` and
                :func:`pipecat.evals.services.stt_service_from_config`).

        Example::

            # In the scenario: judge.eval.factory: "my_pkg.make_judge_llm"
            def make_judge_llm(config):
                return TogetherLLMService(...)  # any service exposing run_inference()
        """
        return cls(llm_service_from_config(judge_config, where="judge.eval"))

    def add_user_message(self, text: str | None) -> None:
        """Record a user turn, so a later reply is judged in context.

        Args:
            text: The user's utterance, or ``None`` for a bot-first turn (ignored).
        """
        if text and text.strip():
            self._context.add_message({"role": "user", "content": text})

    def add_assistant_message(self, text: str | None) -> None:
        """Add a segment of the bot's current reply to the conversation the judge sees.

        Args:
            text: The new reply segment; empty or ``None`` is ignored.
        """
        if text and text.strip():
            self._context.add_message({"role": "assistant", "content": text})

    async def evaluate(self, criterion: str) -> JudgeVerdict:
        """Judge whether the bot's latest reply satisfies ``criterion``, in the conversation so far.

        Args:
            criterion: Natural-language description of what the reply should express.

        Returns:
            A :class:`JudgeVerdict` with the pass/fail decision and a one-sentence
            justification. Cached by ``(criterion, conversation)`` so the same
            assertion over the same conversation hits the judge only once.
        """
        ask = JUDGE_ASK_TEMPLATE.format(criterion=criterion)
        return await self._evaluate(criterion, JUDGE_SYSTEM_INSTRUCTION, ask)

    async def evaluate_run(
        self, transcript: Sequence[dict], criteria: dict[str, str], success: str
    ) -> "RunVerdicts":
        """Judge a whole conversation in one call: every bot turn on every criterion, and the goal.

        The transcript goes in the question, turns numbered and tool calls inline.

        Args:
            transcript: The conversation in order: dicts with a ``role`` of
                ``assistant`` (a bot turn), ``user`` (the persona), or ``tool``
                (a tool call the bot made, one line as :func:`str`), and the
                ``content``.
            criteria: The per-turn criteria to decide, by name.
            success: The goal criterion, decided over the whole conversation.

        Returns:
            The goal's verdict and, per criterion, a verdict per bot turn in
            order; a turn the judge left out is a ``no``.
        """
        lines = []
        turn = 0
        for entry in transcript:
            if entry["role"] == "assistant":
                turn += 1
                lines.append(f"Bot turn {turn}: {entry['content']}")
            elif entry["role"] == "tool":
                lines.append(f"[tool call] {entry['content']}")
            else:
                lines.append(f"User: {entry['content']}")
        listed = "\n".join(f"- {name}: {criterion}" for name, criterion in criteria.items())
        ask = RUN_JUDGE_ASK_TEMPLATE.format(
            transcript="\n".join(lines) or "(nothing was said)",
            criteria=listed or "(none)",
            success=success,
            turn_count=turn,
        )
        key = _cache_key(ask, [])
        if key not in self._run_cache:
            # Room for a verdict per turn per criterion, a reason per "no", and
            # the goal's verdict; a budget sized for one verdict cuts it short.
            budget = max(300, 4 * turn * len(criteria) + 60 * len(criteria) + 80)
            response = await self._call_judge_text(
                success, [], RUN_JUDGE_SYSTEM_INSTRUCTION, ask, max_tokens=budget
            )
            self._run_cache[key] = _parse_run_verdicts(response, list(criteria), turn)
        return self._run_cache[key]

    async def _evaluate(self, criterion: str, instruction: str, ask: str) -> JudgeVerdict:
        messages = self._context.get_messages()
        key = _cache_key(ask, messages)
        if key in self._cache:
            return self._cache[key]
        verdict = await self._call_judge(criterion, messages, instruction, ask)
        self._cache[key] = verdict
        return verdict

    async def _call_judge(
        self, criterion: str, messages: list, instruction: str, ask: str
    ) -> JudgeVerdict:
        """Single round-trip to the judge LLM over the conversation + a verdict ask."""
        response = await self._call_judge_text(criterion, messages, instruction, ask)
        if response.startswith("\0"):
            return JudgeVerdict(verdict="no", reason=response[1:], raw_response="")
        return _parse_verdict(response)

    async def _call_judge_text(
        self,
        criterion: str,
        messages: list,
        instruction: str,
        ask: str,
        *,
        max_tokens: int | None = None,
    ) -> str:
        """The judge's raw answer to ``ask``.

        A failed or empty call comes back as a NUL-prefixed reason, which no
        answer starts with, so callers can report it as a ``no``.
        """
        # Copy the conversation and append the transient ask, so neither the ask
        # nor the judge's answer ever lands in the persistent context.
        context = LLMContext(messages=list(messages))
        context.add_message({"role": "user", "content": ask})

        # Log the conversation the judge is about to evaluate, before its verdict,
        # so the debug log shows exactly what the judge saw (handy when a terse or
        # mis-transcribed reply gets an unexpected verdict).
        # A run-level ask carries the transcript itself, so that is what to show.
        transcript = "\n".join(f"  [{m.get('role')}] {m.get('content')}" for m in messages)
        logger.debug(
            "Judge evaluating {!r} over conversation:\n{}",
            criterion,
            transcript or "\n".join(f"  {line}" for line in ask.splitlines()),
        )

        try:
            response = await self._service.run_inference(
                context=context,
                max_tokens=self._max_tokens if max_tokens is None else max_tokens,
                system_instruction=instruction,
            )
        except Exception as e:
            logger.error(f"EvalJudge call failed: {e.__class__.__name__} ({e})")
            return f"\0judge call failed: {e.__class__.__name__}"

        if not response:
            return "\0judge returned empty response"

        return response


# The reason a verdict carries when the judge gave none.
_NO_VERDICT = "(judge gave no verdict)"


def _parse_run_verdicts(response: str, names: list[str], turn_count: int) -> RunVerdicts:
    """Parse the run judge's answer into the goal's verdict and one per turn per criterion.

    Anything missing or malformed is a ``no`` with a reason, and the raw
    answer is logged, so a bad answer never passes a turn silently.
    """
    if response.startswith("\0"):
        failed = JudgeVerdict(verdict="no", reason=response[1:], raw_response="")
        return RunVerdicts(goal=failed, turns={n: [failed] * turn_count for n in names})
    obj = _judge_json(response)
    goal = obj.get("goal")
    if not isinstance(goal, dict):
        goal = {}
    goal_verdict = "yes" if str(goal.get("verdict", "")).strip().lower() == "yes" else "no"
    goal_reason = str(goal.get("reason", "")).strip()
    if goal_verdict == "no" and not goal_reason:
        goal_reason = _NO_VERDICT
    return RunVerdicts(
        goal=JudgeVerdict(verdict=goal_verdict, reason=goal_reason, raw_response=response),
        turns={name: _turn_verdicts(obj, name, turn_count, response) for name in names},
    )


def _judge_json(response: str) -> dict:
    """The JSON object in the judge's answer, or ``{}`` when there is none; a fenced or prefaced answer still parses."""
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()
    start = cleaned.find("{")
    if start != -1:
        try:
            parsed, _ = json.JSONDecoder().raw_decode(cleaned[start:])
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, AttributeError):
            pass
    logger.warning(f"Judge answer was not the expected JSON: {response!r}")
    return {}


def _turn_verdicts(obj: dict, name: str, turn_count: int, response: str) -> list[JudgeVerdict]:
    """One verdict per bot turn for criterion ``name``; a turn the judge left out is a ``no``.

    Criterion names match case-insensitively; a ``reasons`` entry, keyed by
    the turn number, gives a ``no`` its reason.
    """
    turns_by_name = {
        str(k).lower(): v for k, v in (obj.get("turns") or {}).items() if isinstance(v, list)
    }
    reasons_by_name = {
        str(k).lower(): v for k, v in (obj.get("reasons") or {}).items() if isinstance(v, dict)
    }
    answers = turns_by_name.get(name.lower(), [])
    reasons = reasons_by_name.get(name.lower(), {})
    if len(answers) != turn_count:
        logger.warning(
            f"Judge gave {len(answers)} verdict(s) for {name!r} over {turn_count} bot "
            f"turn(s); its answer was: {response!r}"
        )
    verdicts = []
    for index in range(turn_count):
        answer = answers[index] if index < len(answers) else None
        if isinstance(answer, dict):
            answer = answer.get("verdict")
        if answer is None:
            verdicts.append(JudgeVerdict(verdict="no", reason=_NO_VERDICT, raw_response=response))
            continue
        verdict = "yes" if str(answer).strip().lower() == "yes" else "no"
        reason = str(reasons.get(str(index + 1), "")).strip()
        if verdict == "no" and not reason:
            reason = "(no reason given)"
        verdicts.append(JudgeVerdict(verdict=verdict, reason=reason, raw_response=response))
    return verdicts


def _cache_key(criterion: str, messages: list) -> str:
    """Hash a (criterion, conversation) pair for cache lookup."""
    h = hashlib.sha256()
    h.update(criterion.encode("utf-8"))
    h.update(b"\x00")
    h.update(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return h.hexdigest()


def _parse_verdict(response: str) -> JudgeVerdict:
    """Parse the judge's response. Tolerant of extra whitespace and code fences."""
    cleaned = response.strip()
    # Strip markdown code fences if the judge ignored instructions
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()

    # Parse the first JSON object and ignore anything around it. Some judge models
    # ignore "respond ONLY with JSON" and wrap the verdict in prose (e.g. a trailing
    # "Let me know if you'd like to evaluate further turns!"); raw_decode from the
    # first '{' parses the object and stops, leaving the trailing text out.
    start = cleaned.find("{")
    if start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(cleaned[start:])
            verdict = str(obj.get("verdict", "")).strip().lower()
            if verdict not in ("yes", "no", "continue"):
                verdict = "no"
            reason = str(obj.get("reason", "")).strip()
            return JudgeVerdict(
                verdict=verdict,
                reason=reason or "(no reason given)",
                raw_response=response,
            )
        except (json.JSONDecodeError, AttributeError):
            pass

    # Fallback: scan for a verdict keyword in the raw text.
    lowered = cleaned.lower()
    if "continue" in lowered:
        return JudgeVerdict(
            verdict="continue", reason="(unstructured continue)", raw_response=response
        )
    if "yes" in lowered and "no" not in lowered:
        return JudgeVerdict(verdict="yes", reason="(unstructured yes)", raw_response=response)
    if "no" in lowered and "yes" not in lowered:
        return JudgeVerdict(verdict="no", reason="(unstructured no)", raw_response=response)
    return JudgeVerdict(
        verdict="no",
        reason=f"could not parse judge response: {response!r}",
        raw_response=response,
    )
