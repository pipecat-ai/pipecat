#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""EvalJudge LLM for content assertions in behavioral evaluations.

Uses an LLM to decide whether a bot's response satisfies a natural-language
criterion (``judge: "describes the bot's capabilities"``). The judge runs as
a one-shot, out-of-pipeline inference via
:meth:`pipecat.services.openai.base_llm.BaseOpenAILLMService.run_inference`,
so it works with any pipecat LLM service backed by an OpenAI-compatible API
(OpenAI, Ollama, Together, etc.).

The judge keeps the conversation as an :class:`LLMContext`: the harness feeds it
the user turns and the bot's reply segments, and ``evaluate`` judges the most
recent reply against the criterion *in that context*. This lets it resolve a
terse or ambiguous reply (e.g. "That's four", which an STT pass might render as
"That's for") that wouldn't make sense in isolation.

Verdicts are cached by ``(criterion, conversation)`` hash so that re-runs are
stable and so that a single scenario doesn't pay multiple judge round-trips for
the same assertion.

Example::

    from pipecat.services.ollama.llm import OLLamaLLMService

    service = OLLamaLLMService(settings=OLLamaLLMService.Settings(model="gemma2:9b"))
    judge = EvalJudge(service)
    judge.add_user_message("What can you help me with?")
    judge.add_assistant_message("I can answer questions, set reminders, and look things up.")
    verdict = await judge.evaluate("describes the bot's capabilities")
    if not verdict.passed:
        print(f"judge said no: {verdict.reason}")
"""

import hashlib
import importlib
import json
import re
from dataclasses import dataclass
from typing import Any

from loguru import logger

from pipecat.evals.services import ollama_service, openai_service
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
    'Use "yes" if the reply satisfies the criterion. Use "no" if the reply gives a '
    'substantive answer that fails it. Use "continue" if the reply so far is only an '
    'interim or filler utterance (e.g. "Let me check on that.", a greeting, or an '
    "obviously incomplete fragment) that does not yet contain enough to decide — more "
    "text is expected. Do not include any other text, explanation, or markdown."
)

# Transient final user message appended for the judge call. The conversation it
# refers to ("the bot's most recent reply") is the LLMContext built up by the
# harness; this just poses the question and is never stored in that context.
JUDGE_ASK_TEMPLATE = (
    "Does the bot's most recent reply satisfy this criterion?\n\n"
    "Criterion: {criterion}\n\n"
    "Answer yes, no, or continue."
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

    @classmethod
    def from_config(cls, judge_config: dict | None) -> "EvalJudge":
        """Build an :class:`EvalJudge` from a scenario's ``judge.eval:`` config block.

        Honors a custom ``factory`` (dotted path to a callable taking ``(config)``
        and returning a pipecat LLM service with ``run_inference()``); otherwise
        dispatches on the ``service`` name (default ``"ollama"``). Add providers by
        extending this. To use a fully custom judge, construct ``EvalJudge``
        directly and pass it to :meth:`pipecat.evals.harness.EvalSession.from_scenario`.

        Args:
            judge_config: Mapping with keys ``service`` (default ``"ollama"``),
                ``model`` (default ``"gemma2:9b"``), and optional ``endpoint``
                (service-specific default if omitted). ``None`` uses all defaults.

        Returns:
            A configured EvalJudge.

        Raises:
            ValueError: If ``service`` is unknown (matching
                :meth:`pipecat.evals.speech.EvalSpeech.from_config` and
                :meth:`pipecat.evals.transcribe.EvalTranscriber.from_config`).

        Example::

            # In the scenario: judge.eval.factory: "my_pkg.make_judge_llm"
            def make_judge_llm(config):
                return TogetherLLMService(...)  # any service exposing run_inference()
        """
        config = judge_config or {}

        custom = config.get("factory")
        if custom:
            module_name, _, attr = custom.rpartition(".")
            if not module_name:
                raise ValueError(f"judge.eval.factory must be a dotted path: {custom!r}")
            factory = getattr(importlib.import_module(module_name), attr)
            return cls(factory(config))

        service_name = str(config.get("service", "ollama")).lower()
        if service_name == "ollama":
            llm_service = ollama_service(config)
        elif service_name == "openai":
            llm_service = openai_service(config)
        else:
            raise ValueError(
                f"Unknown judge service: {service_name!r}. Known: ollama, openai. "
                "Or set judge.eval.factory to a 'module.func' returning an LLM service."
            )

        return cls(llm_service)

    def add_user_message(self, text: str | None) -> None:
        """Record a user turn in the conversation the judge evaluates against.

        Called by the harness when it sends a user turn, so a later reply can be
        judged in context (e.g. a terse "That's four" after "What is two plus two?").

        Args:
            text: The user's utterance, or ``None`` for a bot-first turn (ignored).
        """
        if text and text.strip():
            self._context.add_message({"role": "user", "content": text})

    def add_assistant_message(self, text: str | None) -> None:
        """Append a streamed segment of the bot's current reply to the conversation.

        The bot's reply may arrive in several segments; each is added as its own
        ``assistant`` message, so the accumulated conversation is exactly what the
        judge sees — there is no separate "commit" step.

        Args:
            text: The new reply segment; empty or ``None`` is ignored.
        """
        if text and text.strip():
            self._context.add_message({"role": "assistant", "content": text})

    async def evaluate(self, criterion: str) -> JudgeVerdict:
        """Judge whether the bot's most recent reply satisfies ``criterion``.

        Evaluates the conversation built up via :meth:`add_user_message` and
        :meth:`add_assistant_message`. The judge's own answer is never written
        back into that conversation.

        Args:
            criterion: Natural-language description of what the reply should express.

        Returns:
            A :class:`JudgeVerdict` with the pass/fail decision and a one-sentence
            justification. Cached by ``(criterion, conversation)`` so the same
            assertion over the same conversation hits the judge only once.
        """
        messages = self._context.get_messages()
        key = _cache_key(criterion, messages)
        if key in self._cache:
            return self._cache[key]

        verdict = await self._call_judge(criterion, messages)
        self._cache[key] = verdict
        return verdict

    async def _call_judge(self, criterion: str, messages: list) -> JudgeVerdict:
        """Single round-trip to the judge LLM over the conversation + a verdict ask."""
        # Copy the conversation and append a transient verdict ask, so neither the
        # ask nor the judge's answer ever lands in the persistent context.
        context = LLMContext(messages=list(messages))
        context.add_message(
            {"role": "user", "content": JUDGE_ASK_TEMPLATE.format(criterion=criterion)}
        )

        # Log the conversation the judge is about to evaluate, before its verdict,
        # so the debug log shows exactly what the judge saw (handy when a terse or
        # mis-transcribed reply gets an unexpected verdict).
        transcript = "\n".join(f"  [{m.get('role')}] {m.get('content')}" for m in messages)
        logger.debug(
            "Judge evaluating {!r} over conversation:\n{}", criterion, transcript or "  (empty)"
        )

        try:
            response = await self._service.run_inference(
                context=context,
                max_tokens=self._max_tokens,
                system_instruction=JUDGE_SYSTEM_INSTRUCTION,
            )
        except Exception as e:
            logger.error(f"EvalJudge call failed: {e.__class__.__name__} ({e})")
            return JudgeVerdict(
                verdict="no",
                reason=f"judge call failed: {e.__class__.__name__}",
                raw_response="",
            )

        if not response:
            return JudgeVerdict(
                verdict="no", reason="judge returned empty response", raw_response=""
            )

        return _parse_verdict(response)


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
