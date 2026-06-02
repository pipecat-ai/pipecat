#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Judge LLM for content assertions in behavioral evaluations.

Uses an LLM to decide whether a bot's response satisfies a natural-language
criterion (``judge: "describes the agent's capabilities"``). The judge runs as
a one-shot, out-of-pipeline inference via
:meth:`pipecat.services.openai.base_llm.BaseOpenAILLMService.run_inference`,
so it works with any pipecat LLM service backed by an OpenAI-compatible API
(OpenAI, Ollama, Together, etc.).

Verdicts are cached by ``(criterion, text)`` hash so that re-runs are stable
and so that a single scenario doesn't pay multiple judge round-trips for the
same assertion.

Example::

    from pipecat.services.ollama.llm import OLLamaLLMService

    service = OLLamaLLMService(settings=OLLamaLLMService.Settings(model="qwen2.5:3b"))
    judge = Judge(service)
    verdict = await judge.evaluate(
        criterion="describes the agent's capabilities",
        text="I can help with questions, set reminders, and look things up.",
    )
    if not verdict.passed:
        print(f"judge said no: {verdict.reason}")
"""

import hashlib
import json
import re
from dataclasses import dataclass

from loguru import logger

from pipecat.processors.aggregators.llm_context import LLMContext

JUDGE_SYSTEM_INSTRUCTION = (
    "You are a strict but fair judge deciding whether a bot's response satisfies a "
    "given criterion. The text may be a partial response that is still streaming in. "
    "Respond ONLY with a JSON object on a single line containing two fields: "
    '{"verdict": "yes" | "no" | "continue", "reason": "<one short sentence>"}. '
    'Use "yes" if the text satisfies the criterion. Use "no" if the text gives a '
    'substantive answer that fails it. Use "continue" if the text so far is only an '
    'interim or filler utterance (e.g. "Let me check on that.", a greeting, or an '
    "obviously incomplete fragment) that does not yet contain enough to decide — more "
    "text is expected. Do not include any other text, explanation, or markdown."
)

JUDGE_PROMPT_TEMPLATE = """Criterion: {criterion}

Text to evaluate:
\"\"\"
{text}
\"\"\"

Does the text satisfy the criterion? Answer yes, no, or continue."""


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


class Judge:
    """Wraps a pipecat LLM service and runs single-shot evaluations.

    Args:
        service: A pipecat LLM service with a ``run_inference()`` method
            (i.e. ``BaseOpenAILLMService`` or any subclass: OpenAI, Ollama, etc.).
        max_tokens: Cap on the judge's response length. Default 200 — enough
            for a JSON verdict + short reason.
    """

    def __init__(self, service, max_tokens: int = 200):
        """Initialize the judge with a configured pipecat LLM service.

        Args:
            service: A pipecat LLM service exposing ``run_inference()``.
            max_tokens: Cap on the judge's response length.
        """
        self._service = service
        self._max_tokens = max_tokens
        self._cache: dict[str, JudgeVerdict] = {}

    async def evaluate(self, criterion: str, text: str) -> JudgeVerdict:
        """Ask the judge whether ``text`` satisfies ``criterion``.

        Args:
            criterion: Natural-language description of what the text should express.
            text: The actual text to evaluate.

        Returns:
            A :class:`JudgeVerdict` with the pass/fail decision and a one-sentence
            justification. The verdict is cached by ``(criterion, text)`` so the
            same assertion within one run hits the judge only once.
        """
        key = _cache_key(criterion, text)
        if key in self._cache:
            return self._cache[key]

        verdict = await self._call_judge(criterion, text)
        self._cache[key] = verdict
        return verdict

    async def _call_judge(self, criterion: str, text: str) -> JudgeVerdict:
        """Single round-trip to the judge LLM."""
        prompt = JUDGE_PROMPT_TEMPLATE.format(criterion=criterion, text=text)
        context = LLMContext(messages=[{"role": "user", "content": prompt}])

        try:
            response = await self._service.run_inference(
                context=context,
                max_tokens=self._max_tokens,
                system_instruction=JUDGE_SYSTEM_INSTRUCTION,
            )
        except Exception as e:
            logger.error(f"Judge call failed: {e.__class__.__name__} ({e})")
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


def _cache_key(criterion: str, text: str) -> str:
    """Hash a (criterion, text) pair for cache lookup."""
    h = hashlib.sha256()
    h.update(criterion.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _parse_verdict(response: str) -> JudgeVerdict:
    """Parse the judge's response. Tolerant of extra whitespace and code fences."""
    cleaned = response.strip()
    # Strip markdown code fences if the judge ignored instructions
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()

    try:
        obj = json.loads(cleaned)
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


def build_default_judge(judge_config: dict | None) -> Judge | None:
    """Build a :class:`Judge` from a scenario's ``judge:`` config block.

    Args:
        judge_config: Mapping with keys ``service`` (default ``"ollama"``),
            ``model`` (default ``"qwen2.5:3b"``), and optional ``endpoint``
            (service-specific default if omitted). ``None`` uses all defaults.

    Returns:
        A configured Judge, or ``None`` if construction fails (caller decides
        whether to skip ``eval:`` assertions or fail the scenario).
    """
    config = judge_config or {}
    service_name = str(config.get("service", "ollama")).lower()

    try:
        if service_name == "ollama":
            llm_service = _ollama_service(config)
        elif service_name == "openai":
            llm_service = _openai_service(config)
        else:
            logger.error(f"Unknown judge service: {service_name!r}")
            return None
    except ImportError as e:
        logger.error(f"Failed to construct judge service {service_name!r}: {e}")
        return None

    return Judge(llm_service)


def _ollama_service(config: dict):
    """Build a local Ollama LLM service from the ``judge:`` config."""
    from pipecat.services.ollama.llm import OLLamaLLMService

    base_url = config.get("endpoint") or "http://localhost:11434/v1"
    return OLLamaLLMService(
        base_url=base_url,
        settings=OLLamaLLMService.Settings(model=config.get("model", "qwen2.5:3b")),
    )


def _openai_service(config: dict):
    """Build an OpenAI LLM service from the ``judge:`` config."""
    from pipecat.services.openai.llm import OpenAILLMService

    return OpenAILLMService(settings=OpenAILLMService.Settings(model=config.get("model", "gpt-4o")))
