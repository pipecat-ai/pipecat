#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Live checks of the token usage LLM services report, against the real provider APIs.

Each service answers one prompt in a real pipeline, and the usage it reports must
follow Pipecat's convention: ``completion_tokens`` holds every generated token,
``reasoning_tokens`` is the part spent reasoning, and ``total_tokens`` is the input
plus the output. Services whose models reason are asked to, so their reasoning
count is checked too.

The checks call paid APIs, so they run only when ``PIPECAT_LIVE_TESTS`` is set, and
each one only when its service's credentials are in the environment or the
repository's ``.env``::

    PIPECAT_LIVE_TESTS=1 uv run pytest tests/test_llm_token_usage_live.py

Credentials read from ``.env`` are set only for the check that needs them.

A failure means either Pipecat misreads the provider's usage or the provider changed
what it reports. A model the provider has retired fails as an error from the service;
update the model named in the case.
"""

import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from dotenv import dotenv_values

from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.frames.frames import ErrorFrame, LLMContextFrame, MetricsFrame
from pipecat.metrics.metrics import LLMTokenUsage, LLMUsageMetricsData
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import LLMService
from pipecat.tests.utils import SleepFrame, run_test

pytestmark = pytest.mark.skipif(
    not os.getenv("PIPECAT_LIVE_TESTS"), reason="PIPECAT_LIVE_TESTS not set"
)

_DOTENV_PATH = Path(__file__).resolve().parent.parent / ".env"

REASONING_PROMPT = (
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
    "How much does the ball cost? Think it through, then answer in one short sentence."
)
CODE_EXECUTION_PROMPT = (
    "Use code execution to compute the sum of the first 50 prime numbers. "
    "Reply with just the number."
)


@dataclass
class _Case:
    """One service to check.

    Parameters:
        id: Test id.
        env: Environment variables the service needs; the case skips without them.
        optional_env: Environment variables the service uses when they are set.
        make: Builds the service.
        expects_reasoning: Whether the model reasons and the provider reports it,
            so ``reasoning_tokens`` must be positive.
        net_prompt: Whether ``prompt_tokens`` is net of the cache counts, so the
            total adds them back.
        prompt: What the user asks.
        tools: Provider-specific tools to offer the model.
        wait: Seconds to wait for a response that arrives outside
            ``process_frame``, as on a realtime connection.
        start_timeout: Seconds to wait for the pipeline to start.
        checks_provider_total: Whether the provider's own total is reliable
            and must match the reported one. Applies to ``GoogleLLMService``,
            which computes its total rather than passing Google's through.
    """

    id: str
    env: tuple[str, ...]
    make: Callable[[], LLMService]
    optional_env: tuple[str, ...] = ()
    expects_reasoning: bool = False
    net_prompt: bool = False
    prompt: str = REASONING_PROMPT
    tools: dict[AdapterType, list[dict[str, Any]]] = field(default_factory=dict)
    wait: float = 0.0
    start_timeout: float = 5.0
    checks_provider_total: bool = False


def _openai_compatible(module: str, cls: str, env: str, **kwargs) -> Callable[[], LLMService]:
    """Build an OpenAI-compatible service with its default model."""

    def make():
        service_cls = getattr(importlib.import_module(module), cls)
        return service_cls(api_key=os.environ[env], **kwargs)

    return make


def _openai():
    from pipecat.services.openai.llm import OpenAILLMService

    return OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(model="gpt-5-mini"),
    )


def _openai_responses():
    from pipecat.services.openai.responses.llm import (
        OpenAIResponsesLLMService,
        OpenAIResponsesReasoningConfig,
    )

    return OpenAIResponsesLLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAIResponsesLLMService.Settings(
            model="gpt-5-mini", reasoning=OpenAIResponsesReasoningConfig(effort="low")
        ),
    )


def _openai_responses_http():
    from pipecat.services.openai.responses.llm import (
        OpenAIResponsesHttpLLMService,
        OpenAIResponsesReasoningConfig,
    )

    return OpenAIResponsesHttpLLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAIResponsesHttpLLMService.Settings(
            model="gpt-5-mini", reasoning=OpenAIResponsesReasoningConfig(effort="low")
        ),
    )


def _azure():
    from pipecat.services.azure.llm import AzureLLMService

    return AzureLLMService(
        api_key=os.environ["AZURE_CHATGPT_API_KEY"],
        endpoint=os.environ["AZURE_CHATGPT_ENDPOINT"],
        settings=AzureLLMService.Settings(model=os.environ["AZURE_CHATGPT_MODEL"]),
    )


def _grok():
    from pipecat.services.xai.llm import GrokLLMService

    return GrokLLMService(
        api_key=os.environ["XAI_API_KEY"],
        settings=GrokLLMService.Settings(model="grok-4.6", reasoning_effort="low"),
    )


def _anthropic():
    from pipecat.services.anthropic.llm import AnthropicLLMService

    return AnthropicLLMService(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        settings=AnthropicLLMService.Settings(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            thinking=AnthropicLLMService.ThinkingConfig(type="enabled", budget_tokens=1024),
        ),
    )


def _bedrock():
    from pipecat.services.aws.llm import AWSBedrockLLMService

    return AWSBedrockLLMService(
        aws_access_key=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
        aws_region=os.environ["AWS_REGION"],
    )


def _google(model: str) -> Callable[[], LLMService]:
    def make():
        from pipecat.services.google.llm import GoogleLLMService

        return GoogleLLMService(
            api_key=os.environ["GOOGLE_API_KEY"],
            settings=GoogleLLMService.Settings(model=model),
        )

    return make


def _gemini_live():
    from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService

    return GeminiLiveLLMService(
        api_key=os.environ["GOOGLE_API_KEY"],
        settings=GeminiLiveLLMService.Settings(
            model="models/gemini-3.8-live-extended-thinking",
            thinking={"thinking_level": "HIGH"},
        ),
    )


def _compat(name: str, env: str) -> _Case:
    """A case for an OpenAI-compatible service run with its default model."""
    return _Case(
        id=name,
        env=(env,),
        make=_openai_compatible(f"pipecat.services.{name}.llm", _COMPAT_CLASSES[name], env),
    )


_COMPAT_CLASSES = {
    "baseten": "BasetenLLMService",
    "cerebras": "CerebrasLLMService",
    "crusoe": "CrusoeLLMService",
    "deepseek": "DeepSeekLLMService",
    "fireworks": "FireworksLLMService",
    "groq": "GroqLLMService",
    "inception": "InceptionLLMService",
    "mistral": "MistralLLMService",
    "nebius": "NebiusLLMService",
    "novita": "NovitaLLMService",
    "nvidia": "NvidiaLLMService",
    "openrouter": "OpenRouterLLMService",
    "perplexity": "PerplexityLLMService",
    "qwen": "QwenLLMService",
    "sambanova": "SambaNovaLLMService",
    "sarvam": "SarvamLLMService",
    "together": "TogetherLLMService",
}

CASES = [
    _Case("openai", ("OPENAI_API_KEY",), _openai, expects_reasoning=True),
    _Case("openai_responses", ("OPENAI_API_KEY",), _openai_responses, expects_reasoning=True),
    _Case(
        "openai_responses_http",
        ("OPENAI_API_KEY",),
        _openai_responses_http,
        expects_reasoning=True,
    ),
    _Case(
        "azure",
        ("AZURE_CHATGPT_API_KEY", "AZURE_CHATGPT_ENDPOINT", "AZURE_CHATGPT_MODEL"),
        _azure,
    ),
    _Case("grok", ("XAI_API_KEY",), _grok, expects_reasoning=True),
    _Case(
        "anthropic",
        ("ANTHROPIC_API_KEY",),
        _anthropic,
        expects_reasoning=True,
        net_prompt=True,
    ),
    _Case(
        "bedrock",
        ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"),
        _bedrock,
        optional_env=("AWS_SESSION_TOKEN",),
        net_prompt=True,
    ),
    _Case(
        "google",
        ("GOOGLE_API_KEY",),
        _google("gemini-2.5-pro"),
        expects_reasoning=True,
        checks_provider_total=True,
    ),
    _Case(
        "google_code_execution",
        ("GOOGLE_API_KEY",),
        _google("gemini-3.8-flash"),
        prompt=CODE_EXECUTION_PROMPT,
        tools={AdapterType.GEMINI: [{"code_execution": {}}]},
        checks_provider_total=True,
    ),
    _Case(
        "gemini_live",
        ("GOOGLE_API_KEY",),
        _gemini_live,
        expects_reasoning=True,
        wait=25.0,
        start_timeout=15.0,
    ),
    *(_compat(name, f"{name.upper()}_API_KEY") for name in _COMPAT_CLASSES),
]


def _record_google_totals(service: LLMService) -> list[int]:
    """Record the totals Google reports as GoogleLLMService streams a response."""
    totals: list[int] = []
    stream_content = service._stream_content  # type: ignore[attr-defined]

    async def recording_stream(context):
        stream = await stream_content(context)

        async def chunks():
            async for chunk in stream:
                if chunk.usage_metadata and chunk.usage_metadata.total_token_count:
                    totals.append(chunk.usage_metadata.total_token_count)
                yield chunk

        return chunks()

    service._stream_content = recording_stream  # type: ignore[attr-defined]
    return totals


async def _reported_usage(case: _Case) -> tuple[list[LLMTokenUsage], list[int]]:
    """Run the case's prompt through the service.

    Returns:
        The usage the service reported, and the totals the provider reported
        when the case checks them.
    """
    service = case.make()
    provider_totals = _record_google_totals(service) if case.checks_provider_total else []
    context = LLMContext(messages=[{"role": "user", "content": case.prompt}])
    if case.tools:
        context.set_tools(ToolsSchema(standard_tools=[], custom_tools=case.tools))
    frames = [LLMContextFrame(context)]
    if case.wait:
        frames.append(SleepFrame(sleep=case.wait))

    down, up = await run_test(
        service,
        frames_to_send=frames,
        pipeline_params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        start_timeout=case.start_timeout,
    )

    errors = [f.error for f in [*down, *up] if isinstance(f, ErrorFrame)]
    assert not errors, f"{case.id} reported errors: {errors}"
    reports = [
        data.value
        for frame in down
        if isinstance(frame, MetricsFrame)
        for data in frame.data
        if isinstance(data, LLMUsageMetricsData)
    ]
    return reports, provider_totals


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
async def test_reported_token_usage_follows_the_convention(case: _Case, monkeypatch):
    dotenv = dotenv_values(_DOTENV_PATH) if _DOTENV_PATH.exists() else {}
    missing = []
    for name in case.env + case.optional_env:
        value = os.getenv(name) or dotenv.get(name)
        if value:
            monkeypatch.setenv(name, value)
        elif name in case.env:
            missing.append(name)
    if missing:
        pytest.skip(f"{', '.join(missing)} not set")

    reports, provider_totals = await _reported_usage(case)

    assert reports, f"{case.id} reported no token usage"
    for usage in reports:
        assert usage.prompt_tokens > 0, usage
        assert usage.completion_tokens > 0, usage

        expected_total = usage.prompt_tokens + usage.completion_tokens
        if case.net_prompt:
            expected_total += (usage.cache_read_input_tokens or 0) + (
                usage.cache_creation_input_tokens or 0
            )
        assert usage.total_tokens == expected_total, usage

        if usage.reasoning_tokens is not None:
            assert usage.reasoning_tokens <= usage.completion_tokens, usage

    if case.checks_provider_total:
        # The last chunk's usage is the final count for the response.
        assert provider_totals, f"{case.id}: the provider reported no total"
        assert reports[-1].total_tokens == provider_totals[-1], (reports, provider_totals)

    if case.expects_reasoning:
        assert any(usage.reasoning_tokens for usage in reports), (
            f"{case.id} reported no reasoning tokens: {reports}"
        )
