#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the requests and usage of PerplexityAgentHttpLLMService."""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from loguru import logger
from openai.types.responses import (
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
    ResponseUsage,
)
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.perplexity.llm import PerplexityAgentHttpLLMService, _WithoutServerToolCalls


@contextmanager
def _captured_logs():
    """Collect INFO-and-above log records emitted within the block."""
    records = []
    sink_id = logger.add(lambda m: records.append(m.record), level="INFO")
    try:
        yield records
    finally:
        logger.remove(sink_id)


async def _request(
    service: PerplexityAgentHttpLLMService, context: LLMContext | None = None
) -> dict:
    context = context or LLMContext(messages=[{"role": "user", "content": "Hi"}])
    invocation_params = await service.get_llm_adapter().get_llm_invocation_params(context)
    return service._build_response_params(invocation_params)


@pytest.mark.asyncio
async def test_default_request_uses_the_fast_preset_with_a_pinned_model():
    params = await _request(PerplexityAgentHttpLLMService(api_key="test-key"))

    assert params["model"] == "openai/gpt-5.6-luna"
    assert params["extra_body"] == {"preset": "fast"}
    # The preset decides how much to reason.
    assert "reasoning" not in params
    assert params["tools"] == [{"type": "web_search"}]


@pytest.mark.asyncio
async def test_preset_alone_chooses_the_model():
    service = PerplexityAgentHttpLLMService(
        api_key="test-key",
        settings=PerplexityAgentHttpLLMService.Settings(model=None, preset="fast"),
    )
    params = await _request(service)

    assert "model" not in params
    assert params["extra_body"] == {"preset": "fast"}
    assert "reasoning" not in params


def test_neither_model_nor_preset_is_rejected():
    with pytest.raises(ValueError, match="needs a `model`, a `preset`"):
        PerplexityAgentHttpLLMService(
            api_key="test-key",
            settings=PerplexityAgentHttpLLMService.Settings(model=None),
        )


@pytest.mark.asyncio
async def test_update_leaving_neither_model_nor_preset_reports_an_error():
    service = PerplexityAgentHttpLLMService(api_key="test-key")
    service.push_error = AsyncMock()

    await service._update_settings(PerplexityAgentHttpLLMService.Settings(model=None))

    service.push_error.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    ["openai/gpt-5.5", "anthropic/claude-haiku-4-5", "google/gemini-3.8-flash", "xai/grok-4.6"],
)
async def test_every_model_gets_the_fast_preset(model):
    service = PerplexityAgentHttpLLMService(
        api_key="test-key", settings=PerplexityAgentHttpLLMService.Settings(model=model)
    )
    params = await _request(service)

    assert params["model"] == model
    assert params["extra_body"] == {"preset": "fast"}
    assert "reasoning" not in params


@pytest.mark.asyncio
async def test_web_search_joins_context_tools_once():
    weather = FunctionSchema(
        name="get_weather",
        description="Get the weather.",
        properties={"city": {"type": "string"}},
        required=["city"],
    )
    context = LLMContext(
        messages=[{"role": "user", "content": "Hi"}],
        tools=ToolsSchema(standard_tools=[weather]),
    )
    params = await _request(PerplexityAgentHttpLLMService(api_key="test-key"), context)

    assert [tool["type"] for tool in params["tools"]] == ["function", "web_search"]


@pytest.mark.asyncio
async def test_web_search_can_be_turned_off():
    service = PerplexityAgentHttpLLMService(
        api_key="test-key", settings=PerplexityAgentHttpLLMService.Settings(web_search=False)
    )

    assert "tools" not in await _request(service)


@pytest.mark.asyncio
async def test_perplexity_fields_join_extra_body():
    service = PerplexityAgentHttpLLMService(
        api_key="test-key",
        settings=PerplexityAgentHttpLLMService.Settings(
            max_steps=2, extra={"extra_body": {"language_preference": "en"}}
        ),
    )

    assert (await _request(service))["extra_body"] == {
        "language_preference": "en",
        "preset": "fast",
        "max_steps": 2,
    }


def test_usage_reports_cache_creation_tokens():
    # construct() mirrors the SDK's lenient decode of a streamed usage payload,
    # which keeps fields it doesn't know and skips ones Perplexity leaves out.
    usage = ResponseUsage.construct(
        input_tokens=3046,
        input_tokens_details=InputTokensDetails.construct(
            cached_tokens=1780, cache_creation_input_tokens=1263, cache_read_input_tokens=1780
        ),
        output_tokens=29,
        output_tokens_details=OutputTokensDetails.construct(reasoning_tokens=0),
        total_tokens=3075,
    )
    tokens = PerplexityAgentHttpLLMService(api_key="test-key")._token_usage(usage)

    assert tokens.prompt_tokens == 3046
    assert tokens.cache_read_input_tokens == 1780
    assert tokens.cache_creation_input_tokens == 1263
    assert tokens.total_tokens == 3075


@pytest.mark.asyncio
async def test_a_preset_keeps_the_model_unless_it_is_cleared():
    service = PerplexityAgentHttpLLMService(
        api_key="test-key", settings=PerplexityAgentHttpLLMService.Settings(preset="high")
    )
    assert (await _request(service))["model"] == "openai/gpt-5.6-luna"

    service = PerplexityAgentHttpLLMService(
        api_key="test-key",
        settings=PerplexityAgentHttpLLMService.Settings(preset="high", model=None),
    )
    assert "model" not in await _request(service)


@pytest.mark.asyncio
async def test_an_explicit_model_with_a_preset_keeps_the_preset_reasoning():
    service = PerplexityAgentHttpLLMService(
        api_key="test-key",
        settings=PerplexityAgentHttpLLMService.Settings(preset="high", model="openai/gpt-5.5"),
    )
    params = await _request(service)

    assert params["model"] == "openai/gpt-5.5"
    assert "reasoning" not in params


@pytest.mark.asyncio
async def test_explicit_reasoning_is_sent_as_configured():
    service = PerplexityAgentHttpLLMService(
        api_key="test-key",
        settings=PerplexityAgentHttpLLMService.Settings(
            reasoning=PerplexityAgentHttpLLMService.ReasoningConfig(effort="low")
        ),
    )

    params = await _request(service)

    assert params["reasoning"] == {"effort": "low"}
    assert "extra_body" not in params


@pytest.mark.asyncio
async def test_a_preset_in_extra_is_left_alone():
    service = PerplexityAgentHttpLLMService(
        api_key="test-key",
        settings=PerplexityAgentHttpLLMService.Settings(extra={"extra_body": {"preset": "high"}}),
    )

    assert (await _request(service))["extra_body"] == {"preset": "high"}


@pytest.mark.asyncio
async def test_a_preset_update_leaves_the_model_alone():
    service = PerplexityAgentHttpLLMService(api_key="test-key")

    await service._update_settings(PerplexityAgentHttpLLMService.Settings(preset="high"))

    assert service._settings.model == "openai/gpt-5.6-luna"


@pytest.mark.asyncio
async def test_provider_thinking_default_sends_no_preset():
    service = PerplexityAgentHttpLLMService(
        api_key="test-key",
        settings=PerplexityAgentHttpLLMService.Settings(use_provider_thinking_default=True),
    )
    params = await _request(service)

    assert params["model"] == "openai/gpt-5.6-luna"
    assert "extra_body" not in params
    assert "reasoning" not in params


@pytest.mark.asyncio
async def test_provider_thinking_default_keeps_a_chosen_preset():
    service = PerplexityAgentHttpLLMService(
        api_key="test-key",
        settings=PerplexityAgentHttpLLMService.Settings(
            preset="fast", use_provider_thinking_default=True
        ),
    )

    assert (await _request(service))["extra_body"] == {"preset": "fast"}


@pytest.mark.asyncio
async def test_the_default_preset_is_logged_once():
    service = PerplexityAgentHttpLLMService(api_key="test-key")

    with _captured_logs() as records:
        await _request(service)
        await _request(service)

    infos = [
        r["message"] for r in records if r["level"].name == "INFO" and "preset" in r["message"]
    ]
    assert len(infos) == 1
    assert "'preset': 'fast'" in infos[0]


@pytest.mark.asyncio
async def test_a_chosen_preset_is_not_logged():
    service = PerplexityAgentHttpLLMService(
        api_key="test-key", settings=PerplexityAgentHttpLLMService.Settings(preset="high")
    )

    with _captured_logs() as records:
        await _request(service)

    assert not [r for r in records if "preset" in r["message"]]


class _FakeStream:
    """Stand-in for openai.AsyncStream that yields preset events."""

    def __init__(self, events):
        self._events = list(events)
        self.closed = False
        self.iterator_closed = False

    async def _iterator(self):
        try:
            for event in self._events:
                yield event
        finally:
            self.iterator_closed = True

    def __aiter__(self):
        return self._iterator()

    async def close(self):
        self.closed = True


def _call_added(output_index, item_id, name):
    item = MagicMock(spec=ResponseFunctionToolCall)
    item.id = item_id
    item.name = name
    item.call_id = f"call_{item_id}"
    event = MagicMock(spec=ResponseOutputItemAddedEvent)
    event.output_index = output_index
    event.item = item
    return event


def _arguments_delta(output_index, item_id, delta):
    event = MagicMock(spec=ResponseFunctionCallArgumentsDeltaEvent)
    event.output_index = output_index
    event.item_id = item_id
    event.delta = delta
    return event


def _call_done(output_index, item_id, name, arguments):
    item = MagicMock(spec=ResponseFunctionToolCall)
    item.id = item_id
    item.name = name
    item.call_id = f"call_{item_id}"
    item.arguments = arguments
    event = MagicMock(spec=ResponseOutputItemDoneEvent)
    event.output_index = output_index
    event.item = item
    return event


def _search_results_done(output_index):
    event = MagicMock(spec=ResponseOutputItemDoneEvent)
    event.output_index = output_index
    event.item = MagicMock()  # a search_results item, with no id
    return event


def _text_delta(delta):
    event = MagicMock(spec=ResponseTextDeltaEvent)
    event.delta = delta
    return event


async def _filtered(*events):
    return [event async for event in _WithoutServerToolCalls(_FakeStream(events))]


@pytest.mark.asyncio
async def test_a_search_the_server_runs_is_dropped():
    search_added = _call_added(0, "fc_1", "search_web")
    search_done = _search_results_done(0)
    text = _text_delta("Sunny.")

    assert await _filtered(search_added, search_done, text) == [search_done, text]


@pytest.mark.asyncio
async def test_a_function_call_passes_through_in_order():
    added = _call_added(1, "fc_2", "get_weather")
    delta = _arguments_delta(1, "fc_2", '{"city": "Paris"}')
    text = _text_delta("Checking.")
    done = _call_done(1, "fc_2", "get_weather", '{"city": "Paris"}')

    # The announcement waits for the call's output to complete; other events don't.
    assert await _filtered(added, delta, text, done) == [delta, text, added, done]


@pytest.mark.asyncio
async def test_closing_the_filter_closes_the_stream():
    stream = _FakeStream([_text_delta("a"), _text_delta("b")])
    filtered = _WithoutServerToolCalls(stream)
    events = filtered.__aiter__()

    await events.__anext__()
    await events.aclose()
    await filtered.close()

    assert stream.iterator_closed
    assert stream.closed


@pytest.mark.asyncio
async def test_only_application_function_calls_run():
    service = PerplexityAgentHttpLLMService(api_key="test-key")
    service.start_ttfb_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()
    service.start_llm_usage_metrics = AsyncMock()
    service._push_llm_text = AsyncMock()
    service.run_function_calls = AsyncMock()
    adapter = MagicMock()
    adapter.get_messages_for_logging.return_value = []
    adapter.get_llm_invocation_params = AsyncMock(return_value={})
    service.get_llm_adapter = MagicMock(return_value=adapter)
    service._build_response_params = MagicMock(return_value={})
    service._client = AsyncMock()
    service._client.responses.create = AsyncMock(
        return_value=_FakeStream(
            [
                _call_added(0, "fc_1", "search_web"),
                _search_results_done(0),
                _call_added(1, "fc_2", "get_weather"),
                _call_done(1, "fc_2", "get_weather", '{"city": "Paris"}'),
            ]
        )
    )

    await service._process_context(MagicMock(spec=LLMContext))

    function_calls = service.run_function_calls.call_args.args[0]
    assert [fc.function_name for fc in function_calls] == ["get_weather"]
