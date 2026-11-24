#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Spoken text and client tool results from the public streaming services."""

import asyncio
import json
from functools import partial
from unittest.mock import AsyncMock, patch

import pytest
from anthropic.types.beta import BetaRawMessageStreamEvent
from google.genai.types import (
    Candidate,
    Content,
    FunctionCall,
    GenerateContentResponse,
    GroundingMetadata,
    Part,
)
from openai import AsyncOpenAI
from pydantic import TypeAdapter

from pipecat.adapters.llm.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.aws.llm import AWSBedrockLLMService
from pipecat.services.google.frames import LLMSearchResponseFrame
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.nvidia.llm import NvidiaLLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.responses.llm import (
    OpenAIResponsesHttpLLMService,
    OpenAIResponsesLLMService,
)
from pipecat.services.sambanova.llm import SambaNovaLLMService
from pipecat.services.settings import ToolCallPolicy
from pipecat.tests.utils import run_test
from tests import test_google_interactions_llm as interactions
from tests.openai_http_helpers import ASYNC_CLIENT, http
from tests.test_openai_responses_websocket import _ws_events

BEFORE = "Let me check."
AFTER = " I'm still checking."
ANSWER = "It is sunny in Paris."
TURNS = ("call_1", None, "call_2", None)
ARGUMENTS = '{"city":"Paris"}'
SUPPRESS = ToolCallPolicy.SUPPRESS_TEXT_AFTER_TOOL_CALL_DETECTED


def _http_service(cls, client_factory, turns):
    responses = iter(turns)

    def reply(request):
        data = "".join(f"data: {json.dumps(event)}\n\n" for event in next(responses))
        return http.Response(200, headers={"Content-Type": "text/event-stream"}, content=data)

    client = AsyncOpenAI(
        api_key="test", http_client=ASYNC_CLIENT(transport=http.MockTransport(reply))
    )
    with patch.object(cls, client_factory, return_value=client):
        return cls(api_key="test")


def _chat_turn(call_id):
    deltas = [{"content": ANSWER}]
    if call_id:

        def tool(**fields):
            return {"tool_calls": [{"index": 0, **fields}]}

        deltas = [
            tool(id=call_id, type="function"),
            {"content": BEFORE},
            {
                **tool(function={"name": "get_weather", "arguments": '{"city":'}),
                "content": "Tool payload.",
            },
            {"content": AFTER},
            tool(function={"arguments": '"Paris"}'}),
            tool(
                index=1,
                id=f"{call_id}_tokyo",
                function={"name": "get_weather", "arguments": '{"city":"Tokyo"}'},
            ),
        ]
    return _chat_chunks(deltas)


def _chat_chunks(deltas):
    chunk = {"id": "chat", "object": "chat.completion.chunk", "model": "test", "created": 0}
    return [{**chunk, "choices": [{"index": 0, "delta": delta}]} for delta in deltas]


def _chat_service(cls):
    return _http_service(cls, "create_client", map(_chat_turn, TURNS))


def _responses_turn(call_id):
    def text(value):
        return {"type": "response.output_text.delta", "delta": value}

    if call_id:
        item = {"type": "function_call", "id": "fc_1", "call_id": call_id, "name": "get_weather"}
        events = [
            text(BEFORE),
            {"type": "response.output_item.added", "item": {**item, "arguments": ""}},
            text(AFTER),
            {
                "type": "response.function_call_arguments.done",
                "item_id": "fc_1",
                "arguments": ARGUMENTS,
            },
        ]
        finished = {**item, "arguments": ARGUMENTS}
    else:
        # A server-side search produces the answer within the same response.
        action = {"type": "search", "query": "Paris weather"}
        item = {"type": "web_search_call", "id": "ws_1", "action": action}
        events = [{"type": "response.output_item.added", "item": {**item, "status": "in_progress"}}]
        finished = {**item, "status": "completed"}
    events.append({"type": "response.output_item.done", "item": finished})
    if not call_id:
        events.append(text(ANSWER))
    content = {
        "type": "output_text",
        "text": BEFORE + AFTER if call_id else ANSWER,
        "annotations": [],
    }
    message = {"type": "message", "id": "msg", "role": "assistant", "content": [content]}
    response = {"id": f"resp_{call_id}", "model": "gpt-4.1", "output": [finished, message]}
    return [*events, {"type": "response.completed", "response": response}]


def _responses_http():
    return _http_service(
        OpenAIResponsesHttpLLMService, "_create_client", map(_responses_turn, TURNS)
    )


def _responses_ws(socket=None, **kwargs):
    service = OpenAIResponsesLLMService(api_key="test", **kwargs)
    if socket is None:
        socket = _ws_events(*(event for call_id in TURNS for event in _responses_turn(call_id)))
    service._websocket_connect = AsyncMock(return_value=socket)
    return service


def _anthropic():
    adapter = TypeAdapter(BetaRawMessageStreamEvent)

    def delta(index, kind, **fields):
        return {"type": "content_block_delta", "index": index, "delta": {"type": kind, **fields}}

    def turn(call_id):
        events = [delta(0, "text_delta", text=ANSWER)]
        if call_id:
            block = {"type": "tool_use", "id": call_id, "name": "get_weather", "input": {}}
            events = [
                delta(0, "text_delta", text=BEFORE),
                {"type": "content_block_start", "index": 1, "content_block": block},
                delta(1, "input_json_delta", partial_json=ARGUMENTS),
                {"type": "content_block_stop", "index": 1},
                delta(2, "text_delta", text=AFTER),
            ]
        stop = {"stop_reason": "tool_use" if call_id else "end_turn"}
        events.append({"type": "message_delta", "delta": stop, "usage": {"output_tokens": 10}})
        return interactions.as_stream([adapter.validate_python(event) for event in events])

    service = AnthropicLLMService(api_key="test")
    service._create_message_stream = AsyncMock(side_effect=map(turn, TURNS))
    return service


def _bedrock():
    def delta(index, **fields):
        return {"contentBlockDelta": {"contentBlockIndex": index, "delta": fields}}

    def turn(call_id):
        events = [delta(0, text=ANSWER)]
        if call_id:
            start = {"toolUse": {"toolUseId": call_id, "name": "get_weather"}}
            events = [
                delta(0, text=BEFORE),
                {"contentBlockStop": {"contentBlockIndex": 0}},
                {"contentBlockStart": {"contentBlockIndex": 1, "start": start}},
                delta(1, toolUse={"input": ARGUMENTS}),
                {"contentBlockStop": {"contentBlockIndex": 1}},
                delta(2, text=AFTER),
            ]
        return {"stream": interactions.as_stream(events)}

    service = AWSBedrockLLMService(
        aws_access_key="test", aws_secret_key="test", aws_region="us-east-1"
    )
    service._create_converse_stream = AsyncMock(side_effect=map(turn, TURNS))
    return service


def _google():
    def turn(call_id):
        parts = [Part(text=ANSWER)]
        if call_id:
            call = FunctionCall(id=call_id, name="get_weather", args={"city": "Paris"})
            parts = [Part(text=BEFORE), Part(function_call=call), Part(text=AFTER)]
        source = GroundingMetadata.model_validate(
            {"groundingChunks": [{"web": {"uri": "https://weather.example", "title": "Forecast"}}]}
        )
        chunk = GenerateContentResponse(
            candidates=[
                Candidate(content=Content(role="model", parts=parts), grounding_metadata=source)
            ]
        )
        return interactions.as_stream([chunk])

    service = GoogleLLMService(api_key="test")
    service._stream_content = AsyncMock(side_effect=map(turn, TURNS))
    return service


def _google_interactions():
    def turn(call_id):
        if call_id:
            return [
                *interactions.text_reply(BEFORE, 0),
                *interactions.function_call(1, call_id, "Paris"),
                *interactions.text_reply(AFTER, 2),
                interactions.completed("requires_action"),
            ]
        return [*interactions.text_reply(ANSWER), interactions.completed()]

    service = interactions.make_service()
    interactions.stub_client(service, *map(turn, TURNS))
    return service


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "make_service",
    [
        pytest.param(partial(_chat_service, OpenAILLMService), id="openai-chat"),
        pytest.param(partial(_chat_service, NvidiaLLMService), id="nvidia"),
        pytest.param(partial(_chat_service, SambaNovaLLMService), id="sambanova"),
        _responses_http,
        _responses_ws,
        _anthropic,
        _bedrock,
        _google,
        _google_interactions,
    ],
)
async def test_tool_call_text_policy(make_service):
    service = make_service()
    calls = []

    async def get_weather(params):
        calls.append((params.tool_call_id, params.arguments))
        await params.result_callback({"weather": "sunny"})

    service.register_function("get_weather", get_weather)
    tools = ToolsSchema(standard_tools=[interactions.WEATHER])
    if isinstance(service, (OpenAIResponsesHttpLLMService, OpenAIResponsesLLMService)):
        tools.custom_tools = {AdapterType.OPENAI: [{"type": "web_search"}]}
    context = interactions.user_context("Paris weather?", tools=tools)
    assistant = LLMAssistantAggregator(context)
    turns = []

    @assistant.event_handler("on_assistant_turn_stopped")
    async def ended(aggregator, message):
        turns.append(message.content)
        if len(turns) == 2:
            context.add_message({"role": "user", "content": "How about tomorrow?"})
            await aggregator.pipeline_worker.queue_frames(
                [
                    LLMUpdateSettingsFrame(delta=service.Settings(tool_call_text_policy=SUPPRESS)),
                    LLMContextFrame(context),
                ]
            )
        elif len(turns) == 4:
            await aggregator.pipeline_worker.queue_frame(EndFrame())

    async with asyncio.timeout(5):
        down, up = await run_test(
            Pipeline([service, assistant]),
            frames_to_send=[LLMContextFrame(context)],
            send_end_frame=False,
        )

    assert [f.error for f in up if isinstance(f, ErrorFrame)] == []
    assert turns == [BEFORE + AFTER, ANSWER, BEFORE, ANSWER]
    if make_service is _google:
        assert [f.search_result for f in down if isinstance(f, LLMSearchResponseFrame)] == turns
    expected = [(call_id, {"city": "Paris"}) for call_id in ("call_1", "call_2")]
    if isinstance(service, (OpenAILLMService, SambaNovaLLMService)):
        expected.extend((f"{call_id}_tokyo", {"city": "Tokyo"}) for call_id in ("call_1", "call_2"))
    assert sorted(calls) == sorted(expected)
    results = [json.loads(m["content"]) for m in context.messages if m["role"] == "tool"]
    assert results == [{"weather": "sunny"}] * len(expected)


async def _text_around_chat_tool(cls, before, after):
    call = {"index": 0, "id": "call", "function": {"name": "lookup", "arguments": "{}"}}
    events = _chat_chunks([before, {"tool_calls": [call]}, after])
    service = _http_service(cls, "create_client", [events])
    await service._update_settings(service.Settings(tool_call_text_policy=SUPPRESS))
    service.push_frame = AsyncMock()
    service.run_function_calls = AsyncMock()
    await service._process_context(interactions.user_context())
    frames = [call.args[0] for call in service.push_frame.await_args_list]
    return [frame.text for frame in frames if isinstance(frame, LLMTextFrame)]


@pytest.mark.asyncio
@pytest.mark.parametrize(("before", "after"), [("", "<th"), ("<", "x")])
async def test_nvidia_partial_think_tag_at_a_tool_boundary(before, after):
    texts = await _text_around_chat_tool(NvidiaLLMService, {"content": before}, {"content": after})
    assert "".join(texts) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", [OpenAILLMService, SambaNovaLLMService])
async def test_chat_suppression_keeps_the_native_audio_transcript(cls):
    audio = {"content": "Generated", "audio": {"transcript": "Spoken"}}
    assert await _text_around_chat_tool(cls, {"content": BEFORE}, audio) == [BEFORE, "Spoken"]


@pytest.mark.asyncio
async def test_interruption_drops_the_unfinished_tool_and_preserves_the_next_reply():
    ws = _ws_events()
    service = _responses_ws(
        ws, settings=OpenAIResponsesLLMService.Settings(tool_call_text_policy=SUPPRESS)
    )
    handler = AsyncMock()
    service.register_function("get_weather", handler)
    first = [{**event, "stream_id": "lane-0"} for event in _responses_turn("incomplete")]
    fresh = [{**event, "stream_id": "lane-1"} for event in _responses_turn(None)]
    # Stop recv after the client tool was announced, with its arguments still in flight.
    events = iter([*first[:2], None, *first[2:], *fresh])

    async def recv():
        event = next(events)
        if event is None:
            await service.pipeline_worker.queue_frame(InterruptionFrame())
            await asyncio.Future()
        return json.dumps(event)

    ws.recv.side_effect = recv

    @service.event_handler("on_after_process_frame")
    async def after_frame(_, frame):
        if isinstance(frame, InterruptionFrame):
            await service.pipeline_worker.queue_frames(
                [
                    LLMContextFrame(interactions.user_context("Continue.")),
                    EndFrame(),
                ]
            )

    async with asyncio.timeout(5):
        down, up = await run_test(
            service,
            frames_to_send=[LLMContextFrame(interactions.user_context())],
            send_end_frame=False,
        )

    assert [f.error for f in up if isinstance(f, ErrorFrame)] == []
    assert [f.text for f in down if isinstance(f, LLMTextFrame)] == [BEFORE, ANSWER]
    handler.assert_not_awaited()
