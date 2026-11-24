#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A client tool affects generated text in its own response, never the audio transcript."""

import base64
import json
from unittest.mock import AsyncMock

import pytest
from google.genai import types

from pipecat.frames.frames import LLMTextFrame, LLMThoughtTextFrame, TTSAudioRawFrame, TTSTextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.azure.voice_live.llm import AzureVoiceLiveLLMService
from pipecat.services.google.frames import LLMSearchResponseFrame
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService, GeminiModalities
from pipecat.services.inworld.realtime.llm import InworldRealtimeLLMService
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.services.settings import ToolCallPolicy
from pipecat.services.ultravox.llm import OneShotInputParams, UltravoxRealtimeLLMService
from pipecat.services.xai.realtime.llm import GrokRealtimeLLMService

SUPPRESS = ToolCallPolicy.SUPPRESS_TEXT_AFTER_TOOL_CALL_DETECTED


async def _wire(events):
    for event in events:
        yield json.dumps({"event_id": "event", **event})


def _frames(service, frame_type):
    return [
        call.args[0]
        for call in service.push_frame.await_args_list
        if type(call.args[0]) is frame_type
    ]


def _texts(service, frame_type):
    return [frame.text for frame in _frames(service, frame_type)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider",
    [
        OpenAIRealtimeLLMService,
        GrokRealtimeLLMService,
        AzureVoiceLiveLLMService,
        InworldRealtimeLLMService,
    ],
)
@pytest.mark.parametrize("suppress", [False, True])
async def test_realtime_text_is_scoped_to_one_wire_response(provider, suppress):
    args = {"api_key": "test"}
    if provider is AzureVoiceLiveLLMService:
        args["endpoint"] = "https://test.services.ai.azure.com"
    if suppress:
        args["settings"] = provider.Settings(tool_call_text_policy=SUPPRESS)
    service = provider(**args)
    service.push_frame = AsyncMock()
    service.run_function_calls = AsyncMock()
    service._context = LLMContext()
    if provider is GrokRealtimeLLMService:
        service._ensure_audio_config(16000, 24000)

    def boundary(response_id, status, modalities=("text",)):
        return {
            "type": "response.created" if status == "in_progress" else "response.done",
            "response": {
                "id": response_id,
                "object": "realtime.response",
                "status": status,
                "output": [],
                "output_modalities": list(modalities),
            },
        }

    def fragment(response_id, text, kind="text"):
        prefix = "response." if provider is AzureVoiceLiveLLMService else "response.output_"
        return {
            "type": prefix + kind + ".delta",
            "response_id": response_id,
            "item_id": response_id + "-msg",
            "content_index": 0,
            "output_index": 0,
            "delta": text,
        }

    def item_fragment(response_id, text):
        event = fragment(response_id, text)
        if provider in (AzureVoiceLiveLLMService, GrokRealtimeLLMService):
            del event["response_id"]
        return event

    tool = {"id": "invocation", "type": "function_call", "call_id": "call", "name": "lookup"}
    arguments = {
        "response_id": "first",
        "output_index": 1,
        "item_id": "invocation",
        "call_id": "call",
    }
    pcm = b"\x01\x00" * 60
    service._websocket = _wire(
        [
            boundary("first", "in_progress"),
            {
                "type": "response.output_item.added",
                "response_id": "first",
                "output_index": 0,
                "item": {"type": "message", "id": "first-msg", "role": "assistant"},
            },
            fragment("first", "Before"),
            {"type": "response.function_call_arguments.delta", **arguments, "delta": "{"},
            fragment("first", "Argument fragment"),
            item_fragment("unknown", "Unrelated item"),
            {
                "type": "response.output_item.added",
                "response_id": "first",
                "output_index": 1,
                "item": tool,
            },
            {
                "type": "conversation.item.created"
                if provider is AzureVoiceLiveLLMService
                else "conversation.item.added",
                "item": tool,
            },
            boundary("second", "in_progress"),
            fragment("second", "Other response"),
            {
                "type": "response.function_call_arguments.done",
                **arguments,
                "arguments": "{}",
                "name": "lookup",
            },
            item_fragment("first", "After"),
            fragment("first", "Spoken words", "audio_transcript"),
            fragment("first", base64.b64encode(pcm).decode(), "audio"),
            boundary("first", "cancelled"),
            item_fragment("first", "Late"),
            fragment("second", "Continuation"),
            boundary("second", "completed"),
            fragment("second", "Late tool-free response"),
        ]
    )
    await service._receive_task_handler()
    expected = ["Before"]
    if not suppress:
        expected.append("Argument fragment")
    if provider is not InworldRealtimeLLMService:
        expected.append("Unrelated item")
    expected.append("Other response")
    if not suppress:
        expected.append("After")
    expected.append("Spoken words")
    if not suppress and provider is not InworldRealtimeLLMService:
        expected.append("Late")
    expected.append("Continuation")
    if provider is not InworldRealtimeLLMService:
        expected.append("Late tool-free response")
    assert _texts(service, LLMTextFrame) == expected
    assert _texts(service, TTSTextFrame) == ["Spoken words"]
    assert [frame.audio for frame in _frames(service, TTSAudioRawFrame)] == [pcm]
    service.run_function_calls.assert_awaited_once()


@pytest.mark.asyncio
async def test_inworld_default_audio_does_not_emit_the_llm_text_twice():
    service = InworldRealtimeLLMService(api_key="test")
    service.push_frame = AsyncMock()
    service._websocket = _wire(
        [
            {
                "type": "response.created",
                "response": {
                    "id": "audio",
                    "object": "realtime.response",
                    "status": "in_progress",
                    "output": [],
                },
            },
            {"type": "response.output_text.delta", "response_id": "audio", "delta": "Generated"},
            {
                "type": "response.output_audio_transcript.delta",
                "response_id": "audio",
                "item_id": "item",
                "delta": "Actually spoken",
            },
        ]
    )
    await service._receive_task_handler()
    assert _texts(service, LLMTextFrame) == _texts(service, TTSTextFrame) == ["Actually spoken"]


@pytest.mark.asyncio
@pytest.mark.parametrize("suppress", [False, True])
async def test_gemini_live_blocks_only_generated_text_and_resumes_after_tool_result(suppress):
    policy = {"tool_call_text_policy": SUPPRESS} if suppress else {}
    service = GeminiLiveLLMService(
        api_key="test",
        settings=GeminiLiveLLMService.Settings(
            model="models/gemini-3.1-flash-live-preview",
            modalities=GeminiModalities.TEXT,
            **policy,
        ),
    )
    service._context = LLMContext()
    service._session = AsyncMock()
    service.push_frame = AsyncMock()
    service.run_function_calls = AsyncMock()

    async def send(**fields):
        await service._handle_server_message(types.LiveServerMessage.model_validate(fields))

    async def text(value, *, thought=False):
        await send(server_content={"model_turn": {"parts": [{"text": value, "thought": thought}]}})

    await text("Before")
    await send(tool_call={"function_calls": [{"name": "lookup", "args": {}, "id": "tool"}]})
    await text("Hidden")
    await text("Reasoning", thought=True)
    await send(server_content={"output_transcription": {"text": "Actually spoken"}})
    await service._tool_result("tool", "lookup", {"result": 1})
    await text("Continuation")
    await send(
        server_content={
            "turn_complete": True,
            "grounding_metadata": {"web_search_queries": ["lookup"]},
        }
    )
    await text("Next turn")
    expected = ["Before"] if suppress else ["Before", "Hidden"]
    expected.extend(["Actually spoken", "Continuation"])
    assert _texts(service, LLMTextFrame) == [*expected, "Next turn"]
    assert _texts(service, TTSTextFrame) == ["Actually spoken"]
    assert _texts(service, LLMThoughtTextFrame) == ["Reasoning"]
    assert [frame.search_result for frame in _frames(service, LLMSearchResponseFrame)] == [
        "".join(expected)
    ]
    service.run_function_calls.assert_awaited_once()


@pytest.mark.asyncio
async def test_gemini_live_nonblocking_results_do_not_reopen_the_same_text_response():
    service = GeminiLiveLLMService(
        api_key="test",
        settings=GeminiLiveLLMService.Settings(
            model="models/gemini-3.8-live-extended-thinking",
            modalities=GeminiModalities.TEXT,
            tool_call_text_policy=SUPPRESS,
        ),
    )
    service._context = LLMContext()
    service._session = AsyncMock()
    service.push_frame = AsyncMock()
    service.run_function_calls = AsyncMock()
    service.broadcast_interruption = AsyncMock()

    async def send(**fields):
        await service._handle_server_message(types.LiveServerMessage.model_validate(fields))

    tool = {"function_calls": [{"name": "lookup", "args": {}, "id": "tool"}]}
    await send(tool_call=tool)
    await service._tool_result("tool", "lookup", {"result": 1})
    await send(server_content={"model_turn": {"parts": [{"text": "Hidden"}]}})
    await send(server_content={"interrupted": True})
    await send(server_content={"model_turn": {"parts": [{"text": "New turn"}]}})
    await send(tool_call=tool, server_content={"turn_complete": True})
    await send(server_content={"model_turn": {"parts": [{"text": "Next turn"}]}})
    assert _texts(service, LLMTextFrame) == ["New turn", "Next turn"]


@pytest.mark.asyncio
async def test_gemini_live_native_audio_and_output_transcription_survive_a_tool_call():
    service = GeminiLiveLLMService(
        api_key="test",
        settings=GeminiLiveLLMService.Settings(
            modalities=GeminiModalities.AUDIO, tool_call_text_policy=SUPPRESS
        ),
    )
    service._context = LLMContext()
    service.push_frame = AsyncMock()
    service.run_function_calls = AsyncMock()
    pcm = b"\x10\x00" * 60
    for fields in (
        {"tool_call": {"function_calls": [{"name": "lookup", "args": {}, "id": "tool"}]}},
        {
            "server_content": {
                "model_turn": {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "audio/pcm;rate=24000",
                                "data": base64.b64encode(pcm).decode(),
                            }
                        }
                    ]
                }
            }
        },
        {"server_content": {"output_transcription": {"text": "Spoken"}}},
        {"server_content": {"turn_complete": True}},
    ):
        await service._handle_server_message(types.LiveServerMessage.model_validate(fields))
    assert _texts(service, LLMTextFrame) == _texts(service, TTSTextFrame) == ["Spoken"]
    assert [frame.audio for frame in _frames(service, TTSAudioRawFrame)] == [pcm]


@pytest.mark.asyncio
@pytest.mark.parametrize("suppress", [False, True])
async def test_ultravox_distinguishes_late_text_ordinals_from_the_tool_continuation(suppress):
    service = UltravoxRealtimeLLMService(
        params=OneShotInputParams(api_key="test", system_prompt="test"),
        settings=UltravoxRealtimeLLMService.Settings(tool_call_text_policy=SUPPRESS)
        if suppress
        else None,
    )
    service.push_frame = AsyncMock()
    service.run_function_calls = AsyncMock()
    service._send = AsyncMock()
    pcm = b"\x01\x00" * 30

    def transcript(ordinal, medium, delta, final=False):
        return {
            "type": "transcript",
            "role": "agent",
            "medium": medium,
            "delta": delta,
            "ordinal": ordinal,
            "final": final,
        }

    async def socket():
        events = [
            transcript(1, "text", "Before"),
            {
                "type": "client_tool_invocation",
                "toolName": "lookup",
                "invocationId": "one",
                "parameters": {},
            },
            transcript(1, "text", "After"),
            transcript(2, "voice", "Spoken"),
        ]
        async for event in _wire(events):
            yield event
        yield pcm
        await service._send_tool_result("one", "result")
        async for event in _wire(
            [
                transcript(3, "text", "Continuation"),
                transcript(1, "text", "Late"),
                transcript(1, "text", "", final=True),
            ]
        ):
            yield event

    service._socket = socket()
    await service._receive_messages()
    assert _texts(service, LLMTextFrame) == (
        ["Before", "Spoken", "Continuation"]
        if suppress
        else ["Before", "After", "Spoken", "Continuation", "Late"]
    )
    assert _texts(service, TTSTextFrame) == ["Spoken"]
    assert [frame.audio for frame in _frames(service, TTSAudioRawFrame)] == [pcm]
    service.run_function_calls.assert_awaited_once()
