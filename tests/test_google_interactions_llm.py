#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for GoogleInteractionsLLMService, against a stubbed Interactions API stream."""

import asyncio
import base64
import io
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from google.genai.interactions import (
    InteractionCompletedEvent,
    StepDelta,
    StepStart,
    StepStop,
)
from google.genai.types import HarmBlockThreshold, HarmCategory, SafetySetting
from loguru import logger

import pipecat.services.google.interactions.llm as interactions_llm
from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    ErrorFrame,
    FunctionCallsStartedFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    LLMThoughtEndFrame,
    LLMThoughtStartFrame,
    LLMThoughtTextFrame,
    MetricsFrame,
)
from pipecat.metrics.metrics import LLMUsageMetricsData
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.google.interactions.llm import GoogleInteractionsLLMService
from pipecat.tests.utils import run_test

SIGNATURE = b"\x12\x34signature"

WEATHER = FunctionSchema(
    name="get_weather",
    description="Get the weather for a city.",
    properties={"city": {"type": "string"}},
    required=["city"],
)


def start(index: int, step: dict) -> StepStart:
    return StepStart.model_validate({"index": index, "step": step, "event_type": "step.start"})


def delta(index: int, data: dict) -> StepDelta:
    return StepDelta.model_validate({"index": index, "delta": data, "event_type": "step.delta"})


def stop(index: int) -> StepStop:
    return StepStop.model_validate({"index": index, "event_type": "step.stop"})


def completed(status: str = "completed", **usage) -> InteractionCompletedEvent:
    return InteractionCompletedEvent.model_validate(
        {
            "interaction": {
                "id": "",
                "status": status,
                "model": "gemini-3.6-flash",
                "object": "interaction",
                "usage": usage or None,
            },
            "event_type": "interaction.completed",
        }
    )


def signed_thought(index: int = 0) -> list:
    return [
        start(index, {"type": "thought"}),
        delta(
            index,
            {"type": "thought_signature", "signature": base64.b64encode(SIGNATURE).decode()},
        ),
        stop(index),
    ]


def text_reply(text: str, index: int = 1) -> list:
    return [
        start(index, {"type": "model_output"}),
        delta(index, {"type": "text", "text": text}),
        stop(index),
    ]


def function_call(index: int, call_id: str, city: str) -> list:
    return [
        start(
            index, {"type": "function_call", "id": call_id, "name": "get_weather", "arguments": {}}
        ),
        delta(index, {"type": "arguments_delta", "arguments": f'{{"city": "{city}"}}'}),
        stop(index),
    ]


async def as_stream(events: list):
    for event in events:
        yield event


def stub_client(service: GoogleInteractionsLLMService, *responses) -> AsyncMock:
    """Have the service's client stream each of `responses` in turn; returns the create mock."""
    create = AsyncMock(side_effect=[as_stream(events) for events in responses])
    service._client = SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
        aio=SimpleNamespace(interactions=SimpleNamespace(create=create), aclose=AsyncMock())
    )
    return create


def make_service(**kwargs) -> GoogleInteractionsLLMService:
    return GoogleInteractionsLLMService(api_key="test-key", **kwargs)


def user_context(text: str = "Hi", **kwargs) -> LLMContext:
    return LLMContext(messages=[{"role": "user", "content": text}], **kwargs)


async def run_context(service, context, **kwargs):
    down, _ = await run_test(service, frames_to_send=[LLMContextFrame(context)], **kwargs)
    return down


def stored_signatures(frames) -> list[dict]:
    return [
        message.message
        for frame in frames
        if isinstance(frame, LLMMessagesAppendFrame)
        for message in frame.messages
    ]


class TestStreaming(unittest.IsolatedAsyncioTestCase):
    async def test_text_reply(self):
        """Test text deltas becoming text frames inside a full response."""
        service = make_service()
        stub_client(service, [*signed_thought(), *text_reply("Hello there."), completed()])

        down = await run_context(service, user_context())

        types = [type(f) for f in down]
        self.assertLess(types.index(LLMFullResponseStartFrame), types.index(LLMTextFrame))
        self.assertEqual(types[-1], LLMFullResponseEndFrame)
        self.assertEqual(
            "".join(f.text for f in down if isinstance(f, LLMTextFrame)), "Hello there."
        )

    async def test_text_signature_stored_with_its_text(self):
        """Test a reply's signature being stored as raw bytes, bookmarked by the reply's text."""
        service = make_service()
        stub_client(service, [*signed_thought(), *text_reply("Hello there."), completed()])

        down = await run_context(service, user_context())

        self.assertEqual(
            stored_signatures(down),
            [
                {
                    "type": "thought_signature",
                    "signature": SIGNATURE,
                    "bookmark": {"text": "Hello there."},
                }
            ],
        )

    async def test_function_call(self):
        """Test a function call's arguments being collected and the call run."""
        service = make_service()
        calls = []

        async def get_weather(params):
            calls.append((params.tool_call_id, params.arguments))
            await params.result_callback({"temperature_f": 72})

        service.register_function("get_weather", get_weather)
        stub_client(
            service,
            [*signed_thought(), *function_call(1, "call_1", "Paris"), completed("requires_action")],
        )

        down = await run_context(service, user_context(tools=ToolsSchema(standard_tools=[WEATHER])))

        started = [f for f in down if isinstance(f, FunctionCallsStartedFrame)]
        self.assertEqual(len(started), 1)
        self.assertEqual(calls, [("call_1", {"city": "Paris"})])
        self.assertEqual(stored_signatures(down)[0]["bookmark"], {"function_call": "call_1"})

    async def test_parallel_calls_signed_once(self):
        """Test parallel calls all running, with the signature stored for the first."""
        service = make_service()
        calls = []

        async def get_weather(params):
            calls.append(params.arguments["city"])
            await params.result_callback({"temperature_f": 72})

        service.register_function("get_weather", get_weather)
        stub_client(
            service,
            [
                *signed_thought(),
                *function_call(1, "call_1", "Paris"),
                *function_call(2, "call_2", "Tokyo"),
                completed("requires_action"),
            ],
        )

        down = await run_context(service, user_context(tools=ToolsSchema(standard_tools=[WEATHER])))

        self.assertEqual(sorted(calls), ["Paris", "Tokyo"])
        self.assertEqual(
            [s["bookmark"] for s in stored_signatures(down)], [{"function_call": "call_1"}]
        )

    async def test_thought_summary(self):
        """Test a thought summary being bracketed as one thought."""
        service = make_service()
        stub_client(
            service,
            [
                start(0, {"type": "thought"}),
                delta(0, {"type": "thought_summary", "content": {"type": "text", "text": "Hmm."}}),
                stop(0),
                *text_reply("Five cents."),
                completed(),
            ],
        )

        down = await run_context(service, user_context())

        thought_frames = [
            f
            for f in down
            if isinstance(f, (LLMThoughtStartFrame, LLMThoughtTextFrame, LLMThoughtEndFrame))
        ]
        self.assertEqual(
            [type(f) for f in thought_frames],
            [LLMThoughtStartFrame, LLMThoughtTextFrame, LLMThoughtEndFrame],
        )
        self.assertEqual(thought_frames[1].text, "Hmm.")

    async def test_token_usage(self):
        """Test thought tokens counting as output and tool-use tokens as input."""
        service = make_service()
        stub_client(
            service,
            [
                *text_reply("Hi."),
                completed(
                    total_input_tokens=10,
                    total_tool_use_tokens=4,
                    total_output_tokens=3,
                    total_thought_tokens=20,
                    total_cached_tokens=2,
                ),
            ],
        )

        down = await run_context(
            service,
            user_context(),
            pipeline_params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        )

        usage = [
            data.value
            for frame in down
            if isinstance(frame, MetricsFrame)
            for data in frame.data
            if isinstance(data, LLMUsageMetricsData)
        ]
        self.assertEqual(len(usage), 1)
        self.assertEqual(usage[0].prompt_tokens, 14)
        self.assertEqual(usage[0].completion_tokens, 23)
        self.assertEqual(usage[0].total_tokens, 37)
        self.assertEqual(usage[0].reasoning_tokens, 20)
        self.assertEqual(usage[0].cache_read_input_tokens, 2)

    async def test_incomplete_response_warns(self):
        """Test a response cut short (e.g. by the token limit) being logged as incomplete."""
        service = make_service()
        stub_client(service, [*text_reply("Han"), completed("incomplete")])

        sink = io.StringIO()
        handler_id = logger.add(sink, level="WARNING", format="{message}")
        try:
            await run_context(service, user_context())
        finally:
            logger.remove(handler_id)

        self.assertIn("response incomplete, the interaction ended incomplete", sink.getvalue())

    async def test_empty_response_reported(self):
        """Test a completed response with no text or function call reported as an error."""
        service = make_service()
        stub_client(service, [*signed_thought(), *text_reply(" "), completed()])

        sink = io.StringIO()
        handler_id = logger.add(sink, level="WARNING", format="{message}")
        try:
            _, up = await run_test(service, frames_to_send=[LLMContextFrame(user_context())])
        finally:
            logger.remove(handler_id)

        self.assertIn(
            "completed with no text or function call (steps: ['thought', 'model_output'])",
            sink.getvalue(),
        )
        errors = [f for f in up if isinstance(f, ErrorFrame)]
        self.assertEqual(
            [e.error for e in errors], ["LLM response completed with no text or function call"]
        )


class TestErrors(unittest.IsolatedAsyncioTestCase):
    async def test_stalled_stream_times_out(self):
        """Test a stream that stops producing ending the response with a timeout error."""

        async def stalled():
            yield start(0, {"type": "model_output"})
            await asyncio.sleep(10)

        service = make_service(stream_idle_timeout_secs=0.05)
        stub_client(service, [])
        service._client.aio.interactions.create = AsyncMock(return_value=stalled())  # pyright: ignore[reportOptionalMemberAccess]

        down, up = await run_test(service, frames_to_send=[LLMContextFrame(user_context())])

        self.assertIsInstance(down[-1], LLMFullResponseEndFrame)
        errors = [f for f in up if isinstance(f, ErrorFrame)]
        self.assertEqual([e.error for e in errors], ["LLM completion timeout"])

    async def test_bad_request_removes_pending_file_message(self):
        """Test a 400 removing the file message that likely caused it."""

        class BadRequestError(Exception):
            status_code = 400

        context = LLMContext()
        context.add_message({"role": "user", "content": "hello"})
        await context.add_file_frame_message(
            type="bytes", format="application/pdf", file="data:application/pdf;base64,JVBERg=="
        )

        service = make_service()
        stub_client(service, [])
        service._client.aio.interactions.create = AsyncMock(side_effect=BadRequestError("bad"))  # pyright: ignore[reportOptionalMemberAccess]

        await run_test(service, frames_to_send=[LLMContextFrame(context)])

        self.assertEqual(context.get_messages(), [{"role": "user", "content": "hello"}])

    def test_requires_google_genai_2(self):
        """Test the service refusing to start on an SDK without the current Interactions schema."""
        with patch.object(interactions_llm, "StepStart", None):
            with self.assertRaisesRegex(ImportError, "google-genai >= 2.0.0"):
                make_service()


class TestRequest(unittest.IsolatedAsyncioTestCase):
    async def request_for(self, service, context=None) -> dict:
        create = stub_client(service, [*text_reply("Hi."), completed()])
        await run_context(service, context or user_context())
        return create.call_args.kwargs

    async def test_stateless_streamed_request(self):
        """Test the request streaming, storing nothing, and naming the model."""
        request = await self.request_for(make_service())
        self.assertTrue(request["stream"])
        self.assertFalse(request["store"])
        self.assertEqual(request["model"], "gemini-3.6-flash")
        self.assertEqual(request["input"][0]["type"], "user_input")
        self.assertNotIn("tools", request)

    async def test_generation_settings(self):
        """Test sampling settings and the system instruction reaching the request."""
        service = make_service(
            settings=GoogleInteractionsLLMService.Settings(
                system_instruction="Be brief.", temperature=0.2, max_tokens=100, seed=7
            )
        )
        request = await self.request_for(service)
        self.assertEqual(request["system_instruction"], "Be brief.")
        self.assertEqual(request["generation_config"]["temperature"], 0.2)
        self.assertEqual(request["generation_config"]["max_output_tokens"], 100)
        self.assertEqual(request["generation_config"]["seed"], 7)

    async def test_thinking_default(self):
        """Test a listed model getting its lowest thinking level when none is set."""
        request = await self.request_for(make_service())
        self.assertEqual(request["generation_config"]["thinking_level"], "minimal")

    async def test_thinking_settings(self):
        """Test an explicit thinking level and thought summaries replacing the default."""
        service = make_service(
            settings=GoogleInteractionsLLMService.Settings(
                thinking=GoogleInteractionsLLMService.ThinkingConfig(
                    thinking_level="high", include_thoughts=True
                )
            )
        )
        request = await self.request_for(service)
        self.assertEqual(request["generation_config"]["thinking_level"], "high")
        self.assertEqual(request["generation_config"]["thinking_summaries"], "auto")

    async def test_provider_thinking_default(self):
        """Test use_provider_thinking_default leaving thinking to Google."""
        service = make_service(
            settings=GoogleInteractionsLLMService.Settings(use_provider_thinking_default=True)
        )
        request = await self.request_for(service)
        self.assertNotIn("thinking_level", request.get("generation_config", {}))

    async def test_unsupported_settings_ignored_with_warning(self):
        """Test top_k and a thinking budget being left out, each warned about once."""
        service = make_service(
            settings=GoogleInteractionsLLMService.Settings(
                top_k=5,
                thinking=GoogleInteractionsLLMService.ThinkingConfig(thinking_budget=100),
            )
        )
        sink = io.StringIO()
        handler_id = logger.add(sink, level="WARNING", format="{message}")
        try:
            request = await self.request_for(service)
        finally:
            logger.remove(handler_id)

        self.assertNotIn("top_k", request["generation_config"])
        self.assertIn("ignoring top_k", sink.getvalue())
        self.assertIn("ignoring thinking_budget", sink.getvalue())

    async def test_safety_settings_converted(self):
        """Test generateContent safety settings taking the Interactions API's names."""
        service = make_service(
            settings=GoogleInteractionsLLMService.Settings(
                safety_settings=[
                    SafetySetting(
                        category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    )
                ]
            )
        )
        request = await self.request_for(service)
        self.assertEqual(
            request["safety_settings"], [{"type": "harassment", "threshold": "block_only_high"}]
        )

    async def test_tool_config_converted(self):
        """Test a function_calling_config becoming the Interactions API's tool_choice."""
        service = make_service(
            tool_config={
                "function_calling_config": {"mode": "ANY", "allowed_function_names": ["a"]}
            }
        )
        request = await self.request_for(service)
        self.assertEqual(
            request["generation_config"]["tool_choice"],
            {"allowed_tools": {"mode": "any", "tools": ["a"]}},
        )

    async def test_context_tools(self):
        """Test the context's tools being sent as function tools."""
        request = await self.request_for(
            make_service(), user_context(tools=ToolsSchema(standard_tools=[WEATHER]))
        )
        self.assertEqual([t["type"] for t in request["tools"]], ["function"])

    async def test_service_tools_used_without_context_tools(self):
        """Test tools given to the service, in generateContent format, filling in for the context's."""
        service = make_service(tools=[{"google_search": {}}])
        request = await self.request_for(service)
        self.assertEqual(request["tools"], [{"type": "google_search"}])

    async def test_extra(self):
        """Test extra request fields being merged in, generation_config field by field."""
        service = make_service(
            settings=GoogleInteractionsLLMService.Settings(
                extra={"service_tier": "flex", "generation_config": {"stop_sequences": ["END"]}}
            )
        )
        request = await self.request_for(service)
        self.assertEqual(request["service_tier"], "flex")
        self.assertEqual(request["generation_config"]["stop_sequences"], ["END"])
        self.assertEqual(request["generation_config"]["thinking_level"], "minimal")


if __name__ == "__main__":
    unittest.main()
