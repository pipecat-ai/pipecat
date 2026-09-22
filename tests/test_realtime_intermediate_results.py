#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Realtime services deliver a tool's results as they are produced.

A speech-to-speech provider takes one output per function call, so an
intermediate result — one reported with
``FunctionCallResultProperties(is_final=False)`` while the call keeps running —
goes in beside the call rather than through it. Each service maps that onto
whatever second channel its provider offers, and onto what the provider needs
to be told about when to speak.

Results are delivered from the ``FunctionCallResultFrame`` the service
broadcasts, as it is produced. What runs the model is the context frame the
assistant aggregator pushes upstream afterwards, which already accounts for
``run_llm``, sibling calls, bot speech and user speech.
"""

import json
import unittest
from unittest.mock import AsyncMock

import pytest

from pipecat.frames.frames import (
    FunctionCallCancelFrame,
    FunctionCallResultFrame,
    FunctionCallResultProperties,
)
from pipecat.processors.aggregators import async_tool_messages
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

WEATHER = {"temperature": 75}


def _result(call_id: str = "call_1", *, is_final: bool, run_llm: bool = True, result=None):
    return FunctionCallResultFrame(
        function_name="delegate",
        tool_call_id=call_id,
        arguments={},
        result=result if result is not None else WEATHER,
        properties=FunctionCallResultProperties(is_final=is_final, run_llm=run_llm),
    )


class OpenAIProtocolIntermediateResultsTests:
    """The checks every service speaking the OpenAI realtime protocol answers.

    Each service has its own module and its own event models, so the suite runs
    once per service rather than testing one and assuming the rest.
    """

    def _build_service(self):
        raise NotImplementedError

    def setUp(self):
        self.service = self._build_service()
        self.service.send_client_event = AsyncMock()
        self.service._api_session_ready = True
        self.service._llm_needs_conversation_setup = False
        self.service._context = LLMContext()
        # The API is waiting on this call.
        self.service._open_function_calls.add("call_1")

    def _sent(self) -> list:
        return [c.args[0] for c in self.service.send_client_event.call_args_list]

    def _items(self) -> list:
        return [e.item for e in self._sent() if hasattr(e, "item")]

    def _responses(self) -> list:
        return [e for e in self._sent() if e.type == "response.create"]

    async def _push_context_upstream(self):
        await self.service._handle_context(self.service._context, FrameDirection.UPSTREAM)

    async def test_an_intermediate_result_goes_in_beside_the_call(self):
        await self.service.push_frame(_result(is_final=False))

        (item,) = self._items()
        self.assertEqual(item.type, "message")
        payload = async_tool_messages.parse_message(
            {"role": "developer", "content": item.content[0].text}
        )
        assert payload is not None
        self.assertEqual(payload.kind, "intermediate")
        self.assertEqual(json.loads(payload.result or ""), WEATHER)
        # The call stays open for the result that settles it.
        self.assertIn("call_1", self.service._open_function_calls)

    async def test_a_final_result_answers_the_call(self):
        await self.service.push_frame(_result(is_final=True))

        (item,) = self._items()
        self.assertEqual(item.type, "function_call_output")
        self.assertEqual(item.call_id, "call_1")
        self.assertEqual(json.loads(item.output), WEATHER)
        self.assertNotIn("call_1", self.service._open_function_calls)

    async def test_a_result_for_a_call_the_api_lost_goes_in_as_a_message(self):
        # A reconnect replays the conversation as text, so the API no longer
        # knows the id and would reject an output for it.
        self.service._open_function_calls.clear()

        await self.service.push_frame(_result(is_final=True))

        (item,) = self._items()
        self.assertEqual(item.type, "message")
        payload = async_tool_messages.parse_message(
            {"role": "developer", "content": item.content[0].text}
        )
        assert payload is not None
        self.assertEqual(payload.kind, "final")

    async def test_nothing_runs_the_model_until_the_aggregator_asks(self):
        await self.service.push_frame(_result(is_final=False))
        self.assertEqual(self._responses(), [])

        await self._push_context_upstream()
        self.assertEqual(len(self._responses()), 1)

    async def test_a_context_frame_from_downstream_records_rather_than_runs(self):
        await self.service.push_frame(_result(is_final=False))

        await self.service._handle_context(self.service._context, FrameDirection.DOWNSTREAM)

        self.assertEqual(self._responses(), [])

    async def test_a_run_asked_for_during_a_response_waits_for_it_to_finish(self):
        await self._response_created()
        await self.service.push_frame(_result(is_final=True))

        await self._push_context_upstream()
        self.assertEqual(self._responses(), [])

        await self._response_done()
        self.assertEqual(len(self._responses()), 1)

    async def test_a_response_takes_in_what_was_delivered_before_it(self):
        await self.service.push_frame(_result(is_final=True))
        await self._response_created()

        await self._response_done()
        await self._push_context_upstream()

        self.assertEqual(self._responses(), [])

    async def test_an_interruption_drops_a_run_that_was_waiting(self):
        await self._response_created()
        await self.service.push_frame(_result(is_final=True))
        await self._push_context_upstream()

        await self.service._handle_interruption()
        await self._response_done()

        self.assertEqual(self._responses(), [])

    async def test_a_cancelled_call_is_settled(self):
        await self.service.push_frame(
            FunctionCallCancelFrame(function_name="delegate", tool_call_id="call_1")
        )

        (item,) = self._items()
        self.assertEqual(item.type, "function_call_output")
        self.assertEqual(item.output, "CANCELLED")

    async def test_the_context_scan_neither_errors_nor_resends(self):
        self.service.push_error = AsyncMock()
        await self.service.push_frame(_result(is_final=False))
        await self.service.push_frame(_result(is_final=True))
        sent = len(self._items())

        # The same results, as the aggregator records them in the context.
        self.service._context.add_message(
            async_tool_messages.build_intermediate_result_message("call_1", json.dumps(WEATHER))
        )
        self.service._context.add_message(
            async_tool_messages.build_final_result_message("call_1", json.dumps(WEATHER))
        )
        await self._push_context_upstream()

        self.assertEqual(len(self._items()), sent)
        complaints = [
            str(c.kwargs.get("error_msg", "")) for c in self.service.push_error.call_args_list
        ]
        self.assertFalse([c for c in complaints if "intermediate" in c])

    async def _response_created(self):
        await self.service._handle_evt_response_created(_ResponseCreated())

    async def _response_done(self):
        """Feed the service a response.done, which is where a held run fires."""
        await self.service._handle_evt_response_done(_ResponseDone())


class TestOpenAIRealtimeIntermediateResults(
    OpenAIProtocolIntermediateResultsTests, unittest.IsolatedAsyncioTestCase
):
    def _build_service(self):
        return OpenAIRealtimeLLMService(api_key="test")


class TestGrokRealtimeIntermediateResults(
    OpenAIProtocolIntermediateResultsTests, unittest.IsolatedAsyncioTestCase
):
    def _build_service(self):
        from pipecat.services.xai.realtime.llm import GrokRealtimeLLMService

        return GrokRealtimeLLMService(api_key="test")


class TestInworldRealtimeIntermediateResults(unittest.IsolatedAsyncioTestCase):
    """Inworld takes none: its model calls the tool again instead of relaying."""

    def setUp(self):
        from pipecat.services.inworld.realtime.llm import InworldRealtimeLLMService

        self.service = InworldRealtimeLLMService(api_key="test")
        self.service.send_client_event = AsyncMock()
        self.service._api_session_ready = True
        self.service._llm_needs_conversation_setup = False
        self.service._context = LLMContext()
        self.service._open_function_calls.add("call_1")

    def _items(self) -> list:
        return [
            c.args[0].item
            for c in self.service.send_client_event.call_args_list
            if hasattr(c.args[0], "item")
        ]

    async def test_it_says_it_takes_no_intermediate_results(self):
        self.assertFalse(self.service.accepts_intermediate_function_call_results)

    async def test_an_intermediate_result_is_dropped(self):
        await self.service.push_frame(_result(is_final=False))

        self.assertEqual(self._items(), [])

    async def test_a_final_result_answers_the_call(self):
        await self.service.push_frame(_result(is_final=True))

        (item,) = self._items()
        self.assertEqual(item.type, "function_call_output")


class TestGeminiLiveIntermediateResults(unittest.IsolatedAsyncioTestCase):
    """Gemini takes them on the tool-response channel itself, as a generator."""

    def setUp(self):
        pytest.importorskip("google.genai")
        from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService

        self.service = GeminiLiveLLMService(api_key="test")
        self.service._session = AsyncMock()
        self.service._tool_call_id_to_name["call_1"] = "delegate"
        # The tool the results belong to is an async one, which is what the
        # NON_BLOCKING declaration and the scheduling hints follow.
        self.service._function_is_async = lambda name: True

    def _responses(self) -> list:
        return [
            c.kwargs["function_responses"]
            for c in self.service._session.send_tool_response.call_args_list
        ]

    async def test_an_intermediate_result_keeps_the_call_open(self):
        await self.service.push_frame(_result(is_final=False))

        (response,) = self._responses()
        self.assertTrue(response.will_continue)
        self.assertEqual(response.scheduling, "WHEN_IDLE")

    async def test_a_silent_result_asks_the_model_not_to_answer(self):
        await self.service.push_frame(_result(is_final=False, run_llm=False))

        (response,) = self._responses()
        self.assertEqual(response.scheduling, "SILENT")
        # The guides put scheduling inside the response, the reference beside it.
        self.assertEqual(response.response["scheduling"], "SILENT")

    async def test_a_final_result_settles_the_call(self):
        await self.service.push_frame(_result(is_final=True))

        (response,) = self._responses()
        self.assertIsNone(response.will_continue)
        self.assertEqual(response.scheduling, "WHEN_IDLE")
        self.assertIn("call_1", self.service._completed_tool_calls)

    async def test_a_model_without_non_blocking_tools_takes_no_intermediate_results(self):
        self.service._settings.model = "models/gemini-3.1-flash-live-preview"
        self.assertFalse(self.service.accepts_intermediate_function_call_results)

        await self.service.push_frame(_result(is_final=False))

        self.assertEqual(self._responses(), [])


class TestNovaSonicIntermediateResults(unittest.IsolatedAsyncioTestCase):
    """Nova Sonic takes them as text beside the call."""

    def setUp(self):
        pytest.importorskip("aws_sdk_bedrock_runtime")
        from pipecat.services.aws.nova_sonic.llm import AWSNovaSonicLLMService

        self.service = AWSNovaSonicLLMService(secret_access_key="k", access_key_id="i", region="r")
        self.service.send_text = AsyncMock()
        self.service._send_tool_result = AsyncMock()
        self.service._stream = object()
        self.service._prompt_name = "prompt"

    def _texts(self) -> list[tuple[str, bool]]:
        # send_text(text, role, prompt_name, stream, interactive)
        return [(c.args[0], c.args[4]) for c in self.service.send_text.call_args_list]

    async def test_an_intermediate_result_goes_in_as_interactive_text(self):
        await self.service.push_frame(_result(is_final=False))

        ((text, interactive),) = self._texts()
        payload = async_tool_messages.parse_message({"role": "developer", "content": text})
        assert payload is not None
        self.assertEqual(payload.kind, "intermediate")
        self.assertTrue(interactive)
        self.service._send_tool_result.assert_not_awaited()

    async def test_a_silent_result_goes_in_without_asking_for_a_reply(self):
        await self.service.push_frame(_result(is_final=False, run_llm=False))

        ((_, interactive),) = self._texts()
        self.assertFalse(interactive)

    async def test_a_final_result_settles_the_call(self):
        await self.service.push_frame(_result(is_final=True))

        self.assertEqual(self._texts(), [])
        self.service._send_tool_result.assert_awaited_once()
        self.assertIn("call_1", self.service._completed_tool_calls)


class TestUltravoxIntermediateResults(unittest.IsolatedAsyncioTestCase):
    """Ultravox takes them as user-side text, urgent or not."""

    def setUp(self):
        from pipecat.services.ultravox.llm import (
            OneShotInputParams,
            UltravoxRealtimeLLMService,
        )

        self.service = UltravoxRealtimeLLMService(
            params=OneShotInputParams(api_key="test", system_prompt="test")
        )
        self.service._socket = object()
        self.service._send = AsyncMock()

    def _sent(self) -> list[dict]:
        return [c.args[0] for c in self.service._send.call_args_list]

    async def test_an_intermediate_result_goes_in_as_text_to_speak_about(self):
        await self.service.push_frame(_result(is_final=False))

        (message,) = self._sent()
        self.assertEqual(message["type"], "user_text_message")
        self.assertIn("still running", message["text"])
        self.assertEqual(message["urgency"], "soon")

    async def test_a_silent_result_goes_in_as_context(self):
        await self.service.push_frame(_result(is_final=False, run_llm=False))

        (message,) = self._sent()
        self.assertEqual(message["urgency"], "later")

    async def test_a_final_result_settles_the_call(self):
        await self.service.push_frame(_result(is_final=True))

        (message,) = self._sent()
        self.assertEqual(message["type"], "client_tool_result")
        self.assertEqual(message["invocationId"], "call_1")

    async def test_a_final_result_for_an_async_call_goes_in_as_text(self):
        # The placeholder already settled the call, so the result can't.
        self.service._started_placeholder_sent.add("call_1")

        await self.service.push_frame(_result(is_final=True))

        (message,) = self._sent()
        self.assertEqual(message["type"], "user_text_message")
        self.assertIn("Async tool result", message["text"])


class _Usage:
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0


class _Response:
    id = "resp_1"
    usage = _Usage()
    status = "completed"
    status_details = None
    output: list = []


class _ResponseCreated:
    """The fields a ``response.created`` handler reads."""

    response = _Response()


class _ResponseDone:
    """The fields a ``response.done`` handler reads."""

    response = _Response()
    usage = _Usage()
