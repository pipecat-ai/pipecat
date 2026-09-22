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


class TestOpenAIRealtimeIntermediateResults(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.service = OpenAIRealtimeLLMService(api_key="test")
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
        await self.service._handle_evt_response_created(None)
        await self.service.push_frame(_result(is_final=True))

        await self._push_context_upstream()
        self.assertEqual(self._responses(), [])

        await self._response_done()
        self.assertEqual(len(self._responses()), 1)

    async def test_a_response_takes_in_what_was_delivered_before_it(self):
        await self.service.push_frame(_result(is_final=True))
        await self.service._handle_evt_response_created(None)

        await self._response_done()
        await self._push_context_upstream()

        self.assertEqual(self._responses(), [])

    async def test_an_interruption_drops_a_run_that_was_waiting(self):
        await self.service._handle_evt_response_created(None)
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
        self.service.push_error.assert_not_awaited()

    async def _response_done(self):
        """Feed the service a response.done, which is where a held run fires."""
        await self.service._handle_evt_response_done(_ResponseDone())


class _Usage:
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0


class _Response:
    usage = _Usage()
    status = "completed"
    output: list = []


class _ResponseDone:
    """The fields ``_handle_evt_response_done`` reads."""

    response = _Response()
