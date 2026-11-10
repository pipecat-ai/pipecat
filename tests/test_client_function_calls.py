#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.frames.frames import (
    ClientFunctionCallRequestFrame,
    ClientFunctionCallResultFrame,
    FunctionCallCancelFrame,
    FunctionCallFromLLM,
    FunctionCallResultFrame,
    InterruptionFrame,
    UserTurnResponseCancelFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.test_llm_service import MockLLMService


class TestClientFunctionCalls(unittest.IsolatedAsyncioTestCase):
    """A client function is sent to the client and settled by the client's answer."""

    TIMEOUT = 0.1
    SETTLE = 0.3

    def _service(self, **kwargs) -> tuple[MockLLMService, list, list]:
        service = MockLLMService(**kwargs)
        service._task_manager = TaskManager()

        broadcast_frames = []
        pushed_frames = []

        async def mock_broadcast_frame(frame_cls, **frame_kwargs):
            broadcast_frames.append(frame_cls(**frame_kwargs))

        async def mock_push_frame(frame, direction=FrameDirection.DOWNSTREAM):
            pushed_frames.append(frame)

        service.broadcast_frame = mock_broadcast_frame
        service.push_frame = mock_push_frame
        return service, broadcast_frames, pushed_frames

    async def _run_call(self, service: MockLLMService, function_name: str, arguments=None):
        await service.run_function_calls(
            [
                FunctionCallFromLLM(
                    function_name=function_name,
                    tool_call_id="call_1",
                    arguments=arguments or {},
                    context=LLMContext(),
                )
            ]
        )
        # Let the call's task start and push its request.
        await asyncio.sleep(0.01)

    async def _answer(self, service: MockLLMService, result, tool_call_id="call_1"):
        service._handle_client_function_call_result(
            ClientFunctionCallResultFrame(tool_call_id=tool_call_id, result=result)
        )
        await asyncio.sleep(0.01)

    def _results(self, frames: list) -> list[FunctionCallResultFrame]:
        return [frame for frame in frames if isinstance(frame, FunctionCallResultFrame)]

    async def test_call_is_sent_to_the_client(self):
        service, _, pushed = self._service()
        service.register_client_function("build")

        await self._run_call(service, "build", {"site_id": 3})

        requests = [f for f in pushed if isinstance(f, ClientFunctionCallRequestFrame)]
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].function_name, "build")
        self.assertEqual(requests[0].tool_call_id, "call_1")
        self.assertEqual(requests[0].arguments, {"site_id": 3})

    async def test_client_answer_settles_the_call(self):
        service, broadcast, _ = self._service()
        service.register_client_function("build")

        await self._run_call(service, "build")
        self.assertEqual(self._results(broadcast), [])

        await self._answer(service, "built")

        results = self._results(broadcast)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].tool_call_id, "call_1")
        self.assertEqual(results[0].result, "built")

    async def test_catch_all_sends_every_call_to_the_client(self):
        service, broadcast, pushed = self._service()
        service.register_client_function(None)

        await self._run_call(service, "anything")
        await self._answer(service, "done")

        self.assertTrue(any(isinstance(f, ClientFunctionCallRequestFrame) for f in pushed))
        self.assertEqual(self._results(broadcast)[0].result, "done")

    async def test_answer_for_an_unknown_call_is_ignored(self):
        service, broadcast, _ = self._service()
        service.register_client_function("build")

        await self._run_call(service, "build")
        await self._answer(service, "wrong", tool_call_id="call_unknown")

        self.assertEqual(self._results(broadcast), [])

    async def test_timeout_cancels_the_call_and_a_late_answer_is_ignored(self):
        service, broadcast, _ = self._service()
        service.register_client_function("build", timeout_secs=self.TIMEOUT)

        await self._run_call(service, "build")
        await asyncio.sleep(self.SETTLE)
        await self._answer(service, "too late")

        self.assertEqual(self._results(broadcast), [])
        self.assertIsInstance(broadcast[-1], FunctionCallCancelFrame)
        self.assertEqual(service._client_function_calls, {})

    async def test_interruption_cancels_the_call_and_a_late_answer_is_ignored(self):
        service, broadcast, _ = self._service()
        service.register_client_function("build")

        await self._run_call(service, "build")
        await service._handle_interruptions(InterruptionFrame())
        await asyncio.sleep(self.SETTLE)
        await self._answer(service, "too late")

        self.assertEqual(self._results(broadcast), [])
        self.assertIsInstance(broadcast[-1], FunctionCallCancelFrame)
        self.assertEqual(service._client_function_calls, {})


if __name__ == "__main__":
    unittest.main()


class TestSpeculativeFunctionCalls(unittest.IsolatedAsyncioTestCase):
    """A speculative function runs before the turn is confirmed; others wait for it."""

    def _service(self) -> tuple[MockLLMService, list]:
        service = MockLLMService()
        service._task_manager = TaskManager()
        broadcast_frames = []

        async def mock_broadcast_frame(frame_cls, **frame_kwargs):
            broadcast_frames.append(frame_cls(**frame_kwargs))

        async def mock_push_frame(frame, direction=FrameDirection.DOWNSTREAM):
            pass

        service.broadcast_frame = mock_broadcast_frame
        service.push_frame = mock_push_frame
        service._speculation_gate.begin_speculation(True)
        return service, broadcast_frames

    async def _run_call(self, service: MockLLMService, name: str):
        await service.run_function_calls(
            [
                FunctionCallFromLLM(
                    function_name=name, tool_call_id="call_1", arguments={}, context=LLMContext()
                )
            ]
        )
        await asyncio.sleep(0.01)

    async def test_non_speculative_call_withdraws_the_speculation(self):
        service, broadcast = self._service()
        ran = []

        async def book(params):
            ran.append(params.tool_call_id)
            await params.result_callback("booked")

        service.register_function("book", book)
        await self._run_call(service, "book")

        self.assertEqual(ran, [])
        self.assertIsInstance(broadcast[0], UserTurnResponseCancelFrame)

    async def test_speculative_call_runs_during_the_speculation(self):
        service, broadcast = self._service()
        ran = []

        async def lookup(params):
            ran.append(params.tool_call_id)
            await params.result_callback("found")

        service.register_function("lookup", lookup, speculative=True)
        await self._run_call(service, "lookup")

        self.assertEqual(ran, ["call_1"])
        self.assertFalse(any(isinstance(f, UserTurnResponseCancelFrame) for f in broadcast))

    async def test_speculative_call_is_cancelled_when_the_speculation_is_withdrawn(self):
        service, broadcast = self._service()
        service.register_client_function("lookup", speculative=True)
        await self._run_call(service, "lookup")

        await service._handle_user_turn_response_cancel(UserTurnResponseCancelFrame())
        await asyncio.sleep(0.05)

        self.assertIsInstance(broadcast[-1], FunctionCallCancelFrame)
        self.assertEqual(service._client_function_calls, {})
