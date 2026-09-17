#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for how RTVIObserver reports function calls, this pipeline's own and external ones."""

import unittest
from unittest.mock import AsyncMock

from pipecat.frames.frames import (
    ExternalFunctionCallCancelFrame,
    ExternalFunctionCallFrame,
    ExternalFunctionCallInProgressFrame,
    ExternalFunctionCallResultFrame,
    ExternalFunctionCallStartedFrame,
    FunctionCallFromLLM,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallResultProperties,
    FunctionCallsStartedFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frameworks.rtvi.observer import (
    RTVIFunctionCallReportLevel,
    RTVIObserver,
    RTVIObserverParams,
)


def _observer(levels: dict[str, RTVIFunctionCallReportLevel]) -> tuple[RTVIObserver, list]:
    observer = RTVIObserver(params=RTVIObserverParams(function_call_report_level=levels))
    sent: list = []
    observer.send_rtvi_message = AsyncMock(side_effect=lambda m: sent.append(m))
    return observer, sent


def _external(phase: str, parent: str | None = "call_delegate") -> ExternalFunctionCallFrame:
    arguments = {"location": "Seattle"}
    if phase == "started":
        return ExternalFunctionCallStartedFrame(
            "get_weather", "toolu_1", parent_tool_call_id=parent
        )
    if phase == "in_progress":
        return ExternalFunctionCallInProgressFrame(
            "get_weather", "toolu_1", arguments=arguments, parent_tool_call_id=parent
        )
    return ExternalFunctionCallResultFrame(
        "get_weather",
        "toolu_1",
        arguments=arguments,
        result={"temp": 62},
        parent_tool_call_id=parent,
    )


class TestExternalFunctionCalls(unittest.IsolatedAsyncioTestCase):
    async def test_an_external_call_is_reported_like_the_pipelines_own(self):
        observer, sent = _observer({"*": RTVIFunctionCallReportLevel.FULL})

        for phase in ("started", "in_progress", "stopped"):
            await observer._report_function_call_frame(_external(phase))

        self.assertEqual(
            [m.type for m in sent],
            [
                "llm-function-call-started",
                "llm-function-call-in-progress",
                "llm-function-call-stopped",
            ],
        )
        self.assertEqual(sent[0].data.function_name, "get_weather")
        self.assertEqual(sent[1].data.arguments, {"location": "Seattle"})
        self.assertEqual(sent[2].data.result, {"temp": 62})
        self.assertEqual([m.data.parent_tool_call_id for m in sent], ["call_delegate"] * 3)

    async def test_an_intermediate_result_leaves_the_call_running(self):
        """Only a final result stops a call, the pipeline's own or an external one."""
        observer, sent = _observer({"*": RTVIFunctionCallReportLevel.FULL})

        await observer._report_function_call_frame(
            FunctionCallResultFrame(
                function_name="get_weather",
                tool_call_id="call_1",
                arguments={},
                result="62",
                properties=FunctionCallResultProperties(is_final=False),
            )
        )
        await observer._report_function_call_frame(
            ExternalFunctionCallResultFrame(
                "get_weather", "toolu_1", arguments={}, result="62", is_final=False
            )
        )
        await observer._report_function_call_frame(
            ExternalFunctionCallCancelFrame("get_weather", "toolu_1")
        )

        self.assertEqual([m.type for m in sent], ["llm-function-call-stopped"])
        self.assertTrue(sent[0].data.cancelled)

    async def test_the_report_level_applies_by_the_external_calls_own_name(self):
        observer, sent = _observer(
            {"*": RTVIFunctionCallReportLevel.FULL, "get_weather": RTVIFunctionCallReportLevel.NAME}
        )

        await observer._report_function_call_frame(_external("in_progress"))

        self.assertEqual(sent[0].data.function_name, "get_weather")
        self.assertIsNone(sent[0].data.arguments)

    async def test_a_hidden_parent_is_left_off_its_children(self):
        """A DISABLED parent stays hidden: its children carry no parent id."""
        observer, sent = _observer(
            {
                "*": RTVIFunctionCallReportLevel.FULL,
                "delegate": RTVIFunctionCallReportLevel.DISABLED,
            }
        )
        context = LLMContext()

        await observer._report_function_call_frame(
            FunctionCallsStartedFrame(
                function_calls=[
                    FunctionCallFromLLM(
                        context=context,
                        function_name="delegate",
                        tool_call_id="call_delegate",
                        arguments={},
                    )
                ]
            )
        )
        await observer._report_function_call_frame(
            FunctionCallInProgressFrame(
                function_name="delegate",
                tool_call_id="call_delegate",
                arguments={},
                cancel_on_interruption=False,
            )
        )
        await observer._report_function_call_frame(_external("in_progress"))
        await observer._report_function_call_frame(
            FunctionCallResultFrame(
                function_name="delegate", tool_call_id="call_delegate", arguments={}, result="ok"
            )
        )

        self.assertEqual([m.type for m in sent], ["llm-function-call-in-progress"])
        self.assertEqual(sent[0].data.function_name, "get_weather")
        self.assertIsNone(sent[0].data.parent_tool_call_id)

    async def test_a_shown_parent_stays_on_its_children(self):
        observer, sent = _observer({"*": RTVIFunctionCallReportLevel.NAME})
        context = LLMContext()

        await observer._report_function_call_frame(
            FunctionCallsStartedFrame(
                function_calls=[
                    FunctionCallFromLLM(
                        context=context,
                        function_name="delegate",
                        tool_call_id="call_delegate",
                        arguments={},
                    )
                ]
            )
        )
        await observer._report_function_call_frame(_external("in_progress"))

        self.assertIsNone(sent[0].data.parent_tool_call_id)
        self.assertEqual(sent[1].data.parent_tool_call_id, "call_delegate")

    async def test_the_pipelines_own_calls_carry_no_parent(self):
        observer, sent = _observer({"*": RTVIFunctionCallReportLevel.FULL})

        await observer._report_function_call_frame(
            FunctionCallResultFrame(
                function_name="get_weather", tool_call_id="call_1", arguments={}, result="62"
            )
        )

        self.assertEqual(sent[0].data.result, "62")
        self.assertIsNone(sent[0].data.parent_tool_call_id)


if __name__ == "__main__":
    unittest.main()
