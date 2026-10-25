#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the function call fields RTVIObserver sends at a report level."""

import unittest
from unittest.mock import AsyncMock

from pipecat.frames.frames import (
    FunctionCallCancelFrame,
    FunctionCallFromLLM,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
)
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frameworks.rtvi.observer import (
    RTVIFunctionCallReportLevel,
    RTVIObserver,
    RTVIObserverParams,
)
from pipecat.tests.utils import run_test


class TestRTVIObserverReportLevel(unittest.IsolatedAsyncioTestCase):
    async def test_arguments_level_reports_arguments_and_no_result(self):
        observer = RTVIObserver(
            params=RTVIObserverParams(
                function_call_report_level={"*": RTVIFunctionCallReportLevel.ARGUMENTS}
            )
        )
        sent = []
        observer.send_rtvi_message = AsyncMock(side_effect=lambda m: sent.append(m))
        arguments = {"query": "pricing"}

        await run_test(
            IdentityFilter(),
            frames_to_send=[
                FunctionCallsStartedFrame(
                    function_calls=[
                        FunctionCallFromLLM(
                            function_name="search",
                            tool_call_id="call_1",
                            arguments=arguments,
                            context=None,
                        )
                    ]
                ),
                FunctionCallInProgressFrame(
                    function_name="search", tool_call_id="call_1", arguments=arguments
                ),
                FunctionCallResultFrame(
                    function_name="search",
                    tool_call_id="call_1",
                    arguments=arguments,
                    result={"passages": ["Pro costs $20 a month."]},
                ),
                FunctionCallCancelFrame(function_name="search", tool_call_id="call_2"),
            ],
            observers=[observer],
        )

        # System frames can overtake the others, so the order of the messages is not fixed.
        self.assertCountEqual(
            [(m.type, m.data.model_dump(exclude_none=True)) for m in sent],
            [
                ("llm-function-call-started", {"function_name": "search"}),
                (
                    "llm-function-call-in-progress",
                    {"tool_call_id": "call_1", "function_name": "search", "arguments": arguments},
                ),
                (
                    "llm-function-call-stopped",
                    {"tool_call_id": "call_1", "cancelled": False, "function_name": "search"},
                ),
                (
                    "llm-function-call-stopped",
                    {"tool_call_id": "call_2", "cancelled": True, "function_name": "search"},
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
