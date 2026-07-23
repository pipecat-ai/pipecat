#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from types import SimpleNamespace

from pipecat.services.mcp_service import MCPClient


class FailingSession:
    async def call_tool(self, function_name, arguments=None):
        raise RuntimeError("upstream unavailable")


class TestMCPService(unittest.IsolatedAsyncioTestCase):
    async def test_call_tool_returns_original_error_message_on_exception(self):
        captured_results = []

        async def result_callback(result):
            captured_results.append(result)

        fake_client = SimpleNamespace(_tools_output_filters={})

        await MCPClient._call_tool(
            fake_client,
            FailingSession(),
            "demo_tool",
            {},
            result_callback,
        )

        self.assertEqual(
            captured_results,
            ["Error calling mcp tool demo_tool: upstream unavailable"],
        )
