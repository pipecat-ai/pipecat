#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AWS Nova Sonic hands every ``toolUse`` event to ``run_function_calls``.

A ``toolUse`` for a function with no registered handler (for example, a name
the model made up) must not raise out of the receive task, since that resets
the whole conversation. ``run_function_calls`` answers it with a terminal tool
result.

The service is imported with ``pytest.importorskip`` so the suite is skipped
rather than failing collection when the optional AWS dependencies aren't
installed.
"""

import json
import unittest
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestAWSNovaSonicToolUse(unittest.IsolatedAsyncioTestCase):
    def _service(self):
        mod = pytest.importorskip("pipecat.services.aws.nova_sonic.llm")
        service = mod.AWSNovaSonicLLMService(
            secret_access_key="test", access_key_id="test", region="us-east-1"
        )
        service._content_being_received = MagicMock()
        service._context = MagicMock()
        service._report_user_transcription_ended = AsyncMock()
        service.run_function_calls = AsyncMock()
        return service

    @staticmethod
    def _tool_use(name: str) -> dict:
        return {
            "toolUse": {
                "toolName": name,
                "toolUseId": "call_1",
                "content": json.dumps({"city": "Paris"}),
            }
        }

    async def test_unregistered_function_is_passed_to_run_function_calls(self):
        service = self._service()

        await service._handle_tool_use_event(self._tool_use("made_up_tool"))

        service.run_function_calls.assert_awaited_once()
        (function_calls,) = service.run_function_calls.await_args.args
        self.assertEqual(
            [(c.function_name, c.tool_call_id, c.arguments) for c in function_calls],
            [("made_up_tool", "call_1", {"city": "Paris"})],
        )

    async def test_registered_function_is_passed_to_run_function_calls(self):
        service = self._service()
        service.register_function("get_weather", AsyncMock())

        await service._handle_tool_use_event(self._tool_use("get_weather"))

        service.run_function_calls.assert_awaited_once()
