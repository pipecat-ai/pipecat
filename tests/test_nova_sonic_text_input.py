#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for typed input in AWS Nova Sonic conversations."""

import unittest
from unittest.mock import AsyncMock

import pytest

from pipecat.frames.frames import InputTextRawFrame


class TestAWSNovaSonicTextInput(unittest.IsolatedAsyncioTestCase):
    def _service(self):
        pytest.importorskip("aws_sdk_bedrock_runtime")
        mod = pytest.importorskip("pipecat.services.aws.nova_sonic.llm")
        service = mod.AWSNovaSonicLLMService(
            secret_access_key="test", access_key_id="test", region="us-east-1"
        )
        service._stream = object()
        service._prompt_name = "test-prompt"
        service.send_text = AsyncMock()
        return service

    async def test_typed_user_text_is_interactive(self):
        service = self._service()

        await service._handle_input_text_frame(InputTextRawFrame(text="Can you hear me?"))

        service.send_text.assert_awaited_once_with(
            "Can you hear me?", "USER", "test-prompt", service._stream, True
        )

    async def test_empty_typed_user_text_is_ignored(self):
        service = self._service()

        await service._handle_input_text_frame(InputTextRawFrame(text=""))

        service.send_text.assert_not_awaited()
