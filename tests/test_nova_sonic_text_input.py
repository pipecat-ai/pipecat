#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for typed input in AWS Nova Sonic conversations."""

import unittest
from unittest.mock import AsyncMock, Mock

import pytest

from pipecat.frames.frames import LLMContextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection


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
        service.push_frame = AsyncMock()
        service._context = LLMContext()
        service._audio_input_started = True
        return service

    async def test_typed_user_text_is_interactive(self):
        service = self._service()

        message = {"role": "user", "content": "Can you hear me?"}
        service._context.add_message(message)
        await service.process_frame(
            LLMContextFrame(service._context, appended_messages=[message]),
            FrameDirection.DOWNSTREAM,
        )

        service.send_text.assert_awaited_once_with(
            "Can you hear me?", "USER", "test-prompt", service._stream, True
        )

    async def test_ordinary_context_update_does_not_replay_user_history(self):
        service = self._service()
        service._context.add_message({"role": "user", "content": "Already sent"})
        await service.process_frame(LLMContextFrame(service._context), FrameDirection.DOWNSTREAM)
        service.send_text.assert_not_awaited()

    async def test_deferred_user_messages_trigger_one_interactive_turn(self):
        service = self._service()
        messages = [
            {"role": "user", "content": "Remember this"},
            {"role": "user", "content": "Answer now"},
        ]
        service._context.add_messages(messages)
        await service.process_frame(
            LLMContextFrame(service._context, appended_messages=messages),
            FrameDirection.DOWNSTREAM,
        )
        service.send_text.assert_awaited_once_with(
            "Remember this\nAnswer now", "USER", "test-prompt", service._stream, True
        )

    async def test_first_text_is_sent_once_after_audio_input_starts(self):
        for connection_ready in (True, False):
            with self.subTest(connection_ready=connection_ready):
                service = self._service()
                service._context = None
                service._audio_input_started = False
                service._ready_to_send_context = connection_ready
                service._send_prompt_start_event = AsyncMock()
                service._send_audio_input_start_event = AsyncMock()
                service._sc = Mock()

                # No receive loop or network is needed to verify initial seeding.
                service.create_task = lambda coroutine: coroutine.close()
                message = {"role": "user", "content": "Hello"}
                context = LLMContext(messages=[message])
                await service.process_frame(
                    LLMContextFrame(context, appended_messages=[message]),
                    FrameDirection.DOWNSTREAM,
                )
                if not connection_ready:
                    service.send_text.assert_not_awaited()
                    service._ready_to_send_context = True
                    await service._finish_connecting_if_context_available()
                service.send_text.assert_awaited_once_with(
                    "Hello", "USER", "test-prompt", service._stream, True
                )
                service._send_audio_input_start_event.assert_awaited_once()

    async def test_initial_seeding_excludes_later_queued_input(self):
        service = self._service()
        service._context = None
        service._audio_input_started = False
        service._ready_to_send_context = True
        service._send_prompt_start_event = AsyncMock()
        service._sc = Mock()
        service.create_task = lambda coroutine: coroutine.close()

        async def start_audio():
            service._audio_input_started = True

        service._send_audio_input_start_event = start_audio
        first = {"role": "user", "content": "First"}
        second = {"role": "user", "content": "Second"}
        context = LLMContext(messages=[first])
        first_frame = LLMContextFrame(context, appended_messages=[first])
        context.add_message(second)
        second_frame = LLMContextFrame(context, appended_messages=[second])
        await service.process_frame(first_frame, FrameDirection.DOWNSTREAM)
        await service.process_frame(second_frame, FrameDirection.DOWNSTREAM)

        self.assertEqual(
            [call.args[0] for call in service.send_text.await_args_list], ["First", "Second"]
        )
        self.assertTrue(all(call.args[-1] for call in service.send_text.await_args_list))
