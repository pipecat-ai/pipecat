#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Regression tests for RTVI ``send-text`` in speech-to-speech pipelines.

These tests cover the fix for GitHub issue #3829: when a client sends a
``send-text`` message while a speech-to-speech LLM service (OpenAI Realtime,
AWS Nova Sonic) is active, the user text must not only interrupt the bot
but also be forwarded to the provider to trigger a new model response.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.frames.frames import (
    InputTextRawFrame,
    LLMMessagesAppendFrame,
)
from pipecat.processors.frameworks.rtvi.processor import RTVIProcessor


class TestRTVIProcessorSendText(unittest.IsolatedAsyncioTestCase):
    """Validate that ``send-text`` emits both an append frame and an input-text frame."""

    def setUp(self):
        self.processor = RTVIProcessor()
        # Prevent the processor from trying to push through an un-linked pipeline.
        self.processor.push_frame = AsyncMock()
        self.processor.interrupt_bot = AsyncMock()

    async def asyncTearDown(self):
        await self.processor.cleanup()

    async def test_send_text_run_immediately_pushes_append_and_input_text(self):
        """run_immediately=True should push LLMMessagesAppendFrame and InputTextRawFrame."""
        await self.processor._handle_send_text(
            RTVI.SendTextData(
                content="hello realtime",
                options=RTVI.SendTextOptions(run_immediately=True),
            )
        )

        self.processor.interrupt_bot.assert_awaited_once()
        pushed_frames = [call.args[0] for call in self.processor.push_frame.await_args_list]

        append_frames = [f for f in pushed_frames if isinstance(f, LLMMessagesAppendFrame)]
        input_text_frames = [f for f in pushed_frames if isinstance(f, InputTextRawFrame)]

        self.assertEqual(len(append_frames), 1)
        self.assertEqual(append_frames[0].messages[0]["content"], "hello realtime")
        self.assertTrue(append_frames[0].run_llm)

        self.assertEqual(len(input_text_frames), 1)
        self.assertEqual(input_text_frames[0].text, "hello realtime")

        # The input-text frame must arrive after the append frame so the
        # context aggregator ingests the user turn before the realtime service
        # triggers a response on the provider side.
        self.assertLess(
            pushed_frames.index(append_frames[0]),
            pushed_frames.index(input_text_frames[0]),
        )

    async def test_send_text_without_run_immediately_does_not_push_input_text(self):
        """run_immediately=False appends to context but must not drive a response."""
        await self.processor._handle_send_text(
            RTVI.SendTextData(
                content="queued message",
                options=RTVI.SendTextOptions(run_immediately=False),
            )
        )

        self.processor.interrupt_bot.assert_not_awaited()
        pushed_frames = [call.args[0] for call in self.processor.push_frame.await_args_list]

        self.assertTrue(any(isinstance(f, LLMMessagesAppendFrame) for f in pushed_frames))
        self.assertFalse(any(isinstance(f, InputTextRawFrame) for f in pushed_frames))


class TestOpenAIRealtimeSendUserText(unittest.IsolatedAsyncioTestCase):
    """Validate OpenAI Realtime forwards typed user text to the provider."""

    def setUp(self):
        # Import inside setUp so optional deps can be skipped in environments
        # without the OpenAI realtime extras.
        from pipecat.services.openai.realtime.llm import (  # noqa: WPS433
            OpenAIRealtimeLLMService,
        )

        self.service = OpenAIRealtimeLLMService(api_key="test-key")
        # Simulate an active WebSocket session.
        self.service._websocket = MagicMock()
        self.service._disconnecting = False
        self.service.send_client_event = AsyncMock()
        self.service._create_response = AsyncMock()

    async def asyncTearDown(self):
        await self.service.cleanup()

    async def test_send_user_text_emits_conversation_item_and_response(self):
        from pipecat.services.openai.realtime import events  # noqa: WPS433

        await self.service._send_user_text("hello realtime")

        self.service.send_client_event.assert_awaited_once()
        sent_event = self.service.send_client_event.await_args.args[0]
        self.assertIsInstance(sent_event, events.ConversationItemCreateEvent)
        self.assertEqual(sent_event.item.type, "message")
        self.assertEqual(sent_event.item.role, "user")
        self.assertEqual(sent_event.item.content[0].type, "input_text")
        self.assertEqual(sent_event.item.content[0].text, "hello realtime")
        self.assertIn(sent_event.item.id, self.service._messages_added_manually)

        self.service._create_response.assert_awaited_once()

    async def test_send_user_text_skips_response_when_disabled(self):
        await self.service._send_user_text("append only", trigger_response=False)

        self.service.send_client_event.assert_awaited_once()
        self.service._create_response.assert_not_awaited()

    async def test_send_user_text_noop_when_not_connected(self):
        self.service._websocket = None

        await self.service._send_user_text("hello")

        self.service.send_client_event.assert_not_awaited()
        self.service._create_response.assert_not_awaited()

    async def test_handle_messages_append_forwards_user_text(self):
        frame = LLMMessagesAppendFrame(
            messages=[{"role": "user", "content": "typed in"}],
            run_llm=True,
        )

        self.service._send_user_text = AsyncMock()
        await self.service._handle_messages_append(frame)

        self.service._send_user_text.assert_awaited_once_with("typed in", trigger_response=False)
        self.service._create_response.assert_awaited_once()

    async def test_handle_messages_append_respects_run_llm_false(self):
        frame = LLMMessagesAppendFrame(
            messages=[{"role": "user", "content": "queued"}],
            run_llm=False,
        )

        self.service._send_user_text = AsyncMock()
        await self.service._handle_messages_append(frame)

        self.service._send_user_text.assert_awaited_once_with("queued", trigger_response=False)
        self.service._create_response.assert_not_awaited()

    async def test_handle_messages_append_ignores_non_user_messages(self):
        frame = LLMMessagesAppendFrame(
            messages=[{"role": "assistant", "content": "hi"}],
            run_llm=True,
        )

        self.service._send_user_text = AsyncMock()
        await self.service._handle_messages_append(frame)

        self.service._send_user_text.assert_not_awaited()
        self.service._create_response.assert_not_awaited()

    async def test_handle_messages_append_with_structured_content(self):
        frame = LLMMessagesAppendFrame(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "multi"},
                        {"type": "text", "text": "-part"},
                    ],
                }
            ],
            run_llm=True,
        )

        self.service._send_user_text = AsyncMock()
        await self.service._handle_messages_append(frame)

        self.service._send_user_text.assert_awaited_once_with("multi-part", trigger_response=False)
        self.service._create_response.assert_awaited_once()

    async def test_existing_context_update_does_not_resend_user_message(self):
        """Guardrail: plain context refresh should not double-send typed text."""
        from pipecat.processors.aggregators.llm_context import LLMContext  # noqa: WPS433

        context = LLMContext(messages=[{"role": "user", "content": "hi"}])
        self.service._context = context  # already seeded, simulate running session
        self.service._process_completed_function_calls = AsyncMock()

        await self.service._handle_context(context)

        self.service.send_client_event.assert_not_awaited()
        self.service._create_response.assert_not_awaited()
        self.service._process_completed_function_calls.assert_awaited_once_with(
            send_new_results=True
        )


class TestNovaSonicInputTextHandling(unittest.IsolatedAsyncioTestCase):
    """Validate AWS Nova Sonic forwards typed user text as an interactive event."""

    def setUp(self):
        from pipecat.adapters.services.aws_nova_sonic_adapter import Role  # noqa: WPS433
        from pipecat.services.aws.nova_sonic.llm import AWSNovaSonicLLMService  # noqa: WPS433

        self._role = Role
        self.service = AWSNovaSonicLLMService(
            secret_access_key="x", access_key_id="y", region="us-east-1"
        )
        # Pretend the bidirectional stream is already running.
        self.service._stream = MagicMock()
        self.service._prompt_name = "prompt-1"
        self.service._disconnecting = False
        self.service._send_text_event = AsyncMock()

    async def asyncTearDown(self):
        await self.service.cleanup()

    async def test_input_text_frame_sends_interactive_user_text(self):
        frame = InputTextRawFrame(text="hello nova")

        await self.service._handle_input_text_frame(frame)

        self.service._send_text_event.assert_awaited_once()
        kwargs = self.service._send_text_event.await_args.kwargs
        self.assertEqual(kwargs["text"], "hello nova")
        self.assertEqual(kwargs["role"], self._role.USER)
        self.assertTrue(kwargs["interactive"])

    async def test_input_text_frame_noop_when_not_connected(self):
        self.service._stream = None

        await self.service._handle_input_text_frame(InputTextRawFrame(text="hi"))

        self.service._send_text_event.assert_not_awaited()

    async def test_input_text_frame_noop_on_empty_text(self):
        await self.service._handle_input_text_frame(InputTextRawFrame(text=""))

        self.service._send_text_event.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
