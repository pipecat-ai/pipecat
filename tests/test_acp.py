#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for Agent Client Protocol support."""

import os
import sys
import unittest

from pipecat.frames.frames import (
    CancelFrame,
    ErrorFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.services.acp.aggregator import ACPUserAggregator
from pipecat.services.acp.client import ACPClient
from pipecat.services.acp.frames import (
    ACPAgentMessageFrame,
    ACPAgentThoughtFrame,
    ACPClientResponseFrame,
    ACPPermissionRequestFrame,
    ACPPromptFrame,
    ACPSessionEndedFrame,
    ACPSessionStartedFrame,
    ACPToolCallFrame,
    ACPToolCallUpdateFrame,
    ACPTurnEndedFrame,
    ACPTurnStartedFrame,
)
from pipecat.services.acp.permissions import ACPAutoPermission
from pipecat.services.acp.service import ACPService
from pipecat.services.acp.types import (
    ClientCapabilities,
    StopReason,
    ToolCall,
    ToolCallStatus,
    ToolCallUpdate,
    text_block,
)
from pipecat.tests.utils import SleepFrame, run_test

FAKE_AGENT = [sys.executable, os.path.join(os.path.dirname(__file__), "acp_fake_agent.py")]


class TestACPClient(unittest.IsolatedAsyncioTestCase):
    """The protocol client, driven against a real agent subprocess."""

    async def test_initialize_and_new_session(self):
        client = ACPClient()
        await client.start(FAKE_AGENT, cwd=os.getcwd())
        try:
            init = await client.initialize(ClientCapabilities())
            self.assertEqual(init.protocol_version, 1)
            self.assertTrue(init.agent_capabilities.load_session)

            session = await client.new_session(os.getcwd())
            self.assertEqual(session.session_id, "test-session")
            self.assertEqual(session.modes.current_mode_id, "default")
        finally:
            await client.stop()

    async def test_prompt_streams_updates_and_serves_requests(self):
        updates = []
        client = ACPClient()
        client.on_session_update = lambda params: _collect(updates, params)
        client.on_client_request = _allow_permission

        await client.start(FAKE_AGENT, cwd=os.getcwd())
        try:
            await client.initialize(ClientCapabilities())
            session = await client.new_session(os.getcwd())
            stop_reason = await client.prompt(session.session_id, [text_block("hello")])
        finally:
            await client.stop()

        self.assertEqual(stop_reason, StopReason.END_TURN)
        kinds = [u["sessionUpdate"] for u in updates]
        self.assertEqual(
            kinds,
            [
                "agent_thought_chunk",
                "tool_call",
                "tool_call_update",
                "agent_message_chunk",
            ],
        )


async def _collect(sink, params):
    sink.append(params)


async def _allow_permission(request_id, method, params):
    assert method == "session/request_permission"
    return {"outcome": {"outcome": "selected", "optionId": "yes"}}


class TestACPUserAggregator(unittest.IsolatedAsyncioTestCase):
    """Transcriptions collected into one prompt per turn."""

    async def test_emits_one_prompt_per_turn(self):
        """Transcriptions arriving after the turn ends still make the prompt."""
        aggregator = ACPUserAggregator(aggregation_timeout=0.1)
        frames_to_send = [
            UserStartedSpeakingFrame(),
            TranscriptionFrame(text="what does", user_id="u", timestamp=""),
            UserStoppedSpeakingFrame(),
            TranscriptionFrame(text="worker.py do", user_id="u", timestamp=""),
            SleepFrame(sleep=0.3),
        ]
        expected_down_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            TranscriptionFrame,
            TranscriptionFrame,
            ACPPromptFrame,
        ]

        received_down, _ = await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        prompt = received_down[-1]
        self.assertEqual(prompt.blocks[0].text, "what does worker.py do")

    async def test_silent_turn_emits_no_prompt(self):
        aggregator = ACPUserAggregator(aggregation_timeout=0.1)
        await run_test(
            aggregator,
            frames_to_send=[
                UserStartedSpeakingFrame(),
                UserStoppedSpeakingFrame(),
                SleepFrame(sleep=0.3),
            ],
            expected_down_frames=[UserStartedSpeakingFrame, UserStoppedSpeakingFrame],
        )


class TestACPService(unittest.IsolatedAsyncioTestCase):
    """The service, bridging a real agent subprocess to frames."""

    async def test_prompt_produces_acp_frames(self):
        pipeline = Pipeline([ACPService(command=FAKE_AGENT, cwd=os.getcwd()), ACPAutoPermission()])

        frames_to_send = [
            ACPPromptFrame(blocks=[text_block("hello")]),
            SleepFrame(sleep=1.0),
        ]
        expected_down_frames = [
            ACPSessionStartedFrame,
            ACPTurnStartedFrame,
            ACPAgentThoughtFrame,
            ACPToolCallFrame,
            ACPPermissionRequestFrame,
            ACPClientResponseFrame,
            ACPToolCallUpdateFrame,
            ACPAgentMessageFrame,
            ACPTurnEndedFrame,
            ACPSessionEndedFrame,
        ]

        received_down, _ = await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        turn_ended = next(f for f in received_down if isinstance(f, ACPTurnEndedFrame))
        self.assertEqual(turn_ended.stop_reason, StopReason.END_TURN)

        # The update omits the title; the service merges it from the original call.
        update = next(f for f in received_down if isinstance(f, ACPToolCallUpdateFrame))
        self.assertEqual(update.tool_call.title, "Read worker.py")
        self.assertEqual(update.tool_call.status, ToolCallStatus.COMPLETED)

    async def test_tool_call_update_carries_merged_call(self):
        """An update omits the title, so the service supplies it from the original."""
        service = ACPService(command=FAKE_AGENT, cwd=os.getcwd())
        service._tool_calls["call-1"] = ToolCall(
            tool_call_id="call-1", title="Read worker.py", status=ToolCallStatus.PENDING
        )

        merged = service._merge_tool_call(
            ToolCallUpdate(tool_call_id="call-1", status=ToolCallStatus.COMPLETED)
        )

        self.assertEqual(merged.title, "Read worker.py")
        self.assertEqual(merged.status, ToolCallStatus.COMPLETED)

    async def test_agent_exit_ends_session_and_errors(self):
        """A dead agent has to surface, not hang until a timeout."""
        service = ACPService(command=FAKE_AGENT + ["--die-after-session"], cwd=os.getcwd())

        received_down, received_up = await run_test(
            service,
            frames_to_send=[SleepFrame(sleep=0.5)],
            expected_down_frames=[ACPSessionStartedFrame, ACPSessionEndedFrame, CancelFrame],
            expected_up_frames=[ErrorFrame],
        )

        self.assertTrue(received_up[0].fatal)
        self.assertIn("exited", received_down[1].reason)

    async def test_failed_spawn_pushes_fatal_error(self):
        service = ACPService(command=["definitely-not-a-real-binary"], cwd=os.getcwd())

        _, received_up = await run_test(
            service,
            frames_to_send=[SleepFrame(sleep=0.2)],
            expected_up_frames=[ErrorFrame],
        )

        self.assertTrue(received_up[0].fatal)

    async def test_tool_call_update_appends_content(self):
        service = ACPService(command=FAKE_AGENT, cwd=os.getcwd())
        service._tool_calls["call-1"] = ToolCall(
            tool_call_id="call-1", content=[{"type": "content", "text": "first"}]
        )

        merged = service._merge_tool_call(
            ToolCallUpdate(tool_call_id="call-1", content=[{"type": "content", "text": "second"}])
        )

        self.assertEqual(len(merged.content), 2)


if __name__ == "__main__":
    unittest.main()
