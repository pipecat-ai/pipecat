#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.evals.conversation_observer import EvalConversationObserver
from pipecat.frames.frames import (
    ExternalFunctionCallInProgressFrame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    LLMContextAssistantTurnFrame,
    LLMContextUserTurnFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi.frames import RTVISendTextFrame
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.user_start import ExternalUserTurnStartStrategy
from pipecat.turns.user_stop import ExternalUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.asyncio.task_manager import TaskManager


def _in_progress(name: str, arguments: dict) -> FunctionCallInProgressFrame:
    return FunctionCallInProgressFrame(function_name=name, tool_call_id=name, arguments=arguments)


class TestEvalConversationObserver(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.observer = EvalConversationObserver()
        # Event handlers run as tasks, so the observer needs a task manager.
        await self.observer.setup(TaskManager())
        self.reported: list[list[dict]] = []

        @self.observer.event_handler("on_conversation_ended")
        async def on_conversation_ended(observer, message):
            self.reported.append(message.conversation)

    async def _push(self, frame, direction=FrameDirection.DOWNSTREAM):
        """Feed one frame to the observer, as its first push in a pipeline would."""
        await self.observer.on_push_frame(
            FramePushed(
                source=IdentityFilter(name="source"),
                destination=IdentityFilter(name="destination"),
                frame=frame,
                direction=direction,
                timestamp=0,
            )
        )

    async def test_the_turns_and_calls_are_reported_as_a_conversation_once_the_pipeline_is_done(
        self,
    ):
        await self._push(LLMContextUserTurnFrame(text="A table at six.", timestamp="t"))
        # A broadcast call arrives as two frames: the upstream one is the same call.
        downstream, upstream = _in_progress("check", {"time": "6"}), _in_progress("check", {})
        downstream.broadcast_sibling_id, upstream.broadcast_sibling_id = upstream.id, downstream.id
        await self._push(downstream)
        await self._push(upstream, FrameDirection.UPSTREAM)
        await self._push(LLMContextAssistantTurnFrame(text="Let me check.", timestamp="t"))
        await self._push(
            ExternalFunctionCallInProgressFrame("book", "book", arguments={"time": "6"})
        )
        await self._push(FunctionCallCancelFrame(function_name="book", tool_call_id="book"))
        await self._push(LLMContextAssistantTurnFrame(text="Six is free.", timestamp="t"))
        # Text a client sent is a user turn as well.
        await self._push(RTVISendTextFrame(msg_id="1", text="Thanks."))
        await self._push(LLMContextAssistantTurnFrame(text="Bye!", timestamp="t"))
        self.assertEqual(self.reported, [])

        await self.observer.cleanup()

        self.assertEqual(
            self.reported,
            [
                [
                    {"role": "user", "content": "A table at six."},
                    {
                        "role": "tool_call",
                        "name": "check",
                        "arguments": {"time": "6"},
                        "cancelled": False,
                    },
                    {
                        "role": "tool_call",
                        "name": "book",
                        "arguments": {"time": "6"},
                        "cancelled": True,
                    },
                    {"role": "assistant", "content": "Let me check. Six is free."},
                    {"role": "user", "content": "Thanks."},
                    {"role": "assistant", "content": "Bye!"},
                ]
            ],
        )

    async def test_the_aggregators_turns_reach_the_conversation_through_a_pipeline(self):
        context = LLMContext()
        aggregators = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(
                    start=[ExternalUserTurnStartStrategy()],
                    stop=[ExternalUserTurnStopStrategy(timeout=60.0)],
                )
            ),
        )
        observer = EvalConversationObserver()
        reported: list[list[dict]] = []

        @observer.event_handler("on_conversation_ended")
        async def on_conversation_ended(observer, message):
            reported.append(message.conversation)

        await run_test(
            Pipeline([aggregators.user(), aggregators.assistant()]),
            frames_to_send=[
                ProposedUserStartedSpeakingFrame(),
                TranscriptionFrame(text="A table at six, please.", user_id="", timestamp="now"),
                ProposedUserStoppedSpeakingFrame(),
                SleepFrame(sleep=0.1),
                LLMFullResponseStartFrame(),
                LLMTextFrame(text="Six is free."),
                LLMFullResponseEndFrame(),
                SleepFrame(sleep=0.1),
            ],
            expected_down_frames=None,
            observers=[observer],
        )

        self.assertEqual(
            reported,
            [
                [
                    {"role": "user", "content": "A table at six, please."},
                    {"role": "assistant", "content": "Six is free."},
                ]
            ],
        )


if __name__ == "__main__":
    unittest.main()
