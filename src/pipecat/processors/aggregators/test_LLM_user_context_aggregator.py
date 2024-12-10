# tests/test_custom_user_context.py

"""Tests for CustomLLMUserContextAggregator"""

import asyncio
import unittest

from dataclasses import dataclass
from typing import List

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    ControlFrame,
    Frame,
    StartFrame,
    TranscriptionFrame,
    InterimTranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_response import LLMUserContextAggregator
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# Note that UserStartedSpeakingFrame always come with StartInterruptionFrame
# and       UserStoppedSpeakingFrame always come with StopInterruptionFrame
#    S E             -> None
#    S T E           -> T
#    S I T E         -> T
#    S I E T         -> T
#    S I E I T       -> T
#    S E T           -> T
#    S E I T         -> T
#    S T1 I E S T2 E -> (T1 T2)
#    S I E T1 I T2   -> T1 Interruption T2
#    S T1 E T2       -> T1 Interruption T2
#    S E T1 B T2     -> T1 Bot Interruption T2
#    S E T1 T2       -> T1 Interruption T2


@dataclass
class EndTestFrame(ControlFrame):
    pass


class QueuedFrameProcessor(FrameProcessor):
    def __init__(self, queue: asyncio.Queue, ignore_start: bool = True):
        super().__init__()
        self._queue = queue
        self._ignore_start = ignore_start

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if self._ignore_start and isinstance(frame, StartFrame):
            return
        await self._queue.put(frame)


async def make_test(
    frames_to_send: List[Frame], expected_returned_frames: List[type]
) -> List[Frame]:
    context_aggregator = LLMUserContextAggregator(
        OpenAILLMContext(messages=[{"role": "", "content": ""}])
    )

    received = asyncio.Queue()
    test_processor = QueuedFrameProcessor(received)
    context_aggregator.link(test_processor)

    await context_aggregator.queue_frame(StartFrame(clock=SystemClock()))
    for frame in frames_to_send:
        await context_aggregator.process_frame(frame, direction=FrameDirection.DOWNSTREAM)
    await context_aggregator.queue_frame(EndTestFrame())

    received_frames: List[Frame] = []
    running = True
    while running:
        frame = await received.get()
        running = not isinstance(frame, EndTestFrame)
        if running:
            received_frames.append(frame)

    assert len(received_frames) == len(expected_returned_frames)
    for real, expected in zip(received_frames, expected_returned_frames):
        assert isinstance(real, expected)
    return received_frames


class TestFrameProcessing(unittest.IsolatedAsyncioTestCase):
    # S E ->
    async def test_s_e(self):
        """S E case"""
        frames_to_send = [
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
        ]
        expected_returned_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
        ]
        await make_test(frames_to_send, expected_returned_frames)

    #    S T E           -> T
    async def test_s_t_e(self):
        """S T E case"""
        frames_to_send = [
            UserStartedSpeakingFrame(),
            TranscriptionFrame("Hello", "", ""),
            UserStoppedSpeakingFrame(),
        ]
        expected_returned_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            OpenAILLMContextFrame,
        ]
        await make_test(frames_to_send, expected_returned_frames)

    #    S I T E         -> T
    async def test_s_i_t_e(self):
        """S I T E case"""
        frames_to_send = [
            UserStartedSpeakingFrame(),
            InterimTranscriptionFrame("This", "", ""),
            TranscriptionFrame("This is a test", "", ""),
            UserStoppedSpeakingFrame(),
        ]
        expected_returned_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            OpenAILLMContextFrame,
        ]
        await make_test(frames_to_send, expected_returned_frames)

    #    S I E T         -> T
    async def test_s_i_e_t(self):
        """S I E T case"""
        frames_to_send = [
            UserStartedSpeakingFrame(),
            InterimTranscriptionFrame("This", "", ""),
            UserStoppedSpeakingFrame(),
            TranscriptionFrame("This is a test", "", ""),
        ]
        expected_returned_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            OpenAILLMContextFrame,
        ]
        await make_test(frames_to_send, expected_returned_frames)

    #    S I E I T       -> T
    async def test_s_i_e_i_t(self):
        """S I E I T case"""
        frames_to_send = [
            UserStartedSpeakingFrame(),
            InterimTranscriptionFrame("This", "", ""),
            UserStoppedSpeakingFrame(),
            InterimTranscriptionFrame("This is", "", ""),
            TranscriptionFrame("This is a test", "", ""),
        ]
        expected_returned_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            OpenAILLMContextFrame,
        ]
        await make_test(frames_to_send, expected_returned_frames)

    #    S E T           -> T
    async def test_s_e_t(self):
        """S E case"""
        frames_to_send = [
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            TranscriptionFrame("This is a test", "", ""),
        ]
        expected_returned_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            OpenAILLMContextFrame,
        ]
        await make_test(frames_to_send, expected_returned_frames)

    #    S E I T         -> T
    async def test_s_e_i_t(self):
        """S E I T case"""
        frames_to_send = [
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            InterimTranscriptionFrame("This", "", ""),
            TranscriptionFrame("This is a test", "", ""),
        ]
        expected_returned_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            OpenAILLMContextFrame,
        ]
        await make_test(frames_to_send, expected_returned_frames)

    #    S T1 I E S T2 E -> "T1 T2"
    async def test_s_t1_i_e_s_t2_e(self):
        """S T1 I E S T2 E case"""
        frames_to_send = [
            UserStartedSpeakingFrame(),
            TranscriptionFrame("T1", "", ""),
            InterimTranscriptionFrame("", "", ""),
            UserStoppedSpeakingFrame(),
            UserStartedSpeakingFrame(),
            TranscriptionFrame("T2", "", ""),
            UserStoppedSpeakingFrame(),
        ]
        expected_returned_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            OpenAILLMContextFrame,
        ]
        result = await make_test(frames_to_send, expected_returned_frames)
        assert result[-1].context.messages[-1]["content"] == "T1 T2"

    #    S I E T1 I T2   -> T1 Interruption T2
    async def test_s_i_e_t1_i_t2(self):
        """S I E T1 I T2 case"""
        frames_to_send = [
            UserStartedSpeakingFrame(),
            InterimTranscriptionFrame("", "", ""),
            UserStoppedSpeakingFrame(),
            TranscriptionFrame("T1", "", ""),
            InterimTranscriptionFrame("", "", ""),
            TranscriptionFrame("T2", "", ""),
        ]
        expected_returned_frames = [
            UserStartedSpeakingFrame,
            UserStoppedSpeakingFrame,
            OpenAILLMContextFrame,
            OpenAILLMContextFrame,
        ]
        result = await make_test(frames_to_send, expected_returned_frames)
        assert result[-2].context.messages[-2]["content"] == "T1"
        assert result[-1].context.messages[-1]["content"] == "T2"

    # #    S T1 E T2       -> T1 Interruption T2
    # async def test_s_t1_e_t2(self):
    #     """S T1 E T2 case"""
    #     frames_to_send = [
    #         UserStartedSpeakingFrame(),
    #         TranscriptionFrame("T1", "", ""),
    #         UserStoppedSpeakingFrame(),
    #         TranscriptionFrame("T2", "", ""),
    #     ]
    #     expected_returned_frames = [
    #         UserStartedSpeakingFrame,
    #         UserStoppedSpeakingFrame,
    #         OpenAILLMContextFrame,
    #         OpenAILLMContextFrame,
    #     ]
    #     result = await make_test(frames_to_send, expected_returned_frames)
    #     assert result[-1].context.messages[-1]["content"] == " T1 T2"

    # #    S E T1 T2       -> T1 Interruption T2
    # async def test_s_e_t1_t2(self):
    #     """S E T1 T2 case"""
    #     frames_to_send = [
    #         UserStartedSpeakingFrame(),
    #         UserStoppedSpeakingFrame(),
    #         TranscriptionFrame("T1", "", ""),
    #         TranscriptionFrame("T2", "", ""),
    #     ]
    #     expected_returned_frames = [
    #         UserStartedSpeakingFrame,
    #         UserStoppedSpeakingFrame,
    #         OpenAILLMContextFrame,
    #         OpenAILLMContextFrame,
    #     ]
    #     result = await make_test(frames_to_send, expected_returned_frames)
    #     assert result[-1].context.messages[-1]["content"] == " T1 T2"
