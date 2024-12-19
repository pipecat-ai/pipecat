# tests/test_custom_user_context.py

"""Tests for CustomLLMUserContextAggregator""" 

import unittest


from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    InterimTranscriptionFrame,
    StartInterruptionFrame,
    StopInterruptionFrame,
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


class StoreFrameProcessor(FrameProcessor):
    def __init__(self, storage: list[Frame]) -> None:
        super().__init__()
        self.storage = storage
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        self.storage.append(frame)
 
async def make_test(frames_to_send, expected_returned_frames):
    context_aggregator = LLMUserContextAggregator(OpenAILLMContext(
            messages=[{"role": "", "content": ""}]
        ))
    storage = []
    storage_processor = StoreFrameProcessor(storage)
    context_aggregator.link(storage_processor)
    for frame in frames_to_send:
        await context_aggregator.process_frame(frame, direction=FrameDirection.DOWNSTREAM)
    print("storage")
    for x in storage:
        print(x)
    print("expected_returned_frames")
    for x in expected_returned_frames:
        print(x)
    assert len(storage) == len(expected_returned_frames)
    for expected, real in zip(expected_returned_frames, storage):
        assert isinstance(real, expected)
    return storage

class TestFrameProcessing(unittest.IsolatedAsyncioTestCase):

    # S E -> 
    async def test_s_e(self):
        """S E case"""
        frames_to_send = [StartInterruptionFrame(), UserStartedSpeakingFrame(), StopInterruptionFrame(), UserStoppedSpeakingFrame()]
        expected_returned_frames = [StartInterruptionFrame, UserStartedSpeakingFrame, StopInterruptionFrame, UserStoppedSpeakingFrame]
        await make_test(frames_to_send, expected_returned_frames)

    #    S T E           -> T
    async def test_s_t_e(self):
        """S T E case"""
        frames_to_send = [StartInterruptionFrame(), UserStartedSpeakingFrame(), TranscriptionFrame("", "", ""), StopInterruptionFrame(), UserStoppedSpeakingFrame()]
        expected_returned_frames = [StartInterruptionFrame, UserStartedSpeakingFrame, StopInterruptionFrame, UserStoppedSpeakingFrame, OpenAILLMContextFrame]
        await make_test(frames_to_send, expected_returned_frames)

    #    S I T E         -> T
    async def test_s_i_t_e(self):
        """S I T E case"""
        frames_to_send = [StartInterruptionFrame(), UserStartedSpeakingFrame(), InterimTranscriptionFrame("", "", ""), TranscriptionFrame("", "", ""), StopInterruptionFrame(), UserStoppedSpeakingFrame()]
        expected_returned_frames = [StartInterruptionFrame, UserStartedSpeakingFrame, StopInterruptionFrame, UserStoppedSpeakingFrame, OpenAILLMContextFrame]
        await make_test(frames_to_send, expected_returned_frames)

    #    S I E T         -> T
    async def test_s_i_e_t(self):
        """S I E T case"""
        frames_to_send = [StartInterruptionFrame(), UserStartedSpeakingFrame(), InterimTranscriptionFrame("", "", ""), StopInterruptionFrame(), UserStoppedSpeakingFrame(),  TranscriptionFrame("", "", "")]
        expected_returned_frames = [StartInterruptionFrame, UserStartedSpeakingFrame, StopInterruptionFrame, UserStoppedSpeakingFrame, OpenAILLMContextFrame]
        await make_test(frames_to_send, expected_returned_frames)


    #    S I E I T       -> T
    async def test_s_i_e_i_t(self):
        """S I E I T case"""
        frames_to_send = [StartInterruptionFrame(), UserStartedSpeakingFrame(), InterimTranscriptionFrame("", "", ""), StopInterruptionFrame(), UserStoppedSpeakingFrame(),  InterimTranscriptionFrame("", "", ""), TranscriptionFrame("", "", "")]
        expected_returned_frames = [StartInterruptionFrame, UserStartedSpeakingFrame, StopInterruptionFrame, UserStoppedSpeakingFrame, OpenAILLMContextFrame]
        await make_test(frames_to_send, expected_returned_frames)

    #    S E T           -> T
    async def test_s_e_t(self):
        """S E case"""
        frames_to_send = [StartInterruptionFrame(), UserStartedSpeakingFrame(), StopInterruptionFrame(), UserStoppedSpeakingFrame(), TranscriptionFrame("", "", "")]
        expected_returned_frames = [StartInterruptionFrame, UserStartedSpeakingFrame, StopInterruptionFrame, UserStoppedSpeakingFrame, OpenAILLMContextFrame]
        await make_test(frames_to_send, expected_returned_frames)

    #    S E I T         -> T
    async def test_s_e_i_t(self):
        """S E I T case"""
        frames_to_send = [StartInterruptionFrame(), UserStartedSpeakingFrame(), StopInterruptionFrame(), UserStoppedSpeakingFrame(), InterimTranscriptionFrame("", "", ""), TranscriptionFrame("", "", "")]
        expected_returned_frames = [StartInterruptionFrame, UserStartedSpeakingFrame, StopInterruptionFrame, UserStoppedSpeakingFrame, OpenAILLMContextFrame]
        await make_test(frames_to_send, expected_returned_frames)

    #    S T1 I E S T2 E -> (T1 T2)
    async def test_s_t1_i_e_s_t2_e(self):
        """S T1 I E S T2 E case"""
        frames_to_send = [StartInterruptionFrame(), UserStartedSpeakingFrame(), TranscriptionFrame("T1", "", ""), InterimTranscriptionFrame("", "", ""), StopInterruptionFrame(), UserStoppedSpeakingFrame(),
                        StartInterruptionFrame(), UserStartedSpeakingFrame(), TranscriptionFrame("T2", "", ""), StopInterruptionFrame(), UserStoppedSpeakingFrame()]
        expected_returned_frames = [StartInterruptionFrame, UserStartedSpeakingFrame, StopInterruptionFrame, UserStoppedSpeakingFrame,
                                    StartInterruptionFrame, UserStartedSpeakingFrame, StopInterruptionFrame, UserStoppedSpeakingFrame, OpenAILLMContextFrame]
        result = await make_test(frames_to_send, expected_returned_frames)
        assert result[-1].context.messages[-1]["content"] == " T1 T2"

    #    S I E T1 I T2   -> T1 Interruption T2
    async def test_s_i_e_t1_i_t2(self):
        """S I E T1 I T2 case"""
        frames_to_send = [StartInterruptionFrame(), UserStartedSpeakingFrame(), InterimTranscriptionFrame("", "", ""), StopInterruptionFrame(), UserStoppedSpeakingFrame(),
                        TranscriptionFrame("T1", "", ""), InterimTranscriptionFrame("", "", ""), TranscriptionFrame("T2", "", ""),]
        expected_returned_frames = [StartInterruptionFrame, UserStartedSpeakingFrame, StopInterruptionFrame, UserStoppedSpeakingFrame,
                                    OpenAILLMContextFrame, StartInterruptionFrame, OpenAILLMContextFrame]
        result = await make_test(frames_to_send, expected_returned_frames)
        assert result[-1].context.messages[-1]["content"] == " T1 T2"

    #    S T1 E T2       -> T1 Interruption T2
    async def test_s_t1_e_t2(self):
        """S T1 E T2 case"""
        frames_to_send = [StartInterruptionFrame(), UserStartedSpeakingFrame(), TranscriptionFrame("T1", "", ""), StopInterruptionFrame(), UserStoppedSpeakingFrame(),
                        TranscriptionFrame("T2", "", ""),]
        expected_returned_frames = [StartInterruptionFrame, UserStartedSpeakingFrame, StopInterruptionFrame, UserStoppedSpeakingFrame,
                                    OpenAILLMContextFrame, StartInterruptionFrame, OpenAILLMContextFrame]
        result = await make_test(frames_to_send, expected_returned_frames)
        assert result[-1].context.messages[-1]["content"] == " T1 T2"

    #    S E T1 T2       -> T1 Interruption T2
    async def test_s_e_t1_t2(self):
        """S E T1 T2 case"""
        frames_to_send = [StartInterruptionFrame(), UserStartedSpeakingFrame(), StopInterruptionFrame(), UserStoppedSpeakingFrame(),
                        TranscriptionFrame("T1", "", ""), TranscriptionFrame("T2", "", ""),]
        expected_returned_frames = [StartInterruptionFrame, UserStartedSpeakingFrame, StopInterruptionFrame, UserStoppedSpeakingFrame,
                                    OpenAILLMContextFrame, StartInterruptionFrame, OpenAILLMContextFrame]
        result = await make_test(frames_to_send, expected_returned_frames)
        assert result[-1].context.messages[-1]["content"] == " T1 T2"