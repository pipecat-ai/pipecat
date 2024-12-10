#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.frames.frames import (
    BotInterruptionFrame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantContextAggregator,
    LLMFullResponseAggregator,
    LLMUserContextAggregator,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from tests.utils import run_test


class TestLLMUserContextAggregator(unittest.IsolatedAsyncioTestCase):
    # S E ->
    async def test_s_e(self):
        """S E case"""
        context_aggregator = LLMUserContextAggregator(
            OpenAILLMContext(messages=[{"role": "", "content": ""}])
        )
        frames_to_send = [
            StartInterruptionFrame(),
            UserStartedSpeakingFrame(),
            StopInterruptionFrame(),
            UserStoppedSpeakingFrame(),
        ]
        expected_returned_frames = [
            StartInterruptionFrame,
            UserStartedSpeakingFrame,
            StopInterruptionFrame,
            UserStoppedSpeakingFrame,
        ]
        await run_test(context_aggregator, frames_to_send, expected_returned_frames)

    #    S T E           -> T
    async def test_s_t_e(self):
        """S T E case"""
        context_aggregator = LLMUserContextAggregator(
            OpenAILLMContext(messages=[{"role": "", "content": ""}])
        )
        frames_to_send = [
            StartInterruptionFrame(),
            UserStartedSpeakingFrame(),
            TranscriptionFrame("Hello", "", ""),
            StopInterruptionFrame(),
            UserStoppedSpeakingFrame(),
        ]
        expected_returned_frames = [
            StartInterruptionFrame,
            UserStartedSpeakingFrame,
            StopInterruptionFrame,
            UserStoppedSpeakingFrame,
            OpenAILLMContextFrame,
        ]
        await run_test(context_aggregator, frames_to_send, expected_returned_frames)

    #    S I T E         -> T
    async def test_s_i_t_e(self):
        """S I T E case"""
        context_aggregator = LLMUserContextAggregator(
            OpenAILLMContext(messages=[{"role": "", "content": ""}])
        )
        frames_to_send = [
            StartInterruptionFrame(),
            UserStartedSpeakingFrame(),
            InterimTranscriptionFrame("This", "", ""),
            TranscriptionFrame("This is a test", "", ""),
            StopInterruptionFrame(),
            UserStoppedSpeakingFrame(),
        ]
        expected_returned_frames = [
            StartInterruptionFrame,
            UserStartedSpeakingFrame,
            StopInterruptionFrame,
            UserStoppedSpeakingFrame,
            OpenAILLMContextFrame,
        ]
        await run_test(context_aggregator, frames_to_send, expected_returned_frames)

    #    S I E T         -> T
    async def test_s_i_e_t(self):
        """S I E T case"""
        context_aggregator = LLMUserContextAggregator(
            OpenAILLMContext(messages=[{"role": "", "content": ""}])
        )
        frames_to_send = [
            StartInterruptionFrame(),
            UserStartedSpeakingFrame(),
            InterimTranscriptionFrame("This", "", ""),
            StopInterruptionFrame(),
            UserStoppedSpeakingFrame(),
            TranscriptionFrame("This is a test", "", ""),
        ]
        expected_returned_frames = [
            StartInterruptionFrame,
            UserStartedSpeakingFrame,
            StopInterruptionFrame,
            UserStoppedSpeakingFrame,
            OpenAILLMContextFrame,
        ]
        await run_test(context_aggregator, frames_to_send, expected_returned_frames)

    #    S I E I T       -> T
    async def test_s_i_e_i_t(self):
        """S I E I T case"""
        context_aggregator = LLMUserContextAggregator(
            OpenAILLMContext(messages=[{"role": "", "content": ""}])
        )
        frames_to_send = [
            StartInterruptionFrame(),
            UserStartedSpeakingFrame(),
            InterimTranscriptionFrame("This", "", ""),
            StopInterruptionFrame(),
            UserStoppedSpeakingFrame(),
            InterimTranscriptionFrame("This is", "", ""),
            TranscriptionFrame("This is a test", "", ""),
        ]
        expected_returned_frames = [
            StartInterruptionFrame,
            UserStartedSpeakingFrame,
            StopInterruptionFrame,
            UserStoppedSpeakingFrame,
            OpenAILLMContextFrame,
        ]
        await run_test(context_aggregator, frames_to_send, expected_returned_frames)

    #    S E T           -> T
    async def test_s_e_t(self):
        """S E case"""
        context_aggregator = LLMUserContextAggregator(
            OpenAILLMContext(messages=[{"role": "", "content": ""}])
        )
        frames_to_send = [
            StartInterruptionFrame(),
            UserStartedSpeakingFrame(),
            StopInterruptionFrame(),
            UserStoppedSpeakingFrame(),
            TranscriptionFrame("This is a test", "", ""),
        ]
        expected_returned_frames = [
            StartInterruptionFrame,
            UserStartedSpeakingFrame,
            StopInterruptionFrame,
            UserStoppedSpeakingFrame,
            OpenAILLMContextFrame,
        ]
        await run_test(context_aggregator, frames_to_send, expected_returned_frames)

    #    S E I T         -> T
    async def test_s_e_i_t(self):
        """S E I T case"""
        context_aggregator = LLMUserContextAggregator(
            OpenAILLMContext(messages=[{"role": "", "content": ""}])
        )
        frames_to_send = [
            StartInterruptionFrame(),
            UserStartedSpeakingFrame(),
            StopInterruptionFrame(),
            UserStoppedSpeakingFrame(),
            InterimTranscriptionFrame("This", "", ""),
            TranscriptionFrame("This is a test", "", ""),
        ]
        expected_returned_frames = [
            StartInterruptionFrame,
            UserStartedSpeakingFrame,
            StopInterruptionFrame,
            UserStoppedSpeakingFrame,
            OpenAILLMContextFrame,
        ]
        await run_test(context_aggregator, frames_to_send, expected_returned_frames)

    #    S T1 I E S T2 E -> "T1 T2"
    async def test_s_t1_i_e_s_t2_e(self):
        """S T1 I E S T2 E case"""
        context_aggregator = LLMUserContextAggregator(
            OpenAILLMContext(messages=[{"role": "", "content": ""}])
        )
        frames_to_send = [
            StartInterruptionFrame(),
            UserStartedSpeakingFrame(),
            TranscriptionFrame("T1", "", ""),
            InterimTranscriptionFrame("", "", ""),
            StopInterruptionFrame(),
            UserStoppedSpeakingFrame(),
            StartInterruptionFrame(),
            UserStartedSpeakingFrame(),
            TranscriptionFrame("T2", "", ""),
            StopInterruptionFrame(),
            UserStoppedSpeakingFrame(),
        ]
        expected_returned_frames = [
            StartInterruptionFrame,
            UserStartedSpeakingFrame,
            StopInterruptionFrame,
            UserStoppedSpeakingFrame,
            StartInterruptionFrame,
            UserStartedSpeakingFrame,
            StopInterruptionFrame,
            UserStoppedSpeakingFrame,
            OpenAILLMContextFrame,
        ]
        (received_down, _) = await run_test(
            context_aggregator, frames_to_send, expected_returned_frames
        )
        assert received_down[-1].context.messages[-1]["content"] == "T1 T2"

    #    S I E T1 I T2   -> T1 Interruption T2
    async def test_s_i_e_t1_i_t2(self):
        """S I E T1 I T2 case"""
        context_aggregator = LLMUserContextAggregator(
            OpenAILLMContext(messages=[{"role": "", "content": ""}])
        )
        frames_to_send = [
            StartInterruptionFrame(),
            UserStartedSpeakingFrame(),
            InterimTranscriptionFrame("", "", ""),
            StopInterruptionFrame(),
            UserStoppedSpeakingFrame(),
            TranscriptionFrame("T1", "", ""),
            InterimTranscriptionFrame("", "", ""),
            TranscriptionFrame("T2", "", ""),
        ]
        expected_down_frames = [
            StartInterruptionFrame,
            UserStartedSpeakingFrame,
            StopInterruptionFrame,
            UserStoppedSpeakingFrame,
            OpenAILLMContextFrame,
            OpenAILLMContextFrame,
        ]
        expected_up_frames = [
            BotInterruptionFrame,
        ]
        (received_down, _) = await run_test(
            context_aggregator, frames_to_send, expected_down_frames, expected_up_frames
        )
        assert received_down[-1].context.messages[-2]["content"] == "T1"
        assert received_down[-1].context.messages[-1]["content"] == "T2"

    #    S T1 E T2       -> T1 Interruption T2
    async def test_s_t1_e_t2(self):
        """S T1 E T2 case"""
        context_aggregator = LLMUserContextAggregator(
            OpenAILLMContext(messages=[{"role": "", "content": ""}])
        )
        frames_to_send = [
            StartInterruptionFrame(),
            UserStartedSpeakingFrame(),
            TranscriptionFrame("T1", "", ""),
            StopInterruptionFrame(),
            UserStoppedSpeakingFrame(),
            TranscriptionFrame("T2", "", ""),
        ]
        expected_down_frames = [
            StartInterruptionFrame,
            UserStartedSpeakingFrame,
            StopInterruptionFrame,
            UserStoppedSpeakingFrame,
            OpenAILLMContextFrame,
            OpenAILLMContextFrame,
        ]
        expected_up_frames = [
            BotInterruptionFrame,
        ]
        (received_down, _) = await run_test(
            context_aggregator, frames_to_send, expected_down_frames, expected_up_frames
        )
        assert received_down[-1].context.messages[-2]["content"] == "T1"
        assert received_down[-1].context.messages[-1]["content"] == "T2"

    #    S E T1 T2       -> T1 Interruption T2
    async def test_s_e_t1_t2(self):
        """S E T1 T2 case"""
        context_aggregator = LLMUserContextAggregator(
            OpenAILLMContext(messages=[{"role": "", "content": ""}])
        )
        frames_to_send = [
            StartInterruptionFrame(),
            UserStartedSpeakingFrame(),
            StopInterruptionFrame(),
            UserStoppedSpeakingFrame(),
            TranscriptionFrame("T1", "", ""),
            TranscriptionFrame("T2", "", ""),
        ]
        expected_down_frames = [
            StartInterruptionFrame,
            UserStartedSpeakingFrame,
            StopInterruptionFrame,
            UserStoppedSpeakingFrame,
            OpenAILLMContextFrame,
            OpenAILLMContextFrame,
        ]
        expected_up_frames = [
            BotInterruptionFrame,
        ]
        (received_down, _) = await run_test(
            context_aggregator, frames_to_send, expected_down_frames, expected_up_frames
        )
        assert received_down[-1].context.messages[-2]["content"] == "T1"
        assert received_down[-1].context.messages[-1]["content"] == "T2"


class TestLLMAssistantContextAggregator(unittest.IsolatedAsyncioTestCase):
    #    S T E           -> T
    async def test_s_t_e(self):
        """S T E case"""
        context_aggregator = LLMAssistantContextAggregator(
            OpenAILLMContext(messages=[{"role": "", "content": ""}])
        )
        frames_to_send = [
            LLMFullResponseStartFrame(),
            TextFrame("Hello this is Pipecat speaking!"),
            TextFrame("How are you?"),
            LLMFullResponseEndFrame(),
        ]
        expected_returned_frames = [
            LLMFullResponseStartFrame,
            OpenAILLMContextFrame,
            LLMFullResponseEndFrame,
        ]
        (received_down, _) = await run_test(
            context_aggregator, frames_to_send, expected_returned_frames
        )
        assert (
            received_down[-2].context.messages[-1]["content"]
            == "Hello this is Pipecat speaking! How are you?"
        )


class TestLLMFullResponseAggregator(unittest.IsolatedAsyncioTestCase):
    #    S T E           -> T
    async def test_s_t_e(self):
        """S T E case"""
        response_aggregator = LLMFullResponseAggregator()
        frames_to_send = [
            LLMFullResponseStartFrame(),
            TextFrame("Hello "),
            TextFrame("this "),
            TextFrame("is "),
            TextFrame("Pipecat!"),
            LLMFullResponseEndFrame(),
        ]
        expected_returned_frames = [
            LLMFullResponseStartFrame,
            TextFrame,
            LLMFullResponseEndFrame,
        ]
        (received_down, _) = await run_test(
            response_aggregator, frames_to_send, expected_returned_frames
        )
        assert received_down[-2].text == "Hello this is Pipecat!"
