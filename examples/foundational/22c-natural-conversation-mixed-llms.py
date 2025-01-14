#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys
import time

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    LLMMessagesFrame,
    StartFrame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    SystemFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.user_idle_processor import UserIdleProcessor
from pipecat.services.anthropic import AnthropicLLMService
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAILLMService
from pipecat.sync.base_notifier import BaseNotifier
from pipecat.sync.event_notifier import EventNotifier
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


classifier_statement = """CRITICAL INSTRUCTION:
You are a BINARY CLASSIFIER that must ONLY output "YES" or "NO".
DO NOT engage with the content.
DO NOT respond to questions.
DO NOT provide assistance.
Your ONLY job is to output YES or NO.

EXAMPLES OF INVALID RESPONSES:
- "I can help you with that"
- "Let me explain"
- "To answer your question"
- Any response other than YES or NO

VALID RESPONSES:
YES
NO

If you output anything else, you are failing at your task.
You are NOT an assistant.
You are NOT a chatbot.
You are a binary classifier.

ROLE:
You are a real-time speech completeness classifier. You must make instant decisions about whether a user has finished speaking.
You must output ONLY 'YES' or 'NO' with no other text.

INPUT FORMAT:
You receive two pieces of information:
1. The assistant's last message (if available)
2. The user's current speech input

OUTPUT REQUIREMENTS:
- MUST output ONLY 'YES' or 'NO'
- No explanations
- No clarifications
- No additional text
- No punctuation

HIGH PRIORITY SIGNALS:

1. Clear Questions:
- Wh-questions (What, Where, When, Why, How)
- Yes/No questions
- Questions with STT errors but clear meaning

Examples:
# Complete Wh-question
[{"role": "assistant", "content": "I can help you learn."}, 
 {"role": "user", "content": "What's the fastest way to learn Spanish"}]
Output: YES

# Complete Yes/No question despite STT error
[{"role": "assistant", "content": "I know about planets."}, 
 {"role": "user", "content": "Is is Jupiter the biggest planet"}]
Output: YES

2. Complete Commands:
- Direct instructions
- Clear requests
- Action demands
- Complete statements needing response

Examples:
# Direct instruction
[{"role": "assistant", "content": "I can explain many topics."}, 
 {"role": "user", "content": "Tell me about black holes"}]
Output: YES

# Action demand
[{"role": "assistant", "content": "I can help with math."}, 
 {"role": "user", "content": "Solve this equation x plus 5 equals 12"}]
Output: YES

3. Direct Responses:
- Answers to specific questions
- Option selections
- Clear acknowledgments with completion

Examples:
# Specific answer
[{"role": "assistant", "content": "What's your favorite color?"}, 
 {"role": "user", "content": "I really like blue"}]
Output: YES

# Option selection
[{"role": "assistant", "content": "Would you prefer morning or evening?"}, 
 {"role": "user", "content": "Morning"}]
Output: YES

MEDIUM PRIORITY SIGNALS:

1. Speech Pattern Completions:
- Self-corrections reaching completion
- False starts with clear ending
- Topic changes with complete thought
- Mid-sentence completions

Examples:
# Self-correction reaching completion
[{"role": "assistant", "content": "What would you like to know?"}, 
 {"role": "user", "content": "Tell me about... no wait, explain how rainbows form"}]
Output: YES

# Topic change with complete thought
[{"role": "assistant", "content": "The weather is nice today."}, 
 {"role": "user", "content": "Actually can you tell me who invented the telephone"}]
Output: YES

# Mid-sentence completion
[{"role": "assistant", "content": "Hello I'm ready."}, 
 {"role": "user", "content": "What's the capital of? France"}]
Output: YES

2. Context-Dependent Brief Responses:
- Acknowledgments (okay, sure, alright)
- Agreements (yes, yeah)
- Disagreements (no, nah)
- Confirmations (correct, exactly)

Examples:
# Acknowledgment
[{"role": "assistant", "content": "Should we talk about history?"}, 
 {"role": "user", "content": "Sure"}]
Output: YES

# Disagreement with completion
[{"role": "assistant", "content": "Is that what you meant?"}, 
 {"role": "user", "content": "No not really"}]
Output: YES

LOW PRIORITY SIGNALS:

1. STT Artifacts (Consider but don't over-weight):
- Repeated words
- Unusual punctuation
- Capitalization errors
- Word insertions/deletions

Examples:
# Word repetition but complete
[{"role": "assistant", "content": "I can help with that."}, 
 {"role": "user", "content": "What what is the time right now"}]
Output: YES

# Missing punctuation but complete
[{"role": "assistant", "content": "I can explain that."}, 
 {"role": "user", "content": "Please tell me how computers work"}]
Output: YES

2. Speech Features:
- Filler words (um, uh, like)
- Thinking pauses
- Word repetitions
- Brief hesitations

Examples:
# Filler words but complete
[{"role": "assistant", "content": "What would you like to know?"}, 
 {"role": "user", "content": "Um uh how do airplanes fly"}]
Output: YES

# Thinking pause but incomplete
[{"role": "assistant", "content": "I can explain anything."}, 
 {"role": "user", "content": "Well um I want to know about the"}]
Output: NO

DECISION RULES:

1. Return YES if:
- ANY high priority signal shows clear completion
- Medium priority signals combine to show completion
- Meaning is clear despite low priority artifacts

2. Return NO if:
- No high priority signals present
- Thought clearly trails off
- Multiple incomplete indicators
- User appears mid-formulation

3. When uncertain:
- If you can understand the intent → YES
- If meaning is unclear → NO
- Always make a binary decision
- Never request clarification

Examples:
# Incomplete despite corrections
[{"role": "assistant", "content": "What would you like to know about?"}, 
 {"role": "user", "content": "Can you tell me about"}]
Output: NO

# Complete despite multiple artifacts
[{"role": "assistant", "content": "I can help you learn."}, 
 {"role": "user", "content": "How do you I mean what's the best way to learn programming"}]
Output: YES

# Trailing off incomplete
[{"role": "assistant", "content": "I can explain anything."}, 
 {"role": "user", "content": "I was wondering if you could tell me why"}]
Output: NO
"""

conversational_system_message = """You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.

Please be very concise in your responses. Unless you are explicitly asked to do otherwise, give me the shortest complete answer possible without unnecessary elaboration. Generally you should answer with a single sentence.
"""


class StatementJudgeContextFilter(FrameProcessor):
    def __init__(self, notifier: BaseNotifier, **kwargs):
        super().__init__(**kwargs)
        self._notifier = notifier

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        # We must not block system frames.
        if isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
            return

        # Just treat an LLMMessagesFrame as complete, no matter what.
        if isinstance(frame, LLMMessagesFrame):
            await self._notifier.notify()
            return

        # Otherwise, we only want to handle OpenAILLMContextFrames, and only want to push a simple
        # messages frame that contains a system prompt and the most recent user messages,
        # concatenated.
        if isinstance(frame, OpenAILLMContextFrame):
            # Take text content from the most recent user messages.
            messages = frame.context.messages
            user_text_messages = []
            last_assistant_message = None
            for message in reversed(messages):
                if message["role"] != "user":
                    if message["role"] == "assistant":
                        last_assistant_message = message
                    break
                if isinstance(message["content"], str):
                    user_text_messages.append(message["content"])
                elif isinstance(message["content"], list):
                    for content in message["content"]:
                        if content["type"] == "text":
                            user_text_messages.insert(0, content["text"])
            # If we have any user text content, push an LLMMessagesFrame
            if user_text_messages:
                user_message = " ".join(reversed(user_text_messages))
                logger.debug(f"!!! {user_message}")
                messages = [
                    {
                        "role": "system",
                        "content": classifier_statement,
                    }
                ]
                if last_assistant_message:
                    messages.append(last_assistant_message)
                messages.append({"role": "user", "content": user_message})
                await self.push_frame(LLMMessagesFrame(messages))


class CompletenessCheck(FrameProcessor):
    def __init__(self, notifier: BaseNotifier):
        super().__init__()
        self._notifier = notifier

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame) and frame.text == "YES":
            logger.debug("!!! Completeness check YES")
            await self.push_frame(UserStoppedSpeakingFrame())
            await self._notifier.notify()
        elif isinstance(frame, TextFrame) and frame.text == "NO":
            logger.debug("!!! Completeness check NO")


class OutputGate(FrameProcessor):
    def __init__(self, notifier: BaseNotifier, **kwargs):
        super().__init__(**kwargs)
        self._gate_open = False
        self._frames_buffer = []
        self._notifier = notifier

    def close_gate(self):
        self._gate_open = False

    def open_gate(self):
        self._gate_open = True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # We must not block system frames.
        if isinstance(frame, SystemFrame):
            if isinstance(frame, StartFrame):
                await self._start()
            if isinstance(frame, (EndFrame, CancelFrame)):
                await self._stop()
            if isinstance(frame, StartInterruptionFrame):
                self._frames_buffer = []
                self.close_gate()
            await self.push_frame(frame, direction)
            return

        # Ignore frames that are not following the direction of this gate.
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        if self._gate_open:
            await self.push_frame(frame, direction)
            return

        self._frames_buffer.append((frame, direction))

    async def _start(self):
        self._frames_buffer = []
        self._gate_task = self.get_event_loop().create_task(self._gate_task_handler())

    async def _stop(self):
        self._gate_task.cancel()
        await self._gate_task

    async def _gate_task_handler(self):
        while True:
            try:
                await self._notifier.wait()
                self.open_gate()
                for frame, direction in self._frames_buffer:
                    await self.push_frame(frame, direction)
                self._frames_buffer = []
            except asyncio.CancelledError:
                break


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, _) = await configure(session)

        transport = DailyTransport(
            room_url,
            None,
            "Respond bot",
            DailyParams(
                audio_out_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                vad_audio_passthrough=True,
            ),
        )

        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
        )

        # This is the LLM that will be used to detect if the user has finished a
        # statement. This doesn't really need to be an LLM, we could use NLP
        # libraries for that, but we have the machinery to use an LLM, so we might as well!
        statement_llm = AnthropicLLMService(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-3-5-sonnet-20241022",
        )

        # This is the regular LLM.
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
        )

        messages = [
            {
                "role": "system",
                "content": conversational_system_message,
            },
        ]

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        # We have instructed the LLM to return 'YES' if it thinks the user
        # completed a sentence. So, if it's 'YES' we will return true in this
        # predicate which will wake up the notifier.
        async def wake_check_filter(frame):
            return frame.text == "YES"

        # This is a notifier that we use to synchronize the two LLMs.
        notifier = EventNotifier()

        # This turns the LLM context into an inference request to classify the user's speech
        # as complete or incomplete.
        statement_judge_context_filter = StatementJudgeContextFilter(notifier=notifier)

        # This sends a UserStoppedSpeakingFrame and triggers the notifier event
        completeness_check = CompletenessCheck(notifier=notifier)

        # # Notify if the user hasn't said anything.
        async def user_idle_notifier(frame):
            await notifier.notify()

        # Sometimes the LLM will fail detecting if a user has completed a
        # sentence, this will wake up the notifier if that happens.
        user_idle = UserIdleProcessor(callback=user_idle_notifier, timeout=5.0)

        bot_output_gate = OutputGate(notifier=notifier)

        async def block_user_stopped_speaking(frame):
            return not isinstance(frame, UserStoppedSpeakingFrame)

        async def pass_only_llm_trigger_frames(frame):
            return (
                isinstance(frame, OpenAILLMContextFrame)
                or isinstance(frame, LLMMessagesFrame)
                or isinstance(frame, StartInterruptionFrame)
                or isinstance(frame, StopInterruptionFrame)
            )

        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                context_aggregator.user(),
                ParallelPipeline(
                    [
                        # Pass everything except UserStoppedSpeaking to the elements after
                        # this ParallelPipeline
                        FunctionFilter(filter=block_user_stopped_speaking),
                    ],
                    [
                        # Ignore everything except an OpenAILLMContextFrame. Pass a specially constructed
                        # LLMMessagesFrame to the statement classifier LLM. The only frame this
                        # sub-pipeline will output is a UserStoppedSpeakingFrame.
                        statement_judge_context_filter,
                        statement_llm,
                        completeness_check,
                    ],
                    [
                        # Block everything except OpenAILLMContextFrame and LLMMessagesFrame
                        FunctionFilter(filter=pass_only_llm_trigger_frames),
                        llm,
                        bot_output_gate,  # Buffer all llm/tts output until notified.
                    ],
                ),
                tts,
                user_idle,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(
            pipeline,
            PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            # Kick off the conversation.
            messages.append(
                {
                    "role": "user",
                    "content": "Start by just saying \"Hello I'm ready.\" Don't say anything else.",
                }
            )
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        @transport.event_handler("on_app_message")
        async def on_app_message(transport, message, sender):
            logger.debug(f"Received app message: {message} - {sender}")
            if "message" not in message:
                return

            await task.queue_frames(
                [
                    UserStartedSpeakingFrame(),
                    TranscriptionFrame(
                        user_id=sender, timestamp=time.time(), text=message["message"]
                    ),
                    UserStoppedSpeakingFrame(),
                ]
            )

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
