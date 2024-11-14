#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import aiohttp
import asyncio
import os
import sys
import time

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
)
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.anthropic import AnthropicLLMService
from pipecat.sync.event_notifier import EventNotifier
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    StartFrame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    SystemFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame
from pipecat.sync.base_notifier import BaseNotifier
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.processors.user_idle_processor import UserIdleProcessor


from runner import configure

from loguru import logger

from dotenv import load_dotenv

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


classifier_statement = """Determine if the user's statement ends with a complete thought and you should respond.

The user text is transcribed speech. You are trying to determine if:

1. the user has finished talking and expects a response from you, or
2. this statement is incomplete and the user will continue talking

A previous assistant response is provided for additional context. But you are only evaluating the user text. 

The user text may contain multiple fragments concatentated together. There may be repeated words or mistakes in the transcription. There may be grammatical errors. There may be extra punctuation. Ignore all of that. Interpret the transcribed text as text that would have been spoken. Then consider only whether the user has finished speaking and is expecting a response.

Categorize the last user statement as either complete with the user now expecting a response, or incomplete.

Return 'YES' if text is likely complete and the user is expecting a response. Return 'NO' if the text seems to be a partial expression or unfinished thought.

If you are not sure, respond with your best guess. If the user is expecting a response, respond with YES. If the user is not expecting a response, respond with NO. Always output either YES or NO and no other text.

Respond only YES or NO

Examples:

User: What's the capital of
Assistant: NO

User: What's the captial of France?
Assistant: YES

User: Tell me a story about
Assistant: NO

User: Tell me a story about a dragon
Assistant YES

User: Is there a
Assistant: NO

User: Is there a large
Assistant: NO

User: Is there a large lake near Chicago?
Assistant: YES

User: When is the longest day of the year?
Assistant: YES

User: When when is the longest day of the year
Assistant: YES

User: When when is the
ASSISTANT: NO

User: What is the um I u
Assistant: NO

User: What is the um i u largest city in the world
Assistant: YES

User: How much does a how much does an adult elephant weigh?
Assistant: YES

User: How much does a how much does
Assistant: NO

User: What can you tell me All the
Assistant: NO

User: What can you tell me All the prime numbers less than 100
Assistant: YES

User: What's the what's the length of the Amazon River?
Assistant: YES

User: What's what's the length of the Amazon River?
Assistant: YES

User: What's what's the length of the Amazon River
Assistant: YES

User: What's what's the best way to get a coffee stain out of a white shirt
Assistant: YES
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
            api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-3-5-haiku-20241022", name="Haiku"
        )

        # This is the regular LLM.
        llm = AnthropicLLMService(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-3-5-sonnet-20241022",
            name="Sonnet",
            params=AnthropicLLMService.InputParams(enable_prompt_caching_beta=True),
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
            await task.queue_frames([LLMMessagesFrame(messages)])

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
