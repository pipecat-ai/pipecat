#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMRunFrame,
    StartFrame,
    SystemFrame,
    TextFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.user_idle_processor import UserIdleProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.sync.base_notifier import BaseNotifier
from pipecat.sync.event_notifier import EventNotifier
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.utils.time import time_now_iso8601

load_dotenv(override=True)


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
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        # We must not block system frames.
        if isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
            return

        # We only want to handle LLMContextFrames, and only want to push through a simplified
        # context frame that contains a system prompt and the most recent user messages,
        if isinstance(frame, LLMContextFrame):
            # Take text content from the most recent user messages.
            messages = frame.context.get_messages()
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
            # If we have any user text content, push a context frame with the simplified context.
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
                await self.push_frame(LLMContextFrame(LLMContext(messages)))


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
        else:
            await self.push_frame(frame, direction)


class OutputGate(FrameProcessor):
    def __init__(self, *, notifier: BaseNotifier, start_open: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._gate_open = start_open
        self._frames_buffer = []
        self._notifier = notifier
        self._gate_task = None

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
            if isinstance(frame, InterruptionFrame):
                self._frames_buffer = []
                self.close_gate()
            await self.push_frame(frame, direction)
            return

        # Don't block function call frames
        if isinstance(frame, (FunctionCallInProgressFrame, FunctionCallResultFrame)):
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
        if not self._gate_task:
            self._gate_task = self.create_task(self._gate_task_handler())

    async def _stop(self):
        if self._gate_task:
            await self.cancel_task(self._gate_task)
            self._gate_task = None

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


class TurnDetectionLLM(Pipeline):
    def __init__(self, llm: LLMService):
        # This is the LLM that will be used to detect if the user has finished a
        # statement. This doesn't really need to be an LLM, we could use NLP
        # libraries for that, but we have the machinery to use an LLM, so we might as well!
        statement_llm = AnthropicLLMService(api_key=os.getenv("ANTHROPIC_API_KEY"))

        # This is a notifier that we use to synchronize the two LLMs.
        notifier = EventNotifier()

        # This turns the LLM context into an inference request to classify the user's speech
        # as complete or incomplete.
        statement_judge_context_filter = StatementJudgeContextFilter()

        # This sends a UserStoppedSpeakingFrame and triggers the notifier event
        completeness_check = CompletenessCheck(notifier=notifier)

        # # Notify if the user hasn't said anything.
        async def user_idle_notifier(frame):
            await notifier.notify()

        # Sometimes the LLM will fail detecting if a user has completed a
        # sentence, this will wake up the notifier if that happens.
        user_idle = UserIdleProcessor(callback=user_idle_notifier, timeout=5.0)

        # We start with the gate open because we send an initial context frame
        # to start the conversation.
        bot_output_gate = OutputGate(notifier=notifier, start_open=True)

        async def block_user_stopped_speaking(frame):
            return not isinstance(frame, UserStoppedSpeakingFrame)

        async def pass_only_llm_trigger_frames(frame):
            return (
                isinstance(frame, LLMContextFrame)
                or isinstance(frame, InterruptionFrame)
                or isinstance(frame, FunctionCallInProgressFrame)
                or isinstance(frame, FunctionCallResultFrame)
            )

        async def filter_all(frame):
            return False

        super().__init__(
            [
                ParallelPipeline(
                    [
                        # Pass everything except UserStoppedSpeaking to the elements after
                        # this ParallelPipeline
                        FunctionFilter(filter=block_user_stopped_speaking),
                    ],
                    [
                        # Ignore everything except an LLMContextFrame. Pass a specially constructed
                        # simplified context frame to the statement classifier LLM. The only frame this
                        # sub-pipeline will output is a UserStoppedSpeakingFrame.
                        statement_judge_context_filter,
                        statement_llm,
                        completeness_check,
                        FunctionFilter(filter=filter_all, direction=FrameDirection.UPSTREAM),
                    ],
                    [
                        # Block everything except frames that trigger LLM inference.
                        FunctionFilter(filter=pass_only_llm_trigger_frames),
                        llm,
                        bot_output_gate,  # Buffer all llm/tts output until notified.
                    ],
                ),
                user_idle,
            ]
        )


async def fetch_weather_from_api(params: FunctionCallParams):
    await params.result_callback({"conditions": "nice", "temperature": "75"})


# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    # This is the regular LLM.
    llm_main = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))
    # Register a function_name of None to get all functions
    # sent to the same callback with an additional function_name parameter.
    llm_main.register_function("get_current_weather", fetch_weather_from_api)

    @llm_main.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls):
        await tts.queue_frame(TTSSpeakFrame("Let me check on that."))

    weather_function = FunctionSchema(
        name="get_current_weather",
        description="Get the current weather",
        properties={
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "format": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The temperature unit to use. Infer this from the users location.",
            },
        },
        required=["location", "format"],
    )
    tools = ToolsSchema(standard_tools=[weather_function])

    messages = [
        {
            "role": "system",
            "content": conversational_system_message,
        },
    ]

    context = LLMContext(messages, tools)
    context_aggregator = LLMContextAggregatorPair(context)

    # LLM + turn detection (with an extra LLM as a judge)
    llm = TurnDetectionLLM(llm_main)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        messages.append(
            {
                "role": "user",
                "content": "Start by just saying \"Hello I'm ready.\" Don't say anything else.",
            }
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_app_message")
    async def on_app_message(transport, message, sender):
        logger.debug(f"Received app message: {message}")
        if "message" not in message:
            return

        await task.queue_frames(
            [
                UserStartedSpeakingFrame(),
                TranscriptionFrame(
                    user_id="", timestamp=time_now_iso8601(), text=message["message"]
                ),
                UserStoppedSpeakingFrame(),
            ]
        )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
