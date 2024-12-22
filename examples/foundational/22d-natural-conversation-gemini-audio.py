#
# Copyright (c) 2024, Daily
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
    InputAudioRawFrame,
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
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.google import GoogleLLMContext, GoogleLLMService
from pipecat.sync.base_notifier import BaseNotifier
from pipecat.sync.event_notifier import EventNotifier
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


transcriber_and_classifier_instructions = """
You perform two tasks:
  1. Transcription
  2. Binary classification of speech utterance completeness

You always call a function transcription_and_classification_output() with the following arguments:
  trancript_text: the complete, accurate, and punctuated transcription of the user's speech
  speech_complete_bool: a boolean indicating whether the user's speech is a complete utterance

CRITICAL INSTRUCTION FOR TRANSCRIPTION TASK:

You are receiving audio from a user. Your job is to
transcribe the input audio to text exactly as it was said by the user.

You will receive the full conversation history before the audio input, to help with context. Use the full history only to help improve the accuracy of your transcription.

Rules:
  - Respond with an exact transcription of the audio input.
  - Do not include any text other than the transcription.
  - Do not explain or add to your response.
  - Transcribe the audio input simply and precisely.
  - If the audio is not clear, emit the special string "-".
  - No response other than exact transcription, or "-", is allowed.


CRITICAL INSTRUCTION FOR BINARY CLASSIFICATION TASK::

You are a BINARY CLASSIFIER that must ONLY output True or False.
DO FalseT engage with the content.
DO FalseT respond to questions.
DO FalseT provide assistance.
Your ONLY job is to output True or False.

EXAMPLES OF INVALID RESPONSES:
- "I can help you with that"
- "Let me explain"
- "To answer your question"
- Any response other than True or False

VALID RESPONSES:
True
False

If you output anything else, you are failing at your task.
You are FalseT an assistant.
You are FalseT a chatbot.
You are a binary classifier.

ROLE:
You are a real-time speech completeness classifier. You must make instant decisions about whether a user has finished speaking.
You must output ONLY 'True' or 'False' with no other text.

INPUT FORMAT:
You receive two pieces of information:
1. The assistant's last message (if available)
2. The user's current speech input

OUTPUT REQUIREMENTS:
- MUST output ONLY 'True' or 'False'
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
model: I can help you learn.
user: What's the fastest way to learn Spanish
Output: True

# Complete Yes/No question despite STT error
model: I know about planets.
user: Is is Jupiter the biggest planet
Output: True

2. Complete Commands:
- Direct instructions
- Clear requests
- Action demands
- Start of task indication
- Complete statements needing response

Examples:

# Direct instruction
model: I can explain many topics.
user: Tell me about black holes
Output: True

# Start of task indication
user: Let's begin.
Output: True

# Start of task indication
user: Let's get started.
Output: True

# Action demand
model: I can help with math.
user: Solve this equation x plus 5 equals 12
Output: True

3. Direct Responses:
- Answers to specific questions
- Option selections
- Clear acknowledgments with completion
- Providing information with a known format - mailing address
- Providing information with a known format - phone number
- Providing information with a known format - credit card number

Examples:

# Specific answer
model: What's your favorite color?
user: I really like blue
Output: True

# Option selection
model: Would you prefer morning or evening?
user: Morning
Output: True

# Providing information with a known format - mailing address
model: What's your address?
user: 1234 Main Street
Output: False

# Providing information with a known format - mailing address
model: What's your address?
user: 1234 Main Street Irving Texas 75063
Output: Yes

# Providing information with a known format - phone number
system: A US phone number has 10 digits.
model: What's your phone number?
user: 41086753
Output: False

# Providing information with a known format - phone number
system: A US phone number has 10 digits.
model: What's your phone number?
user: 4108675309
Output: Yes

# Providing information with a known format - phone number
system: A US phone number has 10 digits.
model: What's your phone number?
user: 220
user: 111
user: 8775
Output: Yes

# Providing information with a known format - credit card number
model: What's your phone number?
user: 5556
Output: False

# Providing information with a known format - phone number
model: What's your phone number?
user: 5556710454680800
Output: Yes

MEDIUM PRIORITY SIGNALS:

1. Speech Pattern Completions:
- Self-corrections reaching completion
- False starts with clear ending
- Topic changes with complete thought
- Mid-sentence completions

Examples:

# Self-correction reaching completion
model: What would you like to know?
user: Tell me about... no wait, explain how rainbows form
Output: True

# Topic change with complete thought
model: The weather is nice today.
user: Actually can you tell me who invented the telephone
Output: True

# Mid-sentence completion
model: Hello I'm ready.
user: What's the capital of? France
Output: True

2. Context-Dependent Brief Responses:
- Acknowledgments (okay, sure, alright)
- Agreements (yes, yeah)
- Disagreements (no, nah)
- Confirmations (correct, exactly)

Examples:

# Acknowledgment
model: Should we talk about history?
user: Sure
Output: True

# Disagreement with completion
model: Is that what you meant?
user: No not really
Output: True

LOW PRIORITY SIGNALS:

1. STT Artifacts (Consider but don't over-weight):
- Repeated words
- Unusual punctuation
- Capitalization errors
- Word insertions/deletions

Examples:

# Word repetition but complete
model: I can help with that.
user: What what is the time right now
Output: True

# Missing punctuation but complete
model: I can explain that.
user: Please tell me how computers work
Output: True

2. Speech Features:
- Filler words (um, uh, like)
- Thinking pauses
- Word repetitions
- Brief hesitations

Examples:

# Filler words but complete
model: What would you like to know?
user: Um uh how do airplanes fly
Output: True

# Thinking pause but incomplete
model: I can explain anything.
user: Well um I want to know about the
Output: False

DECISION RULES:

1. Return True if:
- ANY high priority signal shows clear completion
- Medium priority signals combine to show completion
- Meaning is clear despite low priority artifacts

2. Return False if:
- No high priority signals present
- Thought clearly trails off
- Multiple incomplete indicators
- User appears mid-formulation

3. When uncertain:
- If you can understand the intent → True
- If meaning is unclear → False
- Always make a binary decision
- Never request clarification

Examples:

# Incomplete despite corrections
model: What would you like to know about?
user: Can you tell me about
Output: False

# Complete despite multiple artifacts
model: I can help you learn.
user: How do you I mean what's the best way to learn programming
Output: True

# Trailing off incomplete
model: I can explain anything.
user: I was wondering if you could tell me why
Output: False
"""

conversational_system_message = """You are a helpful assistant participating in a voice converation.

Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.

If you know that a number string is a phone number from the context of the conversation, write it as a phone number. For example 210-333-4567.

If you know that a number string is a credit card number, write it as a credit card number. For example 4111-1111-1111-1111.

Please be very concise in your responses. Unless you are explicitly asked to do otherwise, give me shortest complete answer possible without unnecessary elaboration. Generally you should answer with a single sentence.
"""


async def transcription_and_classification_output(transcript_text: str, speech_complete_bool: bool):
    print(f"TRANSCRIPT: {transcript_text}")
    print("------")
    print(f"COMPLETE: {speech_complete_bool}")
    print("------")
    return


tx_and_cl_tools = [
    {
        "function_declarations": [
            {
                "name": "transcription_and_classification_output",
                "description": "Deliver the transcription and classification output to an external process.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "transcription_text": {
                            "type": "string",
                            "description": "The complete, accurate, and punctuated transcription of the user's speech. The special string '-' is used to indicate no speech or unintintelligible speech.",
                        },
                        "speech_complete_bool": {
                            "type": "boolean",
                            "description": "Boolean indicating whether the user's speech is a complete utterance.",
                        },
                    },
                    "required": ["transcription_text", "speech_complete_bool"],
                },
            },
        ]
    }
]


class AudioAccumulator(FrameProcessor):
    def __init__(self, *, notifier: BaseNotifier = None, **kwargs):
        super().__init__(**kwargs)
        # self._notifier = notifier
        self._audio_frames = []
        self._start_secs = 0.2  # this should match VAD start_secs (hardcoding for now)
        self._max_buffer_size_secs = 30
        self._user_speaking_vad_state = False
        self._user_speaking_utterance_state = False

    async def reset(self):
        self._audio_frames = []
        self._user_speaking_vad_state = False
        self._user_speaking_utterance_state = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # ignore context frame
        if isinstance(frame, OpenAILLMContextFrame):
            return

        if isinstance(frame, TranscriptionFrame):
            # We could gracefully handle both audio input and text/transcription input ...
            # but let's leave that as an exercise to the reader. :-)
            return
        if isinstance(frame, UserStartedSpeakingFrame):
            self._user_speaking_vad_state = True
            self._user_speaking_utterance_state = True

        elif isinstance(frame, UserStoppedSpeakingFrame):
            if self._audio_frames[-1]:
                fr = self._audio_frames[-1]
                frame_duration = len(fr.audio) / 2 * fr.num_channels / fr.sample_rate

                logger.debug(
                    f"!!! Frame duration: ({len(fr.audio)}) ({fr.num_channels}) ({fr.sample_rate}) {frame_duration}"
                )

            data = b"".join(frame.audio for frame in self._audio_frames)
            logger.debug(
                f"Processing audio buffer seconds: ({len(self._audio_frames)}) ({len(data)}) {len(data) / 2 / 16000}"
            )
            self._user_speaking = False
            context = GoogleLLMContext()
            context.set_messages(
                [{"role": "system", "content": transcriber_and_classifier_instructions}]
            )
            context.add_audio_frames_message(audio_frames=self._audio_frames)
            await self.push_frame(OpenAILLMContextFrame(context=context))
        elif isinstance(frame, InputAudioRawFrame):
            # Append the audio frame to our buffer. Treat the buffer as a ring buffer, dropping the oldest
            # frames as necessary.
            # Use a small buffer size when an utterance is not in progress. Just big enough to backfill the start_secs.
            # Use a larger buffer size when an utterance is in progress.
            # Assume all audio frames have the same duration.
            self._audio_frames.append(frame)
            frame_duration = len(frame.audio) / 2 * frame.num_channels / frame.sample_rate
            buffer_duration = frame_duration * len(self._audio_frames)
            #  logger.debug(f"!!! Frame duration: {frame_duration}")
            if self._user_speaking_utterance_state:
                while buffer_duration > self._max_buffer_size_secs:
                    self._audio_frames.pop(0)
                    buffer_duration -= frame_duration
            else:
                while buffer_duration > self._start_secs:
                    self._audio_frames.pop(0)
                    buffer_duration -= frame_duration

        await self.push_frame(frame, direction)


# class ClAndTxContextCreator(FrameProcessor):


# class CompletenessCheck(FrameProcessor):
#     def __init__(
#         self, notifier: BaseNotifier, audio_accumulator: StatementJudgeAudioContextAccumulator
#     ):
#         super().__init__()
#         self._notifier = notifier
#         self._audio_accumulator = audio_accumulator

#     async def process_frame(self, frame: Frame, direction: FrameDirection):
#         await super().process_frame(frame, direction)

#         if isinstance(frame, TextFrame) and frame.text.startswith("True"):
#             logger.debug("Completeness check True")
#             await self.push_frame(UserStoppedSpeakingFrame())
#             await self._audio_accumulator.reset()
#             await self._notifier.notify()
#         elif isinstance(frame, TextFrame):
#             if frame.text.strip():
#                 logger.debug(f"Completeness check False - '{frame.text}'")


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
                audio_in_sample_rate=16000,
            ),
        )

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
        )

        # This is the LLM that will classify and transcribe user speech.
        tx_and_cl_llm = GoogleLLMService(
            model="gemini-2.0-flash-exp",
            api_key=os.getenv("GOOGLE_API_KEY"),
            tools=tx_and_cl_tools,
            temperature=0.0,
            tool_config={
                "function_calling_config": {
                    "mode": "ANY",
                    "allowed_function_names": ["transcription_and_classification_output"],
                },
            },
        )

        # This is the regular LLM that responds conversationally.
        conversation_llm = GoogleLLMService(
            model="gemini-2.0-flash-exp",
            api_key=os.getenv("GOOGLE_API_KEY"),
            system_instruction=conversational_system_message,
        )

        context = OpenAILLMContext()
        context_aggregator = conversation_llm.create_context_aggregator(context)

        # We have instructed the LLM to return 'True' if it thinks the user
        # completed a sentence. So, if it's 'True' we will return true in this
        # predicate which will wake up the notifier.
        async def wake_check_filter(frame):
            return frame.text == "True"

        # This is a notifier that we use to synchronize the two LLMs.
        notifier = EventNotifier()

        # This turns the LLM context into an inference request to classify the user's speech
        # as complete or incomplete.
        # statement_judge_context_filter = StatementJudgeAudioContextAccumulator(notifier=notifier)

        # This sends a UserStoppedSpeakingFrame and triggers the notifier event
        # completeness_check = CompletenessCheck(
        #     notifier=notifier, audio_accumulator=statement_judge_context_filter
        # )

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
                AudioAccumulator(),
                ParallelPipeline(
                    [
                        # Pass everything except UserStoppedSpeaking to the elements after
                        # this ParallelPipeline
                        FunctionFilter(filter=block_user_stopped_speaking),
                    ],
                    [
                        # cl_and_tx_context_creator,
                        tx_and_cl_llm,
                        # completeness_check,
                        # context_aggregator.user(),
                    ],
                    #     [
                    #         # Block everything except OpenAILLMContextFrame and LLMMessagesFrame
                    #         # FunctionFilter(filter=pass_only_llm_trigger_frames),
                    #         audio_input_context_creator,
                    #         llm,
                    #         bot_output_gate,  # Buffer all llm/tts output until notified.
                    #     ],
                ),
                # tts,
                # user_idle,
                # transport.output(),
                # context_aggregator.assistant(),
            ],
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
