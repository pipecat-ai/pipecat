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
import google.ai.generativelanguage as glm
from dotenv import load_dotenv
from loguru import logger
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
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
from pipecat.processors.aggregators.llm_response import LLMResponseAggregator
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

# TRANSCRIBER_MODEL = "gemini-1.5-flash-latest"
# CLASSIFIER_MODEL = "gemini-1.5-flash-latest"
# CONVERSATION_MODEL = "gemini-1.5-flash-latest"

TRANSCRIBER_MODEL = "gemini-2.0-flash-exp"
CLASSIFIER_MODEL = "gemini-2.0-flash-exp"
CONVERSATION_MODEL = "gemini-2.0-flash-exp"

transcriber_system_instruction = """You are an audio transcriber. You are receiving audio from a user. Your job is to
transcribe the input audio to text exactly as it was said by the user.

You will receive the full conversation history before the audio input, to help with context. Use the full history only to help improve the accuracy of your transcription.

Rules:
  - Respond with an exact transcription of the audio input.
  - Do not include any text other than the transcription.
  - Do not explain or add to your response.
  - Transcribe the audio input simply and precisely.
  - If the audio is not clear, emit the special string "-".
  - No response other than exact transcription, or "-", is allowed.

"""

classifier_system_instruction = """CRITICAL INSTRUCTION:
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
model: I can help you learn.
user: What's the fastest way to learn Spanish
Output: YES

# Complete Yes/No question despite STT error
model: I know about planets.
user: Is is Jupiter the biggest planet
Output: YES

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
Output: YES

# Start of task indication
user: Let's begin.
Output: YES

# Start of task indication
user: Let's get started.
Output: YES

# Action demand
model: I can help with math.
user: Solve this equation x plus 5 equals 12
Output: YES

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
Output: YES

# Option selection
model: Would you prefer morning or evening?
user: Morning
Output: YES

# Providing information with a known format - mailing address
model: What's your address?
user: 1234 Main Street
Output: NO

# Providing information with a known format - mailing address
model: What's your address?
user: 1234 Main Street Irving Texas 75063
Output: Yes

# Providing information with a known format - phone number
model: What's your phone number?
user: 41086753
Output: NO

# Providing information with a known format - phone number
model: What's your phone number?
user: 4108675309
Output: Yes

# Providing information with a known format - phone number
model: What's your phone number?
user: 220
Output: No

# Providing information with a known format - credit card number
model: What's your credit card number?
user: 5556
Output: NO

# Providing information with a known format - phone number
model: What's your credit card number?
user: 5556710454680800
Output: Yes

model: What's your credit card number?
user: 414067
Output: NO


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
Output: YES

# Topic change with complete thought
model: The weather is nice today.
user: Actually can you tell me who invented the telephone
Output: YES

# Mid-sentence completion
model: Hello I'm ready.
user: What's the capital of? France
Output: YES

2. Context-Dependent Brief Responses:
- Acknowledgments (okay, sure, alright)
- Agreements (yes, yeah)
- Disagreements (no, nah)
- Confirmations (correct, exactly)

Examples:

# Acknowledgment
model: Should we talk about history?
user: Sure
Output: YES

# Disagreement with completion
model: Is that what you meant?
user: No not really
Output: YES

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
Output: YES

# Missing punctuation but complete
model: I can explain that.
user: Please tell me how computers work
Output: YES

2. Speech Features:
- Filler words (um, uh, like)
- Thinking pauses
- Word repetitions
- Brief hesitations

Examples:

# Filler words but complete
model: What would you like to know?
user: Um uh how do airplanes fly
Output: YES

# Thinking pause but incomplete
model: I can explain anything.
user: Well um I want to know about the
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
model: What would you like to know about?
user: Can you tell me about
Output: NO

# Complete despite multiple artifacts
model: I can help you learn.
user: How do you I mean what's the best way to learn programming
Output: YES

# Trailing off incomplete
model: I can explain anything.
user: I was wondering if you could tell me why
Output: NO
"""

conversation_system_instruction = """You are a helpful assistant participating in a voice converation.

Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.

If you know that a number string is a phone number from the context of the conversation, write it as a phone number. For example 210-333-4567.

If you know that a number string is a credit card number, write it as a credit card number. For example 4111-1111-1111-1111.

Please be very concise in your responses. Unless you are explicitly asked to do otherwise, give me shortest complete answer possible without unnecessary elaboration. Generally you should answer with a single sentence.
"""


class AudioAccumulator(FrameProcessor):
    """Buffers user audio until the user stops speaking.

    Always pushes a fresh context with a single audio message.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
            data = b"".join(frame.audio for frame in self._audio_frames)
            logger.debug(
                f"Processing audio buffer seconds: ({len(self._audio_frames)}) ({len(data)}) {len(data) / 2 / 16000}"
            )
            self._user_speaking = False
            context = GoogleLLMContext()
            context.add_audio_frames_message(text="Audio follows", audio_frames=self._audio_frames)
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


class CompletenessCheck(FrameProcessor):
    """Checks the result of the classifier LLM to determine if the user has finished speaking.

    Triggers the notifier if the user has finished speaking. Also triggers the notifier if an
    idle timeout is reached.
    """

    wait_time = 5.0

    def __init__(self, notifier: BaseNotifier, audio_accumulator: AudioAccumulator, **kwargs):
        super().__init__()
        self._notifier = notifier
        self._audio_accumulator = audio_accumulator
        self._idle_task = None
        self._wakeup_time = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            if self._idle_task:
                self._idle_task.cancel()
        elif isinstance(frame, TextFrame) and frame.text.startswith("YES"):
            logger.debug("Completeness check YES")
            if self._idle_task:
                self._idle_task.cancel()
            await self.push_frame(UserStoppedSpeakingFrame())
            await self._audio_accumulator.reset()
            await self._notifier.notify()
        elif isinstance(frame, TextFrame):
            if frame.text.strip():
                logger.debug(f"Completeness check NO - '{frame.text}'")
                # start timer to wake up if necessary
                if self._wakeup_time:
                    self._wakeup_time = time.time() + self.wait_time
                else:
                    # logger.debug("!!! CompletenessCheck idle wait START")
                    self._wakeup_time = time.time() + self.wait_time
                    self._idle_task = self.get_event_loop().create_task(self._idle_task_handler())

    async def _idle_task_handler(self):
        try:
            while time.time() < self._wakeup_time:
                await asyncio.sleep(0.01)
            # logger.debug(f"!!! CompletenessCheck idle wait OVER")
            await self._audio_accumulator.reset()
            await self._notifier.notify()
        except asyncio.CancelledError:
            # logger.debug(f"!!! CompletenessCheck idle wait CANCEL")
            pass
        except Exception as e:
            logger.error(f"CompletenessCheck idle wait error: {e}")
            raise e
        finally:
            # logger.debug(f"!!! CompletenessCheck idle wait FINALLY")
            self._wakeup_time = 0
            self._idle_task = None


class UserAggregatorBuffer(LLMResponseAggregator):
    """Buffers the output of the transcription LLM. Used by the bot output gate."""

    def __init__(self, **kwargs):
        super().__init__(
            messages=None,
            role=None,
            start_frame=LLMFullResponseStartFrame,
            end_frame=LLMFullResponseEndFrame,
            accumulator_frame=TextFrame,
            handle_interruptions=True,
            expect_stripped_words=False,
        )
        self._transcription = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        # parent method pushes frames
        if isinstance(frame, UserStartedSpeakingFrame):
            self._transcription = ""

    async def _push_aggregation(self):
        if self._aggregation:
            self._transcription = self._aggregation
            self._aggregation = ""

            logger.debug(f"[Transcription] {self._transcription}")

    async def wait_for_transcription(self):
        while not self._transcription:
            await asyncio.sleep(0.01)
        tx = self._transcription
        self._transcription = ""
        return tx


class ConversationAudioContextAssembler(FrameProcessor):
    """Takes the single-message context generated by the AudioAccumulator and adds it to the conversation LLM's context."""

    def __init__(self, context: OpenAILLMContext, **kwargs):
        super().__init__(**kwargs)
        self._context = context

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # We must not block system frames.
        if isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, OpenAILLMContextFrame):
            GoogleLLMContext.upgrade_to_google(self._context)
            last_message = frame.context.messages[-1]
            self._context._messages.append(last_message)
            await self.push_frame(OpenAILLMContextFrame(context=self._context))


class OutputGate(FrameProcessor):
    """Buffers output frames until the notifier is triggered.

    When the notifier fires, waits until a transcription is ready, then:
      1. Replaces the last user audio message with the transcription.
      2. Flushes the frames buffer.
    """

    def __init__(
        self,
        notifier: BaseNotifier,
        context: OpenAILLMContext,
        user_transcription_buffer: "UserAggregatorBuffer",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._gate_open = False
        self._frames_buffer = []
        self._notifier = notifier
        self._context = context
        self._transcription_buffer = user_transcription_buffer

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

        if isinstance(frame, LLMFullResponseStartFrame):
            # Remove the audio message from the context. We will never need it again.
            # If the completeness check fails, a new audio message will be appended to the context.
            # If the completeness check succeeds, our notifier will fire and we will append the
            # transcription to the context.
            self._context._messages.pop()

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

                transcription = await self._transcription_buffer.wait_for_transcription() or "-"
                self._context._messages.append(
                    glm.Content(role="user", parts=[glm.Part(text=transcription)])
                )

                self.open_gate()
                for frame, direction in self._frames_buffer:
                    await self.push_frame(frame, direction)
                self._frames_buffer = []
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"OutputGate error: {e}")
                raise e
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

        # This is the LLM that will transcribe user speech.
        tx_llm = GoogleLLMService(
            name="Transcriber",
            model=TRANSCRIBER_MODEL,
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.0,
            system_instruction=transcriber_system_instruction,
        )

        # This is the LLM that will classify user speech as complete or incomplete.
        classifier_llm = GoogleLLMService(
            name="Classifier",
            model=CLASSIFIER_MODEL,
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.0,
            system_instruction=classifier_system_instruction,
        )

        # This is the regular LLM that responds conversationally.
        conversation_llm = GoogleLLMService(
            name="Conversation",
            model=CONVERSATION_MODEL,
            api_key=os.getenv("GOOGLE_API_KEY"),
            system_instruction=conversation_system_instruction,
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

        audio_accumulater = AudioAccumulator()
        # This sends a UserStoppedSpeakingFrame and triggers the notifier event
        completeness_check = CompletenessCheck(
            notifier=notifier, audio_accumulator=audio_accumulater
        )

        async def block_user_stopped_speaking(frame):
            return not isinstance(frame, UserStoppedSpeakingFrame)

        async def pass_only_llm_trigger_frames(frame):
            return (
                isinstance(frame, OpenAILLMContextFrame)
                or isinstance(frame, LLMMessagesFrame)
                or isinstance(frame, StartInterruptionFrame)
                or isinstance(frame, StopInterruptionFrame)
            )

        conversation_audio_context_assembler = ConversationAudioContextAssembler(context=context)

        user_aggregator_buffer = UserAggregatorBuffer()

        bot_output_gate = OutputGate(
            notifier=notifier, context=context, user_transcription_buffer=user_aggregator_buffer
        )

        pipeline = Pipeline(
            [
                transport.input(),
                audio_accumulater,
                ParallelPipeline(
                    [
                        # Pass everything except UserStoppedSpeaking to the elements after
                        # this ParallelPipeline
                        FunctionFilter(filter=block_user_stopped_speaking),
                    ],
                    [
                        ParallelPipeline(
                            [
                                classifier_llm,
                                completeness_check,
                            ],
                            [
                                tx_llm,
                                user_aggregator_buffer,
                            ],
                        )
                    ],
                    [
                        conversation_audio_context_assembler,
                        conversation_llm,
                        bot_output_gate,  # buffer output until notified, then flush frames and update context
                        # TempPrinter(),
                    ],
                ),
                tts,
                transport.output(),
                context_aggregator.assistant(),
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
