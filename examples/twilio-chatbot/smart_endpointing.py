import asyncio
import time

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMMessagesFrame,
    StartFrame,
    StartInterruptionFrame,
    SystemFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    InputAudioRawFrame
)

from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.google.llm import GoogleLLMContext
from pipecat.sync.base_notifier import BaseNotifier
from google.genai.types import Content, Part

load_dotenv(override=True)


CLASSIFIER_MODEL = "gemini-2.0-flash-001"

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
You receive a list of dictionaries containing role and content information.
The list ALWAYS contains at least one dictionary with the role "user". There may be an "assistant" element providing context.
Do not consider the assistant's content when determining if the user's final utterance is complete; only use the most recent user input.

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

2. Complete Commands:
   - Direct instructions, clear requests, or action demands that form a complete statement

3. Direct Responses/Statements:
   - Answers to specific questions
   - Option selections
   - Clear acknowledgments or complete statements (even if expressing uncertainty or refusal)

MEDIUM PRIORITY SIGNALS:
1. Speech Pattern Completions:
   - Self-corrections or false starts that resolve into a complete thought
   - Topic changes that express a complete statement

2. Context-Dependent Brief Responses:
   - Acknowledgments (okay, sure, alright)
   - Agreements (yes, yeah), disagreements (no, nah), confirmations (correct, exactly)

LOW PRIORITY SIGNALS:
1. STT Artifacts:
   - Repeated words, unusual punctuation, capitalization errors, word insertions/deletions

2. Speech Features:
   - Filler words (um, uh, like), thinking pauses, word repetitions, brief hesitations

SPECIAL RULES FOR AMBIGUOUS OR FRAGMENTED UTTERANCES:
1. Ambiguous Keywords in Isolation:
   - If the input consists solely of an ambiguous keyword (e.g., "technical" or "voice agent") without additional context, treat the utterance as incomplete and output NO.
   - Do not infer intent (e.g., consultancy vs. development) from a single ambiguous word.

2. Partial Name or Interest Utterances:
   - In contexts where a full name is expected, if the user only says fragments such as "My name is" or "the real" without a complete name following, output NO.
   - Only output YES when the utterance includes a clear, complete name (e.g., "My name is John Smith").

3. Primary Interest Specific Rule:
   - When responding to the primary interest prompt, if the user's utterance ends with or contains an ambiguous keyword like "technical" or "voice agent" without a disambiguating term (e.g., "consultancy" or "development"), and the overall response appears incomplete, output NO.
   - For example, "I think I'm interested in technical" should be considered incomplete (NO) because it lacks the full term "technical consultancy."

DECISION RULES:
1. Return YES if:
   - Any high priority signal shows clear completion.
   - Medium priority signals combine to show a complete thought.
   - The meaning is clear despite minor STT artifacts.
   - The utterance, even if brief (e.g., "Yes", "No", or a complete question/statement), is unambiguous.

2. Return NO if:
   - No high priority signals are present.
   - The utterance trails off or contains multiple incomplete indicators.
   - The user appears to be mid-formulation or provides only a fragment.
   - The response consists solely of ambiguous keywords (per the Special Rules above) or partial phrases where a complete response is expected.
   - In responses to the primary interest prompt, if the response ends with an ambiguous term (e.g., "technical" or "voice agent") without the necessary qualifier, output NO.

3. When Uncertain:
   - If you can understand the intent and it appears complete, return YES.
   - If the meaning is unclear or the response seems unfinished, return NO.
   - Always make a binary decision and never ask for clarification.

# SCENARIO-SPECIFIC EXAMPLES

## Phase 1: Recording Consent
Assistant: We record our calls for quality assurance and training. Is that ok with you?
- User: Yes → Output: YES
- User: No → Output: YES
- User: Why do you need to record? → Output: YES
- User: Why do you → Output: NO
- User: Uhhh → Output: NO
- User: If I have to but → Output: NO
- User: um → Output: NO
- User: Well I suppose it → Output: NO

## Phase 2: Name and Interest Collection
Assistant: May I know your name please?
- User: My name is John Smith → Output: YES
- User: I don't want to give you my name → Output: YES
- User: Why do you need my name? → Output: YES
- User: I don't want to tell you → Output: NO
- User: What do you uh → Output: NO

Assistant: Could you tell me if you're interested in technical consultancy or voice agent development?
- User: I'm interested in technical consultancy → Output: YES
- User: I'm interested in voice agent development → Output: YES
- User: technical → Output: NO  *(Ambiguous keyword without context)*
- User: voice agent → Output: NO  *(Ambiguous keyword without context)*
- User: I think I'm interested in technical → Output: NO  *(Incomplete response lacking the full qualifier)*
- User: I think I'm interested in voice agent → Output: NO  *(Incomplete response lacking the full qualifier)*
- User: Well maybe I → Output: NO
- User: uhm sorry hold on → Output: YES
- User: What's the difference? → Output: YES
- User: I'm really not sure at the moment. → Output: YES
- User: Tell me more about both options first. → Output: YES
- User: I'd rather not say. → Output: YES
- User: Actually, I have a different question for you. → Output: YES

## Phase 3: Lead Qualification (Voice Agent Development Only)
Assistant: So John, what tasks or interactions are you hoping your voice AI agent will handle?
- User: I want it to handle customer service inquiries → Output: YES
- User: Just some stuff → Output: YES
- User: What kind of things can it do? → Output: YES
- User: I was thinking maybe it could → Output: NO

Assistant: And have you thought about what timeline you're looking to get this project completed in, John?
- User: I'm hoping to get this done in the next three months → Output: YES
- User: Not really → Output: YES
- User: ASAP → Output: YES
- User: I was hoping to get it → Output: NO

Assistant: May I know what budget you've allocated for this project, John?
- User: £2000 → Output: YES
- User: £500 → Output: YES
- User: I don't have a budget yet → Output: YES
- User: Well I was thinking → Output: NO
- User: I'm not sure → Output: YES

Assistant: And finally, John, how would you rate the quality of our interaction so far in terms of speed, accuracy, and helpfulness?
- User: I think it's been pretty good → Output: YES
- User: It was ok → Output: YES

## Phase 4: Closing the Call
Assistant: Thank you for your time John. Have a wonderful day.
- User: um → Output: NO

Assistant: And finally, John, how would you rate the quality of our interaction so far in terms of speed, accuracy, and helpfulness?
- User: Well I think it → Output: NO
"""


class AudioAccumulator(FrameProcessor):
    """Buffers user audio until the user stops speaking.

    Always pushes a fresh context with a single audio message.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._audio_frames = []
        self._start_secs = 0.4  # this should match VAD start_secs (hardcoding for now)
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


class CompletenessCheck(FrameProcessor):
    """Checks the result of the classifier LLM to determine if the user has finished speaking.

    Triggers the notifier if the user has finished speaking. Also triggers the notifier if an
    idle timeout is reached.
    """

    wait_time = 2.0

    def __init__(self, notifier: BaseNotifier, audio_accumulator: AudioAccumulator, **kwargs):
        super().__init__()
        self._notifier = notifier
        self._audio_accumulator = audio_accumulator
        self._idle_task = None
        self._wakeup_time = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (EndFrame, CancelFrame)):
            if self._idle_task:
                await self.cancel_task(self._idle_task)
                self._idle_task = None
        elif isinstance(frame, UserStartedSpeakingFrame):
            if self._idle_task:
                await self.cancel_task(self._idle_task)
        elif isinstance(frame, TextFrame) and frame.text.startswith("YES"):
            logger.debug("Completeness check YES")
            if self._idle_task:
                await self.cancel_task(self._idle_task)
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
                    self._idle_task = self.create_task(self._idle_task_handler())
        else:
            await self.push_frame(frame, direction)

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

def get_message_field(message: object, field: str) -> any:
    """
    Retrieve a field from a message.
    If message is a dict, return message[field].
    Otherwise, use getattr.
    """
    if isinstance(message, dict):
        return message.get(field)
    return getattr(message, field, None)


def get_message_text(message: object) -> str:
    """
    Extract text content from a message, handling both dict and Google Content formats.
    """
    # logger.debug(f"Processing message: {message}")

    # First try Google's format with parts array
    parts = get_message_field(message, "parts")
    # logger.debug(f"Found parts: {parts}")

    if parts:
        # Google format with parts array
        text_parts = []
        for part in parts:
            if isinstance(part, dict):
                text = part.get("text", "")
            else:
                text = getattr(part, "text", "")
            if text:
                text_parts.append(text)
        result = " ".join(text_parts)
        # logger.debug(f"Extracted text from parts: {result}")
        return result

    # Try direct content field
    content = get_message_field(message, "content")
    # logger.debug(f"Found content: {content}")

    if isinstance(content, str):
        # logger.debug(f"Using string content: {content}")
        return content
    elif isinstance(content, list):
        # Handle content that might be a list of parts
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text", "")
                if text:
                    text_parts.append(text)
        if text_parts:
            result = " ".join(text_parts)
            # logger.debug(f"Extracted text from content list: {result}")
            return result

    # logger.debug("No text content found, returning empty string")
    return ""


class StatementJudgeContextFilter(FrameProcessor):
    """Extracts recent user messages and constructs an LLMMessagesFrame for the classifier LLM.

    This processor takes the OpenAILLMContextFrame from the main conversation context,
    extracts the most recent user messages, and creates a simplified LLMMessagesFrame
    for the statement classifier LLM to determine if the user has finished speaking.
    """

    def __init__(self, notifier: BaseNotifier, **kwargs):
        super().__init__(**kwargs)
        self._notifier = notifier
        self.Content = Content
        self.Part = Part

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
            logger.debug(f"Processing context messages: {messages}")

            user_text_messages = []
            last_assistant_message = None
            for message in reversed(messages):
                role = get_message_field(message, "role")
                logger.debug(f"Processing message with role: {role}")

                if role != "user":
                    if role == "assistant" or role == "model":
                        last_assistant_message = message
                        logger.debug(f"Found assistant/model message: {message}")
                    break

                text = get_message_text(message)
                logger.debug(f"Extracted user message text: {text}")
                if text:
                    user_text_messages.append(text)

            # If we have any user text content, push an LLMMessagesFrame
            if user_text_messages:
                user_message = " ".join(reversed(user_text_messages))
                logger.debug(f"Final user message: {user_message}")
                
                # Create messages using the correct Google Content objects
                messages = [
                    self.Content(role="user", parts=[self.Part(text=classifier_system_instruction)])
                ]
                
                if last_assistant_message:
                    assistant_text = get_message_text(last_assistant_message)
                    logger.debug(f"Assistant message text: {assistant_text}")
                    if assistant_text:
                        messages.append(
                            self.Content(role="model", parts=[self.Part(text=assistant_text)])
                        )
                
                messages.append(
                    self.Content(role="user", parts=[self.Part(text=user_message)])
                )
                
                await self.push_frame(LLMMessagesFrame(messages))
            return

        # Fallback: for any frames not otherwise handled, forward them.
        await self.push_frame(frame, direction)


class OutputGate(FrameProcessor):
    def __init__(self, *, notifier: BaseNotifier, start_open: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._gate_open = start_open
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
        self._gate_task = self.create_task(self._gate_task_handler())

    async def _stop(self):
        await self.cancel_task(self._gate_task)

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
