#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import argparse
import asyncio
import os
import sys
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    EndFrame,
    EndTaskFrame,
    InputAudioRawFrame,
    StopTaskFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.ai_services import LLMService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.google import GoogleLLMService
from pipecat.services.google.google import GoogleLLMContext
from pipecat.transports.services.daily import (
    DailyDialinSettings,
    DailyParams,
    DailyTransport,
)

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


daily_api_key = os.getenv("DAILY_API_KEY", "")
daily_api_url = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")

system_message = None


class UserAudioCollector(FrameProcessor):
    """This FrameProcessor collects audio frames in a buffer, then adds them to the
    LLM context when the user stops speaking.
    """

    def __init__(self, context, user_context_aggregator):
        super().__init__()
        self._context = context
        self._user_context_aggregator = user_context_aggregator
        self._audio_frames = []
        self._start_secs = 0.2  # this should match VAD start_secs (hardcoding for now)
        self._user_speaking = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            # We could gracefully handle both audio input and text/transcription input ...
            # but let's leave that as an exercise to the reader. :-)
            return
        if isinstance(frame, UserStartedSpeakingFrame):
            self._user_speaking = True
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_speaking = False
            self._context.add_audio_frames_message(audio_frames=self._audio_frames)
            await self._user_context_aggregator.push_frame(
                self._user_context_aggregator.get_context_frame()
            )
        elif isinstance(frame, InputAudioRawFrame):
            if self._user_speaking:
                self._audio_frames.append(frame)
            else:
                # Append the audio frame to our buffer. Treat the buffer as a ring buffer, dropping the oldest
                # frames as necessary. Assume all audio frames have the same duration.
                self._audio_frames.append(frame)
                frame_duration = len(frame.audio) / 16 * frame.num_channels / frame.sample_rate
                buffer_duration = frame_duration * len(self._audio_frames)
                while buffer_duration > self._start_secs:
                    self._audio_frames.pop(0)
                    buffer_duration -= frame_duration

        await self.push_frame(frame, direction)


class ContextSwitcher:
    def __init__(self, llm, context_aggregator):
        self._llm = llm
        self._context_aggregator = context_aggregator

    async def switch_context(self, system_instruction):
        """Switch the context to a new system instruction based on what the bot hears."""
        # Create messages with updated system instruction
        messages = [
            {
                "role": "system",
                "content": system_instruction,
            }
        ]

        # Update context with new messages
        self._context_aggregator.set_messages(messages)
        # Get the context frame with the updated messages
        context_frame = self._context_aggregator.get_context_frame()
        # Trigger LLM response by pushing a context frame
        await self._llm.push_frame(context_frame)


class FunctionHandlers:
    def __init__(self, context_switcher):
        self.context_switcher = context_switcher

    async def voicemail_response(
        self,
        function_name,
        tool_call_id,
        args,
        llm: LLMService,
        context,
        result_callback,
    ):
        """Function the bot can call to leave a voicemail message."""
        message = """You are Chatbot leaving a voicemail message. Say EXACTLY this message and nothing else:

                    "Hello, this is a message for Pipecat example user. This is Chatbot. Please call back on 123-456-7891. Thank you."

                    After saying this message, call the terminate_call function."""

        await self.context_switcher.switch_context(system_instruction=message)
        await result_callback("Leaving a voicemail message")

    async def human_conversation(
        self,
        function_name,
        tool_call_id,
        args,
        llm: LLMService,
        context,
        result_callback,
    ):
        """Function the bot can when it detects it's talking to a human."""
        await llm.push_frame(StopTaskFrame(), FrameDirection.UPSTREAM)


async def terminate_call(
    function_name,
    tool_call_id,
    args,
    llm: LLMService,
    context,
    result_callback,
    call_state=None,
):
    """Function the bot can call to terminate the call upon completion of the call."""
    if call_state:
        call_state.bot_terminated_call = True
    await llm.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)


async def main(
    room_url: str,
    token: str,
    callId: Optional[str],
    callDomain: Optional[str],
    detect_voicemail: bool,
    dialout_number: Optional[str],
):
    dialin_settings = None
    if callId and callDomain:
        dialin_settings = DailyDialinSettings(call_id=callId, call_domain=callDomain)
        transport_params = DailyParams(
            api_url=daily_api_url,
            api_key=daily_api_key,
            dialin_settings=dialin_settings,
            audio_in_enabled=True,
            audio_out_enabled=True,
            camera_out_enabled=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
        )
    else:
        transport_params = DailyParams(
            api_url=daily_api_url,
            api_key=daily_api_key,
            audio_in_enabled=True,
            audio_out_enabled=True,
            camera_out_enabled=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
        )

    class CallState:
        participant_left_early = False
        bot_terminated_call = False

    call_state = CallState()

    transport = DailyTransport(
        room_url,
        token,
        "Chatbot",
        transport_params,
    )

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY", ""),
        voice_id=os.getenv("ELEVENLABS_VOICE_ID", ""),
    )

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    ### VOICEMAIL PIPELINE

    tools = [
        {
            "function_declarations": [
                {
                    "name": "switch_to_voicemail_response",
                    "description": "Call this function when you detect this is a voicemail system.",
                },
                {
                    "name": "switch_to_human_conversation",
                    "description": "Call this function when you detect this is a human.",
                },
                {
                    "name": "terminate_call",
                    "description": "Call this function to terminate the call.",
                },
            ]
        }
    ]

    system_instruction = """You are Chatbot trying to determine if this is a voicemail system or a human.

    If you hear any of these phrases (or very similar ones):
    - "Please leave a message after the beep"
    - "No one is available to take your call"
    - "Record your message after the tone"
    - "You have reached voicemail for..."
    - "You have reached [phone number]"
    - "[phone number] is unavailable"
    - "The person you are trying to reach..."
    - "The number you have dialed..."
    - "Your call has been forwarded to an automated voice messaging system"

    Then call the function switch_to_voicemail_response.

    If it sounds like a human (saying hello, asking questions, etc.), call the function switch_to_human_conversation.

    DO NOT say anything until you've determined if this is a voicemail or human."""

    voicemail_detection_llm = GoogleLLMService(
        model="models/gemini-2.0-flash-lite",
        api_key=os.getenv("GOOGLE_API_KEY"),
        system_instruction=system_instruction,
        tools=tools,
    )

    voicemail_detection_context = GoogleLLMContext()
    voicemail_detection_context_aggregator = voicemail_detection_llm.create_context_aggregator(
        voicemail_detection_context
    )
    context_switcher = ContextSwitcher(
        voicemail_detection_llm, voicemail_detection_context_aggregator.user()
    )
    handlers = FunctionHandlers(context_switcher)

    voicemail_detection_llm.register_function(
        "switch_to_voicemail_response", handlers.voicemail_response
    )
    voicemail_detection_llm.register_function(
        "switch_to_human_conversation", handlers.human_conversation
    )
    voicemail_detection_llm.register_function(
        "terminate_call",
        lambda *args, **kwargs: terminate_call(*args, **kwargs, call_state=call_state),
    )

    voicemail_detection_audio_collector = UserAudioCollector(
        voicemail_detection_context, voicemail_detection_context_aggregator.user()
    )

    voicemail_detection_pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            voicemail_detection_audio_collector,  # Collect audio frames
            voicemail_detection_context_aggregator.user(),  # User responses
            voicemail_detection_llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            voicemail_detection_context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )
    voicemail_detection_pipeline_task = PipelineTask(
        voicemail_detection_pipeline,
        params=PipelineParams(allow_interruptions=True),
    )

    if dialout_number:
        logger.debug("dialout number detected; doing dialout")

        # Configure some handlers for dialing out
        @transport.event_handler("on_joined")
        async def on_joined(transport, data):
            logger.debug(f"Joined; starting dialout to: {dialout_number}")
            await transport.start_dialout({"phoneNumber": dialout_number})

        @transport.event_handler("on_dialout_connected")
        async def on_dialout_connected(transport, data):
            logger.debug(f"Dial-out connected: {data}")

        @transport.event_handler("on_dialout_answered")
        async def on_dialout_answered(transport, data):
            logger.debug(f"Dial-out answered: {data}")

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            # unlike the dialin case, for the dialout case, the caller will speak first. Presumably
            # they will answer the phone and say "Hello?" Since we've captured their transcript,
            # That will put a frame into the pipeline and prompt an LLM completion, which is how the
            # bot will then greet the user.
    elif detect_voicemail:
        logger.debug("Detect voicemail example. You can test this in example in Daily Prebuilt")

        # For the voicemail detection case, we do not want the bot to answer the phone. We want it to wait for the voicemail
        # machine to say something like 'Leave a message after the beep', or for the user to say 'Hello?'.
        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            logger.debug("Detect voicemail; capturing participant transcription")
            await transport.capture_participant_transcription(participant["id"])
    else:
        logger.debug("+++++ No dialout number; assuming dialin")

        # Different handlers for dialin
        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            # This event is not firing for some reason
            await transport.capture_participant_transcription(participant["id"])
            dialin_instructions = """Always call the function switch_to_human_conversation"""
            messages = [
                {
                    "role": "system",
                    "content": dialin_instructions,
                }
            ]
            voicemail_detection_context_aggregator.user().set_messages(messages)
            await voicemail_detection_pipeline_task.queue_frames(
                [voicemail_detection_context_aggregator.user().get_context_frame()]
            )

    runner = PipelineRunner()

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        call_state.participant_left_early = True
        await voicemail_detection_pipeline_task.queue_frame(EndFrame())

    print("!!! starting voicemail detection pipeline")
    await runner.run(voicemail_detection_pipeline_task)
    print("!!! Done with voicemail detection pipeline")

    if call_state.participant_left_early or call_state.bot_terminated_call:
        if call_state.participant_left_early:
            print("!!! Participant left early; terminating call")
        elif call_state.bot_terminated_call:
            print("!!! Bot terminated call; not proceeding to human conversation")
        return

    ### HUMAN CONVERSATION PIPELINE

    human_conversation_system_instruction = """You are Chatbot talking to a human. Be friendly and helpful.

    Start with: "Hello! I'm a friendly chatbot. How can I help you today?"

    Keep your responses brief and to the point. Listen to what the person says.

    When the person indicates they're done with the conversation by saying something like:
    - "Goodbye"
    - "That's all"
    - "I'm done"
    - "Thank you, that's all I needed"

    THEN say: "Thank you for chatting. Goodbye!" and call the terminate_call function."""

    human_conversation_llm = GoogleLLMService(
        model="models/gemini-2.0-flash-001",
        api_key=os.getenv("GOOGLE_API_KEY"),
        system_instruction=human_conversation_system_instruction,
        tools=tools,
    )
    human_conversation_context = GoogleLLMContext()

    human_conversation_context_aggregator = human_conversation_llm.create_context_aggregator(
        human_conversation_context
    )

    human_conversation_llm.register_function(
        "terminate_call",
        lambda *args, **kwargs: terminate_call(*args, **kwargs, call_state=call_state),
    )

    human_conversation_pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
            human_conversation_context_aggregator.user(),  # User responses
            human_conversation_llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            human_conversation_context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    human_conversation_pipeline_task = PipelineTask(
        human_conversation_pipeline,
        params=PipelineParams(allow_interruptions=True),
    )

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        await voicemail_detection_pipeline_task.queue_frame(EndFrame())
        await human_conversation_pipeline_task.queue_frame(EndFrame())

    print("!!! starting human conversation pipeline")
    human_conversation_context_aggregator.user().set_messages(
        [
            {
                "role": "system",
                "content": human_conversation_system_instruction,
            }
        ]
    )
    await human_conversation_pipeline_task.queue_frames(
        [human_conversation_context_aggregator.user().get_context_frame()]
    )
    await runner.run(human_conversation_pipeline_task)

    print("!!! Done with human conversation pipeline")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Simple ChatBot")
    parser.add_argument("-u", type=str, help="Room URL")
    parser.add_argument("-t", type=str, help="Token")
    parser.add_argument("-i", type=str, help="Call ID")
    parser.add_argument("-d", type=str, help="Call Domain")
    parser.add_argument("-v", action="store_true", help="Detect voicemail")
    parser.add_argument("-o", type=str, help="Dialout number", default=None)
    config = parser.parse_args()

    asyncio.run(main(config.u, config.t, config.i, config.d, config.v, config.o))
