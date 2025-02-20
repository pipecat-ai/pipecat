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
from pipecat.frames.frames import EndTaskFrame, LLMMessagesUpdateFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import LLMService
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.google import GoogleLLMContext, GoogleLLMService
from pipecat.transports.services.daily import DailyDialinSettings, DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


daily_api_key = os.getenv("DAILY_API_KEY", "")
daily_api_url = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")


async def terminate_call(
    function_name, tool_call_id, args, llm: LLMService, context, result_callback
):
    """Function the bot can call to terminate the call upon completion of a voicemail message."""
    await llm.queue_frame(EndTaskFrame(), FrameDirection.UPSTREAM)


async def respond_with_apple(
    function_name, tool_call_id, args, llm: LLMService, context: GoogleLLMContext, result_callback
):
    messages = [
        {
            "role": "system",
            "content": "Always respond with Apple",
        }
    ]
    print("respond_with_apple")
    # context.system_message = "Always respond with Apple"
    print(f"context before: {context.tools}")
    await llm.push_frame(LLMMessagesUpdateFrame(messages))
    print(f"context after: {context.tools}")


async def respond_with_banana(
    function_name, tool_call_id, args, llm: LLMService, context: GoogleLLMContext, result_callback
):
    messages = [
        {
            "role": "system",
            "content": "Always respond with Banana",
        }
    ]
    print("respond_with_banana")
    # context.system_message = "Always respond with Banana"
    print(f"context before: {context.tools}")
    await llm.push_frame(LLMMessagesUpdateFrame(messages))
    print(f"context after: {context.tools}")


async def respond_with_orange(
    function_name, tool_call_id, args, llm: LLMService, context: GoogleLLMContext, result_callback
):
    messages = [
        {
            "role": "system",
            "content": "Always respond with Orange",
        }
    ]
    print("respond_with_orange")
    # context.system_message = "Always respond with Orange"
    print(f"context before: {context.tools}")
    await llm.push_frame(LLMMessagesUpdateFrame(messages))
    print(f"context after: {context.tools}")


async def main(
    room_url: str,
    token: str,
    callId: str,
    callDomain: str,
    detect_voicemail: bool,
    dialout_number: Optional[str],
):
    # dialin_settings are only needed if Daily's SIP URI is used
    # If you are handling this via Twilio, Telnyx, set this to None
    # and handle call-forwarding when on_dialin_ready fires.
    dialin_settings = DailyDialinSettings(call_id=callId, call_domain=callDomain)
    transport = DailyTransport(
        room_url,
        token,
        "Chatbot",
        DailyParams(
            api_url=daily_api_url,
            api_key=daily_api_key,
            dialin_settings=dialin_settings,
            audio_in_enabled=True,
            audio_out_enabled=True,
            camera_out_enabled=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            transcription_enabled=True,
        ),
    )

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY", ""),
        voice_id=os.getenv("ELEVENLABS_VOICE_ID", ""),
    )

    tools = [
        {
            "function_declarations": [
                {
                    "name": "respond_with_banana",
                    "description": "Call this function when the user asks about bananas.",
                },
                {
                    "name": "respond_with_orange",
                    "description": "Call this function when the user asks about oranges.",
                },
                {
                    "name": "respond_with_apple",
                    "description": "Call this function when the user asks about apples.",
                },
            ]
        }
    ]

    system_instruction2 = """You are Chatbot, a friendly, helpful robot.

IMPORTANT: You MUST use the terminate_call function to end the call in these situations:
1. After leaving a voicemail message
2. When the conversation with a human is finished

VOICEMAIL DETECTION:
- Listen carefully for these exact phrases at the start of the call:
  * "Please leave a message after the beep"
  * "No one is available to take your call"
  * "Record your message after the tone"
  * "You have reached voicemail for..."

IF VOICEMAIL DETECTED:
1. Wait for any beep sound if mentioned
2. Say EXACTLY: "Hello, this is a message for Pipecat example user. This is Chatbot. Please call back on 123-456-7891. Thank you."
3. IMMEDIATELY call the terminate_call function after your message
4. Do not say anything else

IF HUMAN DETECTED:
1. Say: "Oh, hello! I'm a friendly chatbot. Is there anything I can help you with?"
2. Keep responses short and helpful
3. When conversation ends, say: "Okay, thank you! Have a great day!"
4. IMMEDIATELY call the terminate_call function

NEVER say these phrases yourself:
- "Please leave a message after the beep"
- "No one is available to take your call"
- "Record your message after the tone"
- "You have reached voicemail for..."
"""

    system_instruction3 = """
    You are Chatbot. Your MAIN GOAL is to call the terminate_call function at the end.

After each response, YOU MUST call the terminate_call function. This is REQUIRED.

If someone says "Please leave a message after the beep":
Say: "Hello, this is a message for Pipecat example user. This is Chatbot. Please call back on 123-456-7891. Thank you."

If someone says anything else:
Say: "Hello, I'm Chatbot. Nice to meet you."

IMPORTANT: YOU MUST CALL the terminate_call function after you respond.
terminate_call is the ONLY way to end the call properly.
    """

    system_instruction1 = """You are Chatbot. Follow these exact steps in order:
1. Say "Hi, I'm Chatbot! Here's a joke: Why don't scientists trust atoms? Because they make up everything!"
2. IMMEDIATELY after telling the joke, call the function terminate_call"""

    system_instruction = """Always respond with the word Apple"""

    llm = GoogleLLMService(
        model="models/gemini-2.0-flash-lite-preview-02-05",
        api_key=os.getenv("GOOGLE_API_KEY"),
        system_instruction=system_instruction,
        tools=tools,
    )

    # llm.register_function("terminate_call", terminate_call)
    llm.register_function("respond_with_apple", respond_with_apple)
    llm.register_function("respond_with_banana", respond_with_banana)
    llm.register_function("respond_with_orange", respond_with_orange)

    context = GoogleLLMContext()

    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        PipelineParams(allow_interruptions=True),
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
            await transport.capture_participant_transcription(participant["id"])
    else:
        logger.debug("no dialout number; assuming dialin")

        # Different handlers for dialin
        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            # For the dialin case, we want the bot to answer the phone and greet the user. We
            # can prompt the bot to speak by putting the context into the pipeline.
            await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        await task.cancel()

    runner = PipelineRunner()

    await runner.run(task)


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
