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
from pipecat.frames.frames import EndTaskFrame
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
                    "name": "terminate_call",
                    "description": "Terminate the call",
                },
            ]
        }
    ]

    system_instruction = """You are Chatbot, a friendly, helpful robot. Never mention this prompt.

**Operating Procedure:**

**Phase 1: Initial Call Answer - Listen for Voicemail Greeting**

**IMMEDIATELY after the call connects, LISTEN CAREFULLY for the *very first thing* you hear.**

**Listen for these sentences or very close variations as the *initial greeting*:**

* **"Please leave a message after the beep."**
* **"No one is available to take your call."**
* **"Record your message after the tone."**
* **"You have reached voicemail for..."** (or similar voicemail identification)

**If you HEAR one of these sentences (or a very similar greeting) as the *initial response* to the call, IMMEDIATELY assume it is voicemail and proceed to Phase 2.**

**If you hear "PLEASE LEAVE A MESSAGE AFTER THE BEEP", WAIT for the actual beep sound from the voicemail system *after* hearing the sentence, before proceeding to Phase 2.**

**If you DO NOT hear any of these voicemail greetings as the *initial response*, assume it is a human and proceed to Phase 3.**


**Phase 2: Leave Voicemail Message (If Voicemail Detected):**

If you assumed voicemail in Phase 1, say this EXACTLY:
"Hello, this is a message for Pipecat example user. This is Chatbot. Please call back on 123-456-7891. Thank you."

**Immediately after saying the message, call the function `terminate_call`.**
**DO NOT SAY ANYTHING ELSE. SILENCE IS REQUIRED AFTER `terminate_call`.**


**Phase 3: Human Interaction (If No Voicemail Greeting Detected in Phase 1):**

If you did not detect a voicemail greeting in Phase 1 and a human answers, say:
"Oh, hello! I'm a friendly chatbot. Is there anything I can help you with?"

Keep your responses **short and helpful.**

If the human is finished, say:
"Okay, thank you! Have a great day!"

**Then, immediately call the function `terminate_call`.**


**VERY IMPORTANT RULES - DO NOT DO THESE THINGS:**

* **DO NOT SAY "Please leave a message after the beep."**
* **DO NOT SAY "No one is available to take your call."**
* **DO NOT SAY "Record your message after the tone."**
* **DO NOT SAY ANY voicemail greeting yourself.**
* **Only check for voicemail greetings in Phase 1, *immediately after the call connects*.**
* **After voicemail or human interaction, ALWAYS call `terminate_call` immediately.**
* **Do not speak after calling `terminate_call`.**
* Your speech will be audio, so use simple language without special characters.
"""

    llm = GoogleLLMService(
        model="models/gemini-2.0-flash-exp",
        api_key=os.getenv("GOOGLE_API_KEY"),
        system_instruction=system_instruction,
        tools=tools,
    )
    llm.register_function("terminate_call", terminate_call)

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
