# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger
from openai.types.chat import ChatCompletionToolParam

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.ai_services import LLMService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyDialinSettings, DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

daily_api_key = os.getenv("DAILY_API_KEY", "")
daily_api_url = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")


async def main(room_url: str, token: str, callId: str, callDomain: str):
    # diallin_settings are only needed if Daily's SIP URI is used
    # If you are handling this via Twilio, Telnyx, set this to None
    # and handle call-forwarding when on_dialin_ready fires.
    diallin_settings = DailyDialinSettings(call_id=callId, call_domain=callDomain)

    transport = DailyTransport(
        room_url,
        token,
        "Chatbot",
        DailyParams(
            api_url=daily_api_url,
            api_key=daily_api_key,
            dialin_settings=diallin_settings,
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

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

    content = f"""
You are a delivery service customer support specialist supporting customers with their orders. 
Begin with: "Hello, this is Hailey from customer support. What can I help you with today?"
    """

    messages = [
        {
            "role": "system",
            "content": content,
        },
    ]

    tools = [
        ChatCompletionToolParam(
            type="function",
            function={
                "name": "transfer_call",
                "description": "Transfer the call to a person. This function is used to connect the call to a real person. Examples of real people are: managers, supervisors, or other customer support specialists. Any person is okay as long as they are not a bot.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "call_id": {
                            "type": "string",
                            "description": "This is always {callId}.",
                        },
                        "summary": {
                            "type": "string",
                            "description": """
Provide a concise summary in 3-5 sentences. Highlight any  important details or unusual aspects of the conversation.
                            """,
                        },
                    },
                },
            },
        )
    ]

    context = OpenAILLMContext(messages, tools)
    context_aggregator = llm.create_context_aggregator(context)

    async def default_transfer_call(
        function_name, tool_call_id, args, llm: LLMService, context, result_callback
    ):
        logger.debug(f"default_transfer_call: {function_name} {tool_call_id} {args}")
        await result_callback(
            {
                "transfer_call": False,
                "reason": "To transfer call calls, please dial in to the room using a phone or a SIP client.",
            }
        )

    llm.register_function(
        function_name="transfer_call",
        callback=default_transfer_call,
    )

    pipeline = Pipeline(
        [
            transport.input(),
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        await transport.capture_participant_transcription(participant["id"])
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        await task.queue_frame(EndFrame())

    @transport.event_handler("on_dialin_ready")
    async def on_dialin_ready(_, sip_endpoint):
        logger.info(f"on_dialin_ready: {sip_endpoint}")

    @transport.event_handler("on_dialin_connected")
    async def on_dialin_connected(transport, event):
        logger.info(f"on_dialin_connected: {event}")
        sip_session_id = event["sessionId"]

        async def transfer_call(
            function_name, tool_call_id, args, llm: LLMService, context, result_callback
        ):
            logger.debug(f"transfer_call: {function_name} {tool_call_id} {args}")

            # sip_url = "sip:your_user_name@sip.linphone.org"

            sip_url = (
                f"sip:your_username@dailyco.sip.twilio.com?x-daily_id={room_url.split('/')[-1]}"
            )

            try:
                await transport.sip_refer(
                    settings={
                        "sessionId": sip_session_id,
                        "toEndPoint": sip_url,
                    }
                )
            except Exception as e:
                logger.error(f"An error occurred during SIP refer: {e}")
                await result_callback({"transfer_call": False})

            await result_callback({"transfer_call": True})

        llm.register_function(
            function_name="transfer_call",
            callback=transfer_call,
        )

    runner = PipelineRunner()

    await runner.run(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Simple ChatBot")
    parser.add_argument("-u", type=str, help="Room URL")
    parser.add_argument("-t", type=str, help="Token")
    parser.add_argument("-i", type=str, help="Call ID")
    parser.add_argument("-d", type=str, help="Call Domain")
    config = parser.parse_args()

    asyncio.run(main(config.u, config.t, config.i, config.d))
