#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import asyncio
import os
import sys

from loguru import logger
from openai.types.chat import ChatCompletionToolParam

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame, LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.ai_services import LLMService
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyDialinSettings, DailyParams, DailyTransport

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main(room_url: str, token: str, call_id: str, call_domain: str):
    """Main entrypoint for the voice bot process.

    :param room_url: The Daily.co room URL
    :param token: The Daily.co token
    :param callId: The call ID from Daily.co
    :param callDomain: The domain associated with the call
    """
    diallin_settings = DailyDialinSettings(call_id=call_id, call_domain=call_domain)

    transport = DailyTransport(
        room_url,
        token,
        "Chatbot",
        DailyParams(
            api_url="https://api.daily.co/v1/",
            api_key=os.getenv("DAILY_API_KEY", ""),
            dialin_settings=diallin_settings,
            audio_in_enabled=True,
            audio_out_enabled=True,
            camera_out_enabled=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            transcription_enabled=True,
        ),
    )

    cartesia_params = CartesiaTTSService.InputParams(
        speed=-0.1,
        emotion=["positivity:high", "curiosity"],
        language="en",
    )

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        # Use Helpful Woman voice by default
        voice_id="156fb8d2-335b-4950-9cb3-a2d33befec77",
        params=cartesia_params,
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

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
                            "description": "This is always {call_id}.",
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

    context = OpenAILLMContext(messages, tools)
    context_aggregator = llm.create_context_aggregator(context)
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    pipeline = Pipeline(
        [
            transport.input(),
            stt,  # STT
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(_, participant):
        logger.info(f"on_first_participant_joined: {participant}")
        # await transport.capture_participant_transcription(participant["id"]) Might not need this
        await task.queue_frames([LLMMessagesFrame(messages)])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(_, participant, reason):
        logger.info(f"on_participant_left: {participant} {reason}")
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
    asyncio.run(main())
