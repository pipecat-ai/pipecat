#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame, LLMMessagesFrame, TTSTextFrame, UserStartedSpeakingFrame
from pipecat.observers.loggers.debug_log_observer import DebugLogObserver, FrameEndpoint
from pipecat.observers.loggers.llm_log_observer import LLMLogObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.filters.stt_mute_filter import (
    STTMuteConfig,
    STTMuteFilter,
    STTMuteFrame,
    STTMuteStrategy,
)
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
from pipecat.transports.services.daily import DailyParams

load_dotenv(override=True)


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


async def run_example(transport: BaseTransport, _: argparse.Namespace, handle_sigint: bool):
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    # Configure the mute processor with both strategies
    stt_mute_processor = STTMuteFilter(
        config=STTMuteConfig(
            strategies={
                STTMuteStrategy.FUNCTION_CALL,
                STTMuteStrategy.MUTE_UNTIL_FIRST_BOT_COMPLETE,
            }
        ),
    )

    tts = DeepgramTTSService(api_key=os.getenv("DEEPGRAM_API_KEY"), voice="aura-helios-en")

    async def transfer_to_human(params: FunctionCallParams):
        # Add a delay to test interruption during function calls

        caller_name = params.arguments.get("caller_name", "Unknown")
        human_agent_name = params.arguments.get("human_agent_name", "Unknown")
        logger.info(f"Transfer starting... {caller_name} wants to transfer to {human_agent_name}")
        await task.queue_frame(STTMuteFrame(True))
        await asyncio.sleep(
            5
        )  # 5-second delay to simulate a transfer. You could play hold music here too.
        messages.clear()
        messages.append(
            {
                "role": "system",
                "content": f"You are an agent named {human_agent_name}. Greet {caller_name} and let them know you are taking over the conversation.",
            }
        )
        await params.llm.push_frame(LLMMessagesFrame(messages))
        logger.info("Transfer complete, calling result callback")
        await params.result_callback({"transfer_successful": True})

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))
    llm.register_function("transfer_to_human", transfer_to_human)

    transfer_function = FunctionSchema(
        name="transfer_to_human",
        description="Transfer the conversation to a human agent.",
        properties={
            "caller_name": {
                "type": "string",
                "description": "The name of the person who is calling. This will be used to greet them.",
            },
            "human_agent_name": {
                "type": "string",
                "description": "The name of the human agent to transfer the conversation to.",
            },
        },
        required=["caller_name", "human_agent_name"],
    )
    tools = ToolsSchema(standard_tools=[transfer_function])

    messages = [
        {
            "role": "system",
            "content": "You are a cheerful and helpful assistant named Bob. It is your job to ask the user their name, and the name of the person they want to transfer the conversation to. Start by introducing yourself and asking for the user's name.",
        },
    ]

    context = OpenAILLMContext(messages, tools)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt_mute_processor,  # Add the mute processor before STT
            stt,  # STT
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[
            LLMLogObserver(),
            DebugLogObserver(
                frame_types={
                    TTSTextFrame: (BaseOutputTransport, FrameEndpoint.DESTINATION),
                    UserStartedSpeakingFrame: (BaseInputTransport, FrameEndpoint.SOURCE),
                    EndFrame: None,
                }
            ),
        ],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation with a weather-related prompt
        messages.append(
            {
                "role": "system",
                "content": "Ask the user what city they'd like to know the weather for.",
            }
        )
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)

    await runner.run(task)


if __name__ == "__main__":
    from pipecat.examples.run import main

    main(run_example, transport_params=transport_params)
