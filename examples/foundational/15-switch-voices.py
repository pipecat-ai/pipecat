#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import os

from dotenv import load_dotenv
from loguru import logger
from openai.types.chat import ChatCompletionToolParam

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
from pipecat.transports.services.daily import DailyParams

load_dotenv(override=True)


current_voice = "News Lady"


async def switch_voice(params: FunctionCallParams):
    global current_voice
    current_voice = params.arguments["voice"]
    await params.result_callback(
        {
            "voice": f"You are now using your {current_voice} voice. Your responses should now be as if you were a {current_voice}."
        }
    )


async def news_lady_filter(frame) -> bool:
    return current_voice == "News Lady"


async def british_lady_filter(frame) -> bool:
    return current_voice == "British Lady"


async def barbershop_man_filter(frame) -> bool:
    return current_voice == "Barbershop Man"


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

    news_lady = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="bf991597-6c13-47e4-8411-91ec2de5c466",  # Newslady
    )

    british_lady = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    barbershop_man = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="a0e99841-438c-4a64-b679-ae501e7d6091",  # Barbershop Man
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))
    llm.register_function("switch_voice", switch_voice)

    tools = [
        ChatCompletionToolParam(
            type="function",
            function={
                "name": "switch_voice",
                "description": "Switch your voice only when the user asks you to",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "voice": {
                            "type": "string",
                            "description": "The voice the user wants you to use",
                        },
                    },
                    "required": ["voice"],
                },
            },
        )
    ]
    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities. Respond to what the user said in a creative and helpful way. Your output should not include non-alphanumeric characters. You can do the following voices: 'News Lady', 'British Lady' and 'Barbershop Man'.",
        },
    ]

    context = OpenAILLMContext(messages, tools)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
            context_aggregator.user(),  # User responses
            llm,  # LLM
            ParallelPipeline(  # TTS (one of the following vocies)
                [FunctionFilter(news_lady_filter), news_lady],  # News Lady voice
                [
                    FunctionFilter(british_lady_filter),
                    british_lady,
                ],  # British Reading Lady voice
                [FunctionFilter(barbershop_man_filter), barbershop_man],  # Barbershop Man voice
            ),
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
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        messages.append(
            {
                "role": "system",
                "content": f"Please introduce yourself to the user and let them know the voices you can do. Your initial responses should be as if you were a {current_voice}.",
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
