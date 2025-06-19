#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import os

from deepgram import LiveOptions
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


current_language = "English"


async def switch_language(params: FunctionCallParams):
    global current_language
    current_language = params.arguments["language"]
    await params.result_callback(
        {"voice": f"Your answers from now on should be in {current_language}."}
    )


async def english_filter(frame) -> bool:
    return current_language == "English"


async def spanish_filter(frame) -> bool:
    return current_language == "Spanish"


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

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"), live_options=LiveOptions(language="multi")
    )

    english_tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    spanish_tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="d4db5fb9-f44b-4bd1-85fa-192e0f0d75f9",  # Spanish-speaking Lady
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))
    llm.register_function("switch_language", switch_language)

    tools = [
        ChatCompletionToolParam(
            type="function",
            function={
                "name": "switch_language",
                "description": "Switch to another language when the user asks you to",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "language": {
                            "type": "string",
                            "description": "The language the user wants you to speak",
                        },
                    },
                    "required": ["language"],
                },
            },
        )
    ]
    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities. Respond to what the user said in a creative and helpful way. Your output should not include non-alphanumeric characters. You can speak the following languages: 'English' and 'Spanish'.",
        },
    ]

    context = OpenAILLMContext(messages, tools)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # STT
            context_aggregator.user(),  # User responses
            llm,  # LLM
            ParallelPipeline(  # TTS (bot will speak the chosen language)
                [FunctionFilter(english_filter), english_tts],  # English
                [FunctionFilter(spanish_filter), spanish_tts],  # Spanish
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
                "content": f"Please introduce yourself to the user and let them know the languages you speak. Your initial responses should be in {current_language}.",
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
