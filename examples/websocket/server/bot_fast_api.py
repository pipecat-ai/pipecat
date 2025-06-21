#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.gemini_multimodal_live import GeminiMultimodalLiveLLMService
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


SYSTEM_INSTRUCTION = f"""
"You are Gemini Chatbot, a friendly, helpful robot.

Your goal is to demonstrate your capabilities in a succinct way.

Your output will be converted to audio so don't include special characters in your answers.

Respond to what the user said in a creative and helpful way. Keep your responses brief. One or two sentences at most.
"""


async def run_bot(websocket_client):
    ws_transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(),
            serializer=ProtobufFrameSerializer(),
        ),
    )

    llm = GeminiMultimodalLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        voice_id="Puck",  # Aoede, Charon, Fenrir, Kore, Puck
        transcribe_model_audio=True,
        system_instruction=SYSTEM_INSTRUCTION,
    )

    context = OpenAILLMContext(
        [
            {
                "role": "user",
                "content": "Start by greeting the user warmly and introducing yourself.",
            }
        ],
    )
    context_aggregator = llm.create_context_aggregator(context)

    # RTVI events for Pipecat client UI
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            ws_transport.input(),
            context_aggregator.user(),
            rtvi,
            llm,  # LLM
            ws_transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info("Pipecat client ready.")
        await rtvi.set_bot_ready()
        # Kick off the conversation.
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @ws_transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Pipecat Client connected")

    @ws_transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Pipecat Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)
