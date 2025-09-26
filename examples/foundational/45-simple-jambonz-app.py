#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
import json

from aiohttp import web
from loguru import logger
from pipecat.audio.vad.vad_analyzer import VADParams

from pipecat.audio.vad.silero import SileroVADAnalyzer

from pipecat.transports.jambonz import JambonzTransport, JambonzTransportParams
from pipecat.serializers.jambonz import JambonzFrameSerializer
from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.processors.aggregators.llm_response import LLMUserAggregatorParams
from pipecat.services.gladia.stt import GladiaSTTService
from pipecat.services.gladia.config import (
    GladiaInputParams,
    LanguageConfig,
    MessagesConfig,
)
from pipecat.transcriptions.language import Language

routes = web.RouteTableDef()


async def run_bot(transport):
    call_info = transport.get_call_info()

    stt = GladiaSTTService(
        api_key=os.getenv("GLADIA_API_KEY"),
        confidence=0.1,
        sample_rate=call_info[
            "sampleRate"
        ],  # You can change this but don't forget to update the stt_sample_rate in the JambonzFrameSerializer
        params=GladiaInputParams(
            language_config=LanguageConfig(
                languages=[Language.TL, Language.EN], code_switching=True
            ),
            messages_config=MessagesConfig(receive_partial_transcripts=True),
        ),
    )

    # In this scenario, I'm using BaseTen's API for the LLM. For speed and much cheaper than OpenAI.
    llm = OpenAILLMService(
        model="moonshotai/Kimi-K2-Instruct",
        api_key=os.getenv("BASETEN_API_KEY"),
        params=OpenAILLMService.InputParams(
            temperature=0,
        ),
        base_url="https://inference.baseten.co/v1",
    )

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        model="eleven_flash_v2_5",
        params=ElevenLabsTTSService.InputParams(
            stability=0.7,
            similarity_boost=0.8,
            style=0.5,
            use_speaker_boost=True,
            speed=1.1,
        ),
        sample_rate=8000,
    )

    context = OpenAILLMContext(
        [{"role": "system", "content": "You are a helpful assistant."}]
    )

    context_aggregator = llm.create_context_aggregator(
        context,
        user_params=LLMUserAggregatorParams(aggregation_timeout=0.5),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            allow_interruption=True,
        ),
        enable_turn_tracking=True,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Pipecat Client connected")
        await task.queue_frame(
            LLMMessagesAppendFrame(
                messages=[
                    {
                        "role": "system",
                        "content": f"Say hello.",
                    }
                ],
                run_llm=True,
            )
        )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Pipecat Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)


"""
Connect this endpoint to the LISTEN verb in Jambonz and set the listen url to this endpoint.

Link to Jambonz's documentation: https://jambonz.readthedocs.io/en/latest/voice/listen.html
"""


@routes.get("/stream")
async def stream(request):
    ws = web.WebSocketResponse(
        protocols=("audio.jambonz.org",),
        heartbeat=15,  # More frequent heartbeats for deployed environments
        compress=False,  # do not compress audio
        max_msg_size=0,  # no limit for audio frames
        timeout=30,  # Add explicit timeout
    )
    await ws.prepare(request)
    logger.info(f"WebSocket prepared, waiting for initial message...")
    # This is the initial metadata from Jambonz
    initial_message = await ws.receive()
    call_info = json.loads(initial_message.data)
    logger.info(f"Received call info from Jambonz Server: {call_info}")
    logger.info(f"Sample rate: {call_info.get('sampleRate')}")

    transport = JambonzTransport(
        websocket=ws,
        params=JambonzTransportParams(
            serializer=JambonzFrameSerializer(
                # reference: https://docs.jambonz.org/verbs/verbs/listen#streaming
                JambonzFrameSerializer.InputParams(
                    audio_in_sample_rate=call_info[
                        "sampleRate"
                    ],  # This is the sample rate Jambonz sends us
                    stt_sample_rate=call_info[
                        "sampleRate"
                    ],  # This is the sample_rate you set in your "stt"
                    audio_out_sample_rate="8000",  # This is the sample rate you set in "sampleRate" under "bidirectionalAudio"
                )
            ),
            audio_in_enabled=True,
            audio_out_enabled=True,
            sample_rate=call_info["sampleRate"],
            audio_in_sample_rate=call_info["sampleRate"],
            audio_out_sample_rate=call_info["sampleRate"],
            audio_in_channels=1,
            session_timeout=300,
            vad_analyzer=SileroVADAnalyzer(params=VADParams()),
        ),
    )

    transport.set_call_info(call_info)
    logger.info(f"Starting bot with transport...")

    await run_bot(transport)

    return ws


app = web.Application()
app.add_routes(routes)

if __name__ == "__main__":
    web.run_app(app, port=3002)
