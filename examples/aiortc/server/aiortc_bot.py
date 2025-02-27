#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import os
import sys

from aiortc import MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRelay
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndTaskFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIProcessor
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.ai_services import LLMService
from pipecat.services.gemini_multimodal_live import GeminiMultimodalLiveLLMService
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.transports.webrtc.webrtc_connection import PipecatWebRTCConnection

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


SYSTEM_INSTRUCTION = f"""
"You are Gemini Chatbot, a friendly, helpful robot.

Your goal is to demonstrate your capabilities in a succinct way.

Your output will be converted to audio so don't include special characters in your answers.

Respond to what the user said in a creative and helpful way. Keep your responses brief. One or two sentences at most.
"""


async def terminate_call(
    function_name, tool_call_id, args, llm: LLMService, context, result_callback
):
    logger.debug("Will terminate call!")
    """Function the bot can call to terminate the call upon completion of a voicemail message."""
    await llm.queue_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
    await result_callback("Goodbye")


async def run_bot(websocket_client):
    ws_transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            audio_in_sample_rate=16000,  # Need to be 16_000 in order to VAD to work as expected
            audio_out_sample_rate=24000,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
            serializer=ProtobufFrameSerializer(),
        ),
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

    llm = GeminiMultimodalLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        voice_id="Puck",  # Aoede, Charon, Fenrir, Kore, Puck
        transcribe_user_audio=True,
        transcribe_model_audio=True,
        system_instruction=SYSTEM_INSTRUCTION,
        tools=tools,
    )
    llm.register_function("terminate_call", terminate_call)

    context = OpenAILLMContext()
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
            allow_interruptions=True,
            observers=[rtvi.observer()],
        ),
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        await rtvi.set_bot_ready()

    @ws_transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("WS Client connected")
        # Kick off the conversation.
        # messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @ws_transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("WS Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)


# TODO: only for testing
async def run_aiortc_bot(pipecat_connection: PipecatWebRTCConnection):
    relay = MediaRelay()
    ROOT = os.path.dirname(__file__)

    logger.info("Setting up media handling for the bot")

    player = MediaPlayer(os.path.join(ROOT, "demo-instruct.wav"))
    recorder = MediaBlackhole()
    await recorder.start()

    def handle_track(track: MediaStreamTrack):
        if track.kind == "audio":
            pipecat_connection.replace_audio_track(player.audio)
            recorder.addTrack(track)
        elif track.kind == "video":
            pipecat_connection.replace_video_track(relay.subscribe(track))

    @pipecat_connection.on("connected")
    def on_connected():
        logger.info("Peer connection established.")

    @pipecat_connection.on("disconnected")
    async def on_disconnected():
        logger.info("Peer connection lost.")

    @pipecat_connection.on("track-started")
    def on_track_started(track: MediaStreamTrack):
        logger.info(f"Processing new track: {track.kind}")
        handle_track(track)

    @pipecat_connection.on("track-ended")
    async def on_track_ended(track):
        logger.info(f"Track ended: {track.kind}")
        await recorder.stop()

    # Checking in case already had some existent track
    for track in pipecat_connection.tracks():
        logger.info(f"handling existent track: {track.kind}")
        handle_track(track)
