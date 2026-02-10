#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import logging
import os

import aiohttp
from anam import PersonaConfig
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.anam.video import AnamVideoService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

# Configure Python standard logging to show INFO messages from anam SDK
logging.getLogger("anam").setLevel(logging.DEBUG)

load_dotenv(override=True)

ANAM_SAMPLE_RATE = 24000

# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_out_enabled=True,
        video_out_is_live=True,
        video_out_width=720,
        video_out_height=480,
        video_out_bitrate=1_000_000,  # 1MBps
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_out_enabled=True,
        video_out_is_live=True,
        video_out_width=720,
        video_out_height=480,
        audio_out_sample_rate=48000,  # Anam WebRTC output (OPUS 48kHz stereo)
        audio_out_channels=2,
        audio_in_sample_rate=16000,  # WebRTC input
        audio_in_channels=1,
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="00967b2f-88a6-4a31-8153-110a92134b9f",
        params=CartesiaTTSService.InputParams(sample_rate=ANAM_SAMPLE_RATE),
    )

    llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"))

    avatar_id = os.getenv("ANAM_AVATAR_ID", "").strip().strip('"')
    persona_config = PersonaConfig(
        avatar_id=avatar_id,
        enable_audio_passthrough=True,
    )
    logger.info(f"Persona config: {persona_config}")

    anam = AnamVideoService(
        api_key=os.getenv("ANAM_API_KEY"),
        persona_config=persona_config,
        api_base_url="https://api.anam.ai",
        api_version="v1",
    )

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Your output will be spoken aloud, so avoid special characters that can't easily be spoken, such as emojis or bullet points. Be succinct and respond to what the user said in a creative and helpful way.",
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())]
            ),
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # STT
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            anam,  # Video Avatar (returns synchronised audio/video)
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
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Updating publishing settings to enable adaptive bitrate
        if isinstance(transport, DailyTransport):
            await transport.update_publishing(
                publishing_settings={
                    "camera": {
                        "sendSettings": {
                            "allowAdaptiveLayers": True,
                        }
                    }
                }
            )

        # Kick off the conversation.
        messages.append(
            {
                "role": "system",
                "content": "Start by saying 'Hello' and then a short greeting.",
            }
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
