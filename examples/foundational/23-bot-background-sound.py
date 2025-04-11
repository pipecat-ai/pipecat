#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""
Usage
-----
Set the path to your background audio file using the `INPUT_AUDIO_PATH` environment variable, then run the bot using:

    INPUT_AUDIO_PATH=path/to/your_audio.mp3 python 23-bot-background-sound.py

Example:

    INPUT_AUDIO_PATH=my_audio.mp3 python 23-bot-background-sound.py
"""

import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.mixers.soundfile_mixer import SoundfileMixer
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import MixerEnableFrame, MixerUpdateSettingsFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import SmallWebRTCConnection

load_dotenv(override=True)

audio_path = os.getenv("INPUT_AUDIO_PATH")
if not audio_path:
    raise ValueError("No INPUT_AUDIO_PATH specified in environment variables")


async def run_bot(webrtc_connection: SmallWebRTCConnection):
    logger.info(f"Starting bot")

    soundfile_mixer = SoundfileMixer(
        sound_files={"office": audio_path},
        default_sound="office",
        volume=2.0,
    )

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_mixer=soundfile_mixer,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
        ),
    )

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # STT service
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
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
            report_only_initial_ttfb=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected: {client}")
        # Show how to use mixer control frames.
        await asyncio.sleep(10.0)
        await task.queue_frame(MixerUpdateSettingsFrame({"volume": 0.5}))
        await asyncio.sleep(5.0)
        await task.queue_frame(MixerEnableFrame(False))
        await asyncio.sleep(5.0)
        await task.queue_frame(MixerEnableFrame(True))
        await asyncio.sleep(5.0)
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")

    @transport.event_handler("on_client_closed")
    async def on_client_closed(transport, client):
        logger.info(f"Client closed connection")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)


if __name__ == "__main__":
    from run import main

    main()
