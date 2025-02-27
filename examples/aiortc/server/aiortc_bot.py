#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import asyncio
import os
import sys

import numpy as np
from aiortc import MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaRelay
from av import AudioFrame
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIProcessor, RTVIObserver
from pipecat.services.gemini_multimodal_live import GeminiMultimodalLiveLLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.webrtc.pipecat_webrtc import PipecatWebRTCTransport
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


async def run_bot(webrtc_connection):
    pipecat_transport = PipecatWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
        ),
    )

    llm = GeminiMultimodalLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        voice_id="Puck",  # Aoede, Charon, Fenrir, Kore, Puck
        transcribe_user_audio=True,
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
            pipecat_transport.input(),
            context_aggregator.user(),
            rtvi,
            llm,  # LLM
            pipecat_transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            observers=[RTVIObserver(rtvi)],
        ),
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        await rtvi.set_bot_ready()

    @pipecat_transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Pipecat Client connected")
        # Kick off the conversation.
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @pipecat_transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Pipecat Client disconnected")

    @pipecat_transport.event_handler("on_client_closed")
    async def on_client_closed(transport, client):
        logger.info("Pipecat Client closed")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)


# ---------------- EVERYTHING BELOW THIS IS ONLY FOR TESTING AIORTC DIRECTLY ------------------------------------------


class AudioBeepStreamTrack(MediaStreamTrack):
    """
    A custom MediaStreamTrack that generates a beep sound.
    """

    kind = "audio"

    def __init__(self, sample_rate=48000, frequency=440):
        super().__init__()
        self.sample_rate = sample_rate
        self.frequency = frequency
        self.samples_per_frame = self.sample_rate // 50  # 20ms per frame
        self.time = 0

    async def recv(self):
        """
        Generate a sine wave beep sound.
        """
        await asyncio.sleep(0.02)  # Simulate real-time audio (20ms frame)

        # Generate sine wave
        t = np.arange(self.samples_per_frame) + self.time
        samples = 0.5 * np.sin(2 * np.pi * self.frequency * t / self.sample_rate)

        # Convert float32 (-1 to 1 range) to int16 (-32768 to 32767 range)
        samples = (samples * 32767).astype(np.int16)

        # Create AudioFrame
        frame = AudioFrame(format="s16", layout="mono", samples=len(samples))
        frame.sample_rate = self.sample_rate  # Set sample rate

        self.time += self.samples_per_frame
        frame.pts = self.time  # Set timestamp (must be increasing)

        frame.planes[0].update(samples.tobytes())

        return frame


async def run_aiortc_bot(pipecat_connection: PipecatWebRTCConnection):
    relay = MediaRelay()
    recorder = MediaBlackhole()
    await recorder.start()

    def handle_track(track: MediaStreamTrack):
        if track.kind == "audio":
            pipecat_connection.replace_audio_track(AudioBeepStreamTrack())
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
