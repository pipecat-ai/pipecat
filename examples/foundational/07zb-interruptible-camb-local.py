#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Camb.ai TTS example with local audio (microphone/speakers).

This example demonstrates:
- Camb.ai MARS TTS with streaming audio
- Local audio input/output (no WebRTC or Daily needed)
- TTFB metrics tracking
- End-to-end latency measurement (user speech → AI response)

Requirements:
- CAMB_API_KEY environment variable
- OPENAI_API_KEY environment variable (for LLM)
- DEEPGRAM_API_KEY environment variable (for STT)

Usage:
    python 07zb-interruptible-camb-local.py
    python 07zb-interruptible-camb-local.py --voice-id 147320
"""

import argparse
import asyncio
import os
import sys
import time

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    Frame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    TTSStartedFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.observers.loggers.metrics_log_observer import MetricsLogObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.camb.tts import CambTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams


class LatencyTracker(FrameProcessor):
    """Tracks end-to-end latency from user speech to AI audio response."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._user_stopped_time: float = 0
        self._llm_start_time: float = 0
        self._tts_start_time: float = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStoppedSpeakingFrame):
            self._user_stopped_time = time.time()
            logger.info("⏱️  User stopped speaking - timer started")

        elif isinstance(frame, LLMFullResponseStartFrame):
            self._llm_start_time = time.time()
            if self._user_stopped_time > 0:
                stt_latency = (self._llm_start_time - self._user_stopped_time) * 1000
                logger.info(f"⏱️  STT latency: {stt_latency:.0f}ms")

        elif isinstance(frame, TTSStartedFrame):
            self._tts_start_time = time.time()
            if self._llm_start_time > 0:
                llm_latency = (self._tts_start_time - self._llm_start_time) * 1000
                logger.info(f"⏱️  LLM TTFB: {llm_latency:.0f}ms")

        elif isinstance(frame, BotStartedSpeakingFrame):
            if self._user_stopped_time > 0:
                total_latency = (time.time() - self._user_stopped_time) * 1000
                tts_latency = (time.time() - self._tts_start_time) * 1000 if self._tts_start_time > 0 else 0
                logger.info(f"⏱️  TTS TTFB: {tts_latency:.0f}ms")
                logger.info(f"⏱️  ✨ TOTAL END-TO-END LATENCY: {total_latency:.0f}ms")
                # Reset for next turn
                self._user_stopped_time = 0
                self._llm_start_time = 0
                self._tts_start_time = 0

        await self.push_frame(frame, direction)

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Default voice
DEFAULT_VOICE_ID = 147320


async def main(voice_id: int):
    sample_rate = 48000

    # Local audio transport - uses your microphone and speakers
    # Increase audio_out_10ms_chunks for larger buffer (default is 4 = 40ms)
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_10ms_chunks=10,  # 100ms buffer for smoother playback
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        )
    )

    # Deepgram STT for speech recognition
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    # Camb.ai TTS (48kHz output)
    tts = CambTTSService(
        api_key=os.getenv("CAMB_API_KEY"),
        voice_id=voice_id,
        model="mars-flash",
    )

    # OpenAI LLM
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    # System prompt
    messages = [
        {
            "role": "system",
            "content": """You are a helpful voice assistant powered by Camb.ai
text-to-speech technology. Keep your responses concise and conversational since
they will be spoken aloud. Avoid special characters, emojis, or bullet points.""",
        },
    ]

    # Context management
    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    # Latency tracker for end-to-end timing
    latency_tracker = LatencyTracker()

    # Build the pipeline
    pipeline = Pipeline(
        [
            transport.input(),  # Microphone input
            stt,  # Speech-to-text
            latency_tracker,  # Track latency at various stages
            context_aggregator.user(),  # User context
            llm,  # Language model
            tts,  # TTS
            transport.output(),  # Speaker output
            context_aggregator.assistant(),  # Assistant context
        ]
    )

    # Create pipeline task with TTFB tracking
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_out_sample_rate=sample_rate,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[MetricsLogObserver(include_metrics={TTFBMetricsData})],
    )

    # Start the conversation when the pipeline is ready
    @task.event_handler("on_pipeline_started")
    async def on_pipeline_started(task, frame):
        messages.append(
            {
                "role": "system",
                "content": "Please introduce yourself briefly and ask how you can help.",
            }
        )
        await task.queue_frames([LLMRunFrame()])

    # Run the pipeline
    runner = PipelineRunner()
    logger.info("Starting Camb.ai TTS bot with local audio...")
    logger.info("Speak into your microphone to interact with the bot.")
    await runner.run(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camb.ai TTS with local audio")
    parser.add_argument(
        "--voice-id",
        type=int,
        default=DEFAULT_VOICE_ID,
        help=f"Camb.ai voice ID (default: {DEFAULT_VOICE_ID})",
    )
    args = parser.parse_args()
    asyncio.run(main(args.voice_id))
