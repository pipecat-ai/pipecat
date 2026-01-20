#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import aiohttp
import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    StartFrame,
    SystemFrame,
    TextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.livekit_worker import LiveKitWorkerRunner
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.services.soniox.stt import SonioxSTTService, SonioxInputParams
from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport
from pipecat.turns.user_turn_strategies import UserTurnStrategies

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


class DebugFrameLogger(FrameProcessor):
    """A processor that logs all frames passing through for debugging."""

    def __init__(self, name: str = "DebugLogger"):
        super().__init__()
        self._name = name
        self._audio_frame_count = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        # Skip logging system frames to reduce noise unless they are special
        if isinstance(frame, SystemFrame) and not isinstance(frame, StartFrame):
            # Still push the frame, just don't log it
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, AudioRawFrame):
            self._audio_frame_count += 1
            # Log every 100 frames to confirm audio is flowing
            if self._audio_frame_count % 100 == 0:
                logger.info(f"[{self._name}] 🔊 Audio frames received: {self._audio_frame_count}")
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            logger.info(f"[{self._name}] 🎙️ VAD: USER STARTED SPEAKING")
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            logger.info(f"[{self._name}] 🔇 VAD: USER STOPPED SPEAKING")
        elif isinstance(frame, TranscriptionFrame):
            logger.info(f"[{self._name}] 🎤 TRANSCRIPTION: '{frame.text}'")
        elif isinstance(frame, TextFrame):
            logger.info(f"[{self._name}] 💬 TEXT: '{frame.text}'")
        elif isinstance(frame, TTSSpeakFrame):
            logger.info(f"[{self._name}] 🔊 TTS_SPEAK: '{frame.text}'")
        elif isinstance(frame, UserStartedSpeakingFrame):
            logger.info(f"[{self._name}] 🗣️ USER STARTED SPEAKING")
        elif isinstance(frame, UserStoppedSpeakingFrame):
            logger.info(f"[{self._name}] 🤐 USER STOPPED SPEAKING")
        else:
            logger.debug(f"[{self._name}] Frame: {type(frame).__name__}")

        await self.push_frame(frame, direction)


async def pipeline_factory(room):
    """
    Creates the pipeline for the agent using the connected LiveKit room.
    This function is called by LiveKitWorkerRunner when a job is assigned.
    """
    logger.info(f"🚀 Creating pipeline for room: {room.name}")

    session = aiohttp.ClientSession()

    # Inject the existing room into the transport
    transport = LiveKitTransport(
        url=os.getenv("LIVEKIT_URL", "ws://localhost:7880"),
        token="",
        room_name=room.name,
        room=room,
        params=LiveKitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,   # 16kHz for STT compatibility
            audio_out_sample_rate=24000,  # 24kHz for Sarvam TTS
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    stop_secs=0.5,
                )
            ),
        ),
    )
    logger.info("✅ LiveKitTransport created")

    stt = SonioxSTTService(
        api_key=os.getenv("SONIOX_API_KEY"),
        sample_rate=16000,  # Explicit 16kHz to match transport
        params=SonioxInputParams(
            language_hints=["en"],
        )
    )
    logger.info(f"✅ Soniox STT created (API key set: {bool(os.getenv('SONIOX_API_KEY'))})")

    llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile"  # Using verified working model
    )
    logger.info(f"✅ Groq LLM created (API key set: {bool(os.getenv('GROQ_API_KEY'))})")

    tts = SarvamTTSService(
        api_key=os.getenv("SARVAM_API_KEY"),
        voice_id="abhilash",
        model="bulbul:v2",
        aiohttp_session=session,
        aggregate_sentences=False,
    )
    logger.info(f"✅ Sarvam TTS created (API key set: {bool(os.getenv('SARVAM_API_KEY'))})")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant running as a LiveKit Agent. Respond politely and concisely.",
        },
    ]

    context = LLMContext(messages)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies()
        ),
    )
    logger.info("✅ LLM Context aggregators created")

    # Debug loggers for tracing
    pre_stt_logger = DebugFrameLogger("PRE-STT")
    post_stt_logger = DebugFrameLogger("POST-STT")
    post_llm_logger = DebugFrameLogger("POST-LLM")
    post_tts_logger = DebugFrameLogger("POST-TTS")

    pipeline = Pipeline(
        [
            transport.input(),
            pre_stt_logger,
            stt,
            post_stt_logger,
            user_aggregator,
            llm,
            post_llm_logger,
            tts,
            post_tts_logger,
            transport.output(),
            assistant_aggregator,
        ]
    )
    logger.info("✅ Pipeline created with debug loggers")

    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True))

    # Make sure we close the session when the pipeline is done
    @transport.event_handler("on_disconnected")
    async def on_disconnected(transport):
        logger.info("👋 Participant disconnected, closing session")
        await session.close()

    # Trigger a greeting when the first participant joins
    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant_id):
        logger.info(f"🎉 First participant joined: {participant_id}")
        await asyncio.sleep(1)
        logger.info("📤 Sending greeting TTSSpeakFrame...")
        await task.queue_frame(
            TTSSpeakFrame("Hello! I'm a LiveKit Agent powered by Pipecat. How can I help you today?")
        )
        logger.info("✅ Greeting frame queued")

    # Log when audio track is subscribed
    @transport.event_handler("on_participant_joined")
    async def on_participant_joined(transport, participant_id):
        logger.info(f"👤 Participant joined: {participant_id}")

    return transport, task


def main():
    # Initialize the Worker Runner
    runner = LiveKitWorkerRunner()

    # Register the factory that builds the pipeline per room
    runner.set_pipeline_factory(pipeline_factory)

    # Print a helpful token for the user to join
    try:
        from pipecat.runner.livekit import generate_token
        
        api_key = os.getenv("LIVEKIT_API_KEY", "devkey")
        api_secret = os.getenv("LIVEKIT_API_SECRET", "secret")
        
        token = generate_token(
            "groq-room", 
            "GroqUser", 
            api_key, 
            api_secret
        )
        
        logger.info("\n" + "="*60)
        logger.info("🤖 LiveKit Agent Bot Started")
        logger.info("="*60)
        logger.info(f"Join the room 'groq-room' using this token:\n\n{token}\n")
        logger.info(f"Browser URL: https://agents-playground.livekit.io/ (Set URL to ws://localhost:7880)")
        logger.info("="*60 + "\n")
        
    except Exception as e:
        logger.warning(f"Could not generate startup token: {e}")

    # Start the worker (blocks forever listening for jobs)
    runner.run()


if __name__ == "__main__":
    main()
