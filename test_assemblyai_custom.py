#!/usr/bin/env python3
"""Custom AssemblyAI u3-rt-pro Test Script
Easy parameter tweaking for experimentation

Edit the CONFIGURATION section below to test different settings!
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.assemblyai.models import AssemblyAIConnectionParams
from pipecat.services.assemblyai.stt import AssemblyAISTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

load_dotenv(override=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Log Level: "DEBUG" for detailed logs, "INFO" for normal operation
LOG_LEVEL = "INFO"

# ============================================================================
# BOT IMPLEMENTATION
# ============================================================================


async def main():
    """Run the custom test bot with your configured parameters."""
    # Setup logging
    logger.remove(0)
    logger.add(sys.stderr, level=LOG_LEVEL)

    logger.info("="*80)
    logger.info("AssemblyAI u3-rt-pro Custom Test")
    logger.info("="*80)
    logger.info("Starting bot... Speak after you hear the greeting!")
    logger.info("="*80)

    # Create local audio transport
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )

    # ========================================================================
    # EDIT PARAMETERS HERE
    # ========================================================================

    # Build connection params
    connection_params = AssemblyAIConnectionParams(
        # ====================================================================
        # Model Selection
        # ====================================================================
        speech_model="u3-rt-pro",
        # speech_model="universal-streaming-english",
        # speech_model="universal-streaming-multilingual",

        # ====================================================================
        # Turn Detection Timing
        # ====================================================================

        # Minimum silence when confident about end of turn (milliseconds)
        # Default: 100ms | Higher = more patient | Lower = faster responses
        # Only used in Pipecat mode (vad_force_turn_endpoint=True)
        min_turn_silence=100000,
        # min_turn_silence=200,
        # min_turn_silence=300,

        # Maximum turn silence (milliseconds)
        # WARNING: In Pipecat mode (vad_force_turn_endpoint=True), this is
        # automatically set equal to min_turn_silence
        # to avoid double turn detection. Only used as-is in STT mode.
        max_turn_silence=500,

        # End of turn confidence threshold (0.0 to 1.0)
        # Higher = requires more confidence before ending turn
        # end_of_turn_confidence_threshold=0.8,

        # ====================================================================
        # Prompting & Boosting
        # ====================================================================

        # Custom Prompt (WARNING: test carefully, default is optimized!)
        # None = Use AssemblyAI's optimized default (recommended for 88% accuracy)
        prompt=None,
        # prompt="Transcribe speech with focus on technical terms.",
        # prompt="Context: Medical conversation. Transcribe accurately.",

        # Keyterms Prompting (boosts recognition for specific words)
        # NOTE: Cannot use both prompt and keyterms_prompt!
        keyterms_prompt=None,
        # keyterms_prompt=["Pipecat", "AssemblyAI", "OpenAI", "Cartesia"],
        # keyterms_prompt=["Python", "JavaScript", "TypeScript", "API"],

        # ====================================================================
        # Diarization (Speaker Identification)
        # ====================================================================

        # Enable speaker labels (identifies different speakers)
        speaker_labels=None,  # None or True
        # speaker_labels=True,

        # ====================================================================
        # Audio Configuration
        # ====================================================================

        # Audio sample rate (Hz)
        # sample_rate=16000,
        # sample_rate=8000,

        # Audio encoding format
        # encoding="pcm_s16le",  # Default: 16-bit PCM
        # encoding="pcm_mulaw",  # μ-law encoding (telephony)

        # ====================================================================
        # Other Options
        # ====================================================================

        # Format transcript turns (applies formatting rules)
        # format_turns=True,  # Default
        # format_turns=False,

        # Language detection (only for universal-streaming-multilingual)
        # language_detection=True,
    )

    # Log connection parameters for debugging
    logger.info("="*80)
    logger.info("CONNECTION PARAMETERS:")
    logger.info(f"  speech_model: {connection_params.speech_model}")
    logger.info(f"  min_turn_silence: {connection_params.min_turn_silence}")
    logger.info(f"  max_turn_silence: {connection_params.max_turn_silence}")
    logger.info(f"  sample_rate: {connection_params.sample_rate}")
    logger.info(f"  encoding: {connection_params.encoding}")
    logger.info(f"  prompt: {connection_params.prompt}")
    logger.info(f"  keyterms_prompt: {connection_params.keyterms_prompt}")
    logger.info(f"  speaker_labels: {connection_params.speaker_labels}")
    logger.info(f"  format_turns: {connection_params.format_turns}")
    logger.info(f"  end_of_turn_confidence_threshold: {connection_params.end_of_turn_confidence_threshold}")
    logger.info(f"  language_detection: {connection_params.language_detection}")
    logger.info("="*80)

    # AssemblyAI Speech-to-Text Service
    stt = AssemblyAISTTService(
        api_key=os.getenv("ASSEMBLYAI_API_KEY"),
        connection_params=connection_params,

        # Turn Detection Mode
        # True = Pipecat mode (VAD + Smart Turn controls turns)
        # False = STT mode (u3-rt-pro model controls turns)
        vad_force_turn_endpoint=True,

        # Speaker Formatting (only used if speaker_labels=True)
        # None = Just log speaker IDs, don't modify transcript
        speaker_format=None,
        # speaker_format="<Speaker {speaker}>{text}</Speaker {speaker}>",
        # speaker_format="{speaker}: {text}",
        # speaker_format="[{speaker}] {text}",

        # Additional available parameters (uncomment to use):
        # should_interrupt=True,  # Only for STT mode
    )

    # ========================================================================

    # Text-to-Speech
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="a0e99841-438c-4a64-b679-ae501e7d6091",  # Conversational English
    )

    # LLM
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4",
    )

    # Conversation context
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful voice assistant testing the AssemblyAI u3-rt-pro model. "
                "Keep responses very brief (1-2 sentences). "
                "Start by introducing yourself briefly and asking the user to speak."
            ),
        },
    ]

    context = LLMContext(messages)

    # Configure aggregator based on mode
    # In STT mode, don't use VAD (model handles turn detection)
    # In Pipecat mode, use VAD + Smart Turn
    vad_force_turn_endpoint = True  # Must match the value in stt configuration above
    user_params = None
    if vad_force_turn_endpoint:
        user_params = LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=user_params,
    )

    # Pipeline
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    # Task
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # Start the conversation
    await task.queue_frames([LLMRunFrame()])

    # Run
    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
