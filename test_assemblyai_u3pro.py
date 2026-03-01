#!/usr/bin/env python3
"""AssemblyAI u3-rt-pro Comprehensive Test Script

Tests all features:
- Basic configuration
- Prompting and keyterms
- Diarization
- Dynamic updates
- Turn detection modes

Usage:
    python test_assemblyai_u3pro.py --test <test_name>
    python test_assemblyai_u3pro.py --interactive
"""

import argparse
import asyncio
import os
import sys
from typing import List

from dotenv import load_dotenv
from loguru import logger

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    LLMRunFrame,
    STTUpdateSettingsFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.assemblyai.models import AssemblyAIConnectionParams
from pipecat.services.assemblyai.stt import AssemblyAISTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

load_dotenv()


# Test configuration
class TestConfig:
    """Centralized test configuration."""

    ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")

    @classmethod
    def validate(cls):
        """Validate all required API keys are set."""
        missing = []
        if not cls.ASSEMBLYAI_API_KEY:
            missing.append("ASSEMBLYAI_API_KEY")
        if not cls.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        if not cls.CARTESIA_API_KEY:
            missing.append("CARTESIA_API_KEY")

        if missing:
            logger.error(f"Missing required environment variables: {', '.join(missing)}")
            return False
        return True


class TranscriptionLogger(FrameProcessor):
    """Log transcriptions for test verification."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, TranscriptionFrame):
            logger.info(f"📝 TRANSCRIPTION: {frame.text}")
            logger.info(f"   Speaker: {frame.user_id}")
            logger.info(f"   Finalized: {frame.finalized}")
            if hasattr(frame, "result") and frame.result:
                if hasattr(frame.result, "speaker"):
                    logger.info(f"   Diarization: {frame.result.speaker}")

        await self.push_frame(frame, direction)


async def create_basic_voice_agent(
    connection_params: AssemblyAIConnectionParams,
    vad_force_turn_endpoint: bool = True,
    speaker_format: str = None,
) -> tuple[PipelineTask, LocalAudioTransport]:
    """Create a basic voice agent for testing.

    Args:
        connection_params: AssemblyAI connection parameters
        vad_force_turn_endpoint: Turn detection mode
        speaker_format: Optional speaker formatting string

    Returns:
        Tuple of (PipelineTask, LocalAudioTransport)
    """
    # Create local audio transport (uses your microphone and speakers)
    transport = LocalAudioTransport(
        params=LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )

    # Create STT
    stt = AssemblyAISTTService(
        api_key=TestConfig.ASSEMBLYAI_API_KEY,
        connection_params=connection_params,
        vad_force_turn_endpoint=vad_force_turn_endpoint,
        speaker_format=speaker_format,
    )

    # Create TTS
    tts = CartesiaTTSService(
        api_key=TestConfig.CARTESIA_API_KEY,
        voice_id="a0e99841-438c-4a64-b679-ae501e7d6091",  # Conversational English
    )

    # Create LLM context and service
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful voice assistant. Keep responses brief and natural. "
                "If you see speaker tags like <Speaker A>text</Speaker A>, acknowledge "
                "that you understand multiple speakers are present."
            ),
        }
    ]

    context = LLMContext(messages)
    llm = OpenAILLMService(api_key=TestConfig.OPENAI_API_KEY, model="gpt-4")

    # Create aggregators with VAD
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    # Create transcription logger
    transcription_logger = TranscriptionLogger()

    # Create pipeline
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            transcription_logger,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    # Create task
    task = PipelineTask(pipeline)

    return task, transport


# ============================================================================
# Test Functions
# ============================================================================


async def test_basic_config():
    """Test 1: Basic default configuration."""
    logger.info("=" * 80)
    logger.info("TEST 1: Basic Default Configuration")
    logger.info("=" * 80)

    connection_params = AssemblyAIConnectionParams(speech_model="u3-rt-pro")

    task, transport = await create_basic_voice_agent(connection_params)

    logger.info("✅ Service created successfully with default params")
    logger.info("Expected: min=max=100ms, u3-rt-pro model")
    logger.info("Speak into your microphone to test transcription")

    # Trigger initial bot greeting
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner()
    await runner.run(task)


async def test_custom_min_silence():
    """Test 2: Custom min_turn_silence."""
    logger.info("=" * 80)
    logger.info("TEST 2: Custom min_turn_silence")
    logger.info("=" * 80)

    connection_params = AssemblyAIConnectionParams(speech_model="u3-rt-pro", min_turn_silence=200)

    task, transport = await create_basic_voice_agent(connection_params)

    logger.info("✅ Service created with min=200ms")
    logger.info("Expected: Both min and max set to 200ms")
    logger.info("Speak short phrases and observe turn detection timing")

    runner = PipelineRunner()
    await runner.run(task)


async def test_max_silence_warning():
    """Test 3: Setting max_turn_silence should trigger warning."""
    logger.info("=" * 80)
    logger.info("TEST 3: max_turn_silence Warning")
    logger.info("=" * 80)

    connection_params = AssemblyAIConnectionParams(
        speech_model="u3-rt-pro",
        min_turn_silence=100,
        max_turn_silence=500,  # Should trigger warning
    )

    task, transport = await create_basic_voice_agent(connection_params)

    logger.info("⚠️  Check logs above for warning about max_turn_silence being overridden")
    logger.info("Expected: Warning logged, max set to 100ms (same as min)")

    runner = PipelineRunner()
    await runner.run(task)


async def test_custom_prompt_warning():
    """Test 5: Custom prompt should trigger warning."""
    logger.info("=" * 80)
    logger.info("TEST 5: Custom Prompt Warning")
    logger.info("=" * 80)

    connection_params = AssemblyAIConnectionParams(
        speech_model="u3-rt-pro",
        prompt="Transcribe verbatim. Always include punctuation.",
    )

    task, transport = await create_basic_voice_agent(connection_params)

    logger.info("⚠️  Check logs above for warning about testing without prompt first")
    logger.info("Expected: Warning logged, service continues with custom prompt")

    runner = PipelineRunner()
    await runner.run(task)


async def test_prompt_keyterms_conflict():
    """Test 6: Prompt + keyterms_prompt should raise error."""
    logger.info("=" * 80)
    logger.info("TEST 6: Prompt + Keyterms Conflict (Error)")
    logger.info("=" * 80)

    try:
        connection_params = AssemblyAIConnectionParams(
            speech_model="u3-rt-pro",
            prompt="Custom prompt",
            keyterms_prompt=["test", "words"],
        )

        task, transport = await create_basic_voice_agent(connection_params)
        logger.error("❌ TEST FAILED: Should have raised ValueError")
    except ValueError as e:
        logger.info(f"✅ TEST PASSED: ValueError raised as expected")
        logger.info(f"   Error message: {e}")


async def test_keyterms_basic():
    """Test 7: Basic keyterms at initialization."""
    logger.info("=" * 80)
    logger.info("TEST 7: Basic Keyterms Prompting")
    logger.info("=" * 80)

    connection_params = AssemblyAIConnectionParams(
        speech_model="u3-rt-pro",
        keyterms_prompt=["Pipecat", "AssemblyAI", "Universal-3", "streaming"],
    )

    task, transport = await create_basic_voice_agent(connection_params)

    logger.info("✅ Service created with keyterms: Pipecat, AssemblyAI, Universal-3, streaming")
    logger.info("Expected: Boosted recognition for these terms")
    logger.info("Try saying: 'I'm testing Pipecat with AssemblyAI Universal-3 for streaming'")

    runner = PipelineRunner()
    await runner.run(task)


async def test_diarization_no_format():
    """Test 10: Diarization enabled without formatting."""
    logger.info("=" * 80)
    logger.info("TEST 10: Diarization Enabled (No Formatting)")
    logger.info("=" * 80)

    connection_params = AssemblyAIConnectionParams(speech_model="u3-rt-pro", speaker_labels=True)

    task, transport = await create_basic_voice_agent(connection_params)

    logger.info("✅ Service created with speaker_labels=True")
    logger.info("Expected: Speaker IDs in user_id field, plain text in transcript")
    logger.info("Have multiple people speak to see different speaker labels")

    runner = PipelineRunner()
    await runner.run(task)


async def test_diarization_xml_format():
    """Test 11: Diarization with XML formatting."""
    logger.info("=" * 80)
    logger.info("TEST 11: Diarization with XML Formatting")
    logger.info("=" * 80)

    connection_params = AssemblyAIConnectionParams(speech_model="u3-rt-pro", speaker_labels=True)

    task, transport = await create_basic_voice_agent(
        connection_params, speaker_format="<{speaker}>{text}</{speaker}>"
    )

    logger.info("✅ Service created with XML speaker formatting")
    logger.info("Expected: Text like '<Speaker A>Hello</Speaker A>'")
    logger.info("Have multiple people speak to see formatted speaker tags")

    runner = PipelineRunner()
    await runner.run(task)


async def test_dynamic_keyterms():
    """Test 13: Dynamic keyterms updates."""
    logger.info("=" * 80)
    logger.info("TEST 13: Dynamic Keyterms Updates")
    logger.info("=" * 80)

    connection_params = AssemblyAIConnectionParams(speech_model="u3-rt-pro")

    task, transport = await create_basic_voice_agent(connection_params)

    async def update_keyterms_stages():
        """Simulate multi-stage conversation with keyterms updates."""
        await asyncio.sleep(5)  # Wait for connection

        # Stage 1: Greeting
        logger.info("🔄 STAGE 1: Greeting (general terms)")
        update1 = STTUpdateSettingsFrame(
            settings={"keyterms_prompt": ["hello", "hi", "good morning", "welcome"]}
        )
        await task.queue_frames([update1])

        await asyncio.sleep(10)

        # Stage 2: Name collection
        logger.info("🔄 STAGE 2: Name Collection")
        update2 = STTUpdateSettingsFrame(
            settings={
                "keyterms_prompt": [
                    "first name",
                    "last name",
                    "John",
                    "Jane",
                    "Smith",
                    "Johnson",
                ]
            }
        )
        await task.queue_frames([update2])

        await asyncio.sleep(10)

        # Stage 3: Medical info
        logger.info("🔄 STAGE 3: Medical Information")
        update3 = STTUpdateSettingsFrame(
            settings={
                "keyterms_prompt": [
                    "cardiology",
                    "echocardiogram",
                    "blood pressure",
                    "Dr. Smith",
                    "metoprolol",
                ]
            }
        )
        await task.queue_frames([update3])

        await asyncio.sleep(10)

        # Stage 4: Clear keyterms
        logger.info("🔄 STAGE 4: Clear Keyterms")
        update4 = STTUpdateSettingsFrame(settings={"keyterms_prompt": []})
        await task.queue_frames([update4])

    # Start update task
    asyncio.create_task(update_keyterms_stages())

    logger.info("✅ Service created, will update keyterms every 10 seconds")
    logger.info("Expected: Different keyterms at each stage")
    logger.info("Watch logs for 'STAGE X' messages and test relevant terms")

    runner = PipelineRunner()
    await runner.run(task)


async def test_dynamic_silence_params():
    """Test 15: Dynamic silence parameter updates."""
    logger.info("=" * 80)
    logger.info("TEST 15: Dynamic Silence Parameters")
    logger.info("=" * 80)

    connection_params = AssemblyAIConnectionParams(speech_model="u3-rt-pro")

    task, transport = await create_basic_voice_agent(connection_params)

    async def update_silence_params():
        """Update silence parameters for different scenarios."""
        await asyncio.sleep(5)

        # Normal conversation
        logger.info("🔄 PHASE 1: Normal conversation (default timing)")
        await asyncio.sleep(10)

        # Reading credit card
        logger.info("🔄 PHASE 2: Reading numbers (longer silence tolerance)")
        update1 = STTUpdateSettingsFrame(
            settings={
                "max_turn_silence": 5000,
                "min_turn_silence": 300,
            }
        )
        await task.queue_frames([update1])

        await asyncio.sleep(15)

        # Back to normal
        logger.info("🔄 PHASE 3: Back to normal conversation")
        update2 = STTUpdateSettingsFrame(
            settings={
                "max_turn_silence": 1200,
                "min_turn_silence": 100,
            }
        )
        await task.queue_frames([update2])

    asyncio.create_task(update_silence_params())

    logger.info("✅ Service will update silence parameters during conversation")
    logger.info("Expected: Longer pauses tolerated in Phase 2")
    logger.info("Try pausing between words to test")

    runner = PipelineRunner()
    await runner.run(task)


async def test_multi_param_update():
    """Test 17: Update multiple parameters at once."""
    logger.info("=" * 80)
    logger.info("TEST 17: Multiple Parameter Update")
    logger.info("=" * 80)

    connection_params = AssemblyAIConnectionParams(speech_model="u3-rt-pro")

    task, transport = await create_basic_voice_agent(connection_params)

    async def multi_update():
        await asyncio.sleep(5)

        logger.info("🔄 Updating multiple parameters together")
        update = STTUpdateSettingsFrame(
            settings={
                "keyterms_prompt": ["account", "routing", "number"],
                "max_turn_silence": 3000,
                "min_turn_silence": 200,
            }
        )
        await task.queue_frames([update])

        logger.info("✅ Check logs for single UpdateConfiguration message")

    asyncio.create_task(multi_update())

    logger.info("Expected: All params updated in single WebSocket message")

    runner = PipelineRunner()
    await runner.run(task)


# ============================================================================
# Main Test Runner
# ============================================================================


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Test AssemblyAI u3-rt-pro integration")
    parser.add_argument(
        "--test",
        type=str,
        default="basic",
        help="Test to run (basic, custom_min, max_warning, prompt_warning, "
        "prompt_keyterms_conflict, keyterms, diarization, diarization_xml, "
        "dynamic_keyterms, dynamic_silence, multi_param, all)",
    )
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")

    args = parser.parse_args()

    # Validate environment
    if not TestConfig.validate():
        logger.error("Please set all required environment variables in .env")
        sys.exit(1)

    # Test mapping
    tests = {
        "basic": test_basic_config,
        "custom_min": test_custom_min_silence,
        "max_warning": test_max_silence_warning,
        "prompt_warning": test_custom_prompt_warning,
        "prompt_keyterms_conflict": test_prompt_keyterms_conflict,
        "keyterms": test_keyterms_basic,
        "diarization": test_diarization_no_format,
        "diarization_xml": test_diarization_xml_format,
        "dynamic_keyterms": test_dynamic_keyterms,
        "dynamic_silence": test_dynamic_silence_params,
        "multi_param": test_multi_param_update,
    }

    if args.interactive:
        logger.info("Interactive mode - select test to run:")
        for i, (name, _) in enumerate(tests.items(), 1):
            logger.info(f"{i}. {name}")
        logger.info(f"{len(tests) + 1}. Run all tests")

        choice = input("\nEnter test number: ")
        try:
            choice_num = int(choice)
            if choice_num == len(tests) + 1:
                args.test = "all"
            else:
                args.test = list(tests.keys())[choice_num - 1]
        except (ValueError, IndexError):
            logger.error("Invalid choice")
            sys.exit(1)

    # Run test(s)
    if args.test == "all":
        logger.info("Running all tests sequentially...")
        for test_name, test_func in tests.items():
            try:
                asyncio.run(test_func())
            except KeyboardInterrupt:
                logger.info(f"Test '{test_name}' interrupted")
                break
            except Exception as e:
                logger.error(f"Test '{test_name}' failed: {e}")
    else:
        if args.test not in tests:
            logger.error(f"Unknown test: {args.test}")
            logger.info(f"Available tests: {', '.join(tests.keys())}")
            sys.exit(1)

        try:
            asyncio.run(tests[args.test]())
        except KeyboardInterrupt:
            logger.info("Test interrupted")
        except Exception as e:
            logger.error(f"Test failed: {e}")
            raise


if __name__ == "__main__":
    main()
