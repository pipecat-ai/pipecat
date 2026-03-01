#!/usr/bin/env python3
"""Interactive AssemblyAI u3-rt-pro Comprehensive Test Suite

Tests all features with detailed scenarios:
- Basic configuration variations
- Prompting and keyterms with difficult names
- Diarization
- Dynamic parameter updates (single and multiple)
- Mode comparisons
- STT mode timing experiments (testing silence parameters)
- Edge cases

Usage:
    python test_assemblyai_interactive.py
"""

import asyncio
import os
import sys
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame, STTUpdateSettingsFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.assemblyai.models import AssemblyAIConnectionParams
from pipecat.services.assemblyai.stt import AssemblyAISTTService, AssemblyAISTTSettings
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="INFO")


async def run_bot(
    connection_params: AssemblyAIConnectionParams,
    test_name: str,
    vad_force_turn_endpoint: bool = True,
    speaker_format: Optional[str] = None,
    test_dynamic_updates: Optional[callable] = None,
):
    """Run the voice bot with specified configuration."""
    logger.info("=" * 80)
    logger.info(f"TEST: {test_name}")
    logger.info("=" * 80)
    logger.info("Starting bot... Speak into your microphone after you hear the greeting!")
    logger.info("=" * 80)

    # Create local audio transport
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )

    # AssemblyAI Speech-to-Text
    stt = AssemblyAISTTService(
        api_key=os.getenv("ASSEMBLYAI_API_KEY"),
        connection_params=connection_params,
        vad_force_turn_endpoint=vad_force_turn_endpoint,
        speaker_format=speaker_format,
    )

    # Text-to-Speech
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="a0e99841-438c-4a64-b679-ae501e7d6091",
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

    # Handle dynamic updates if provided
    if test_dynamic_updates:
        asyncio.create_task(test_dynamic_updates(task))

    # Start the conversation
    await task.queue_frames([LLMRunFrame()])

    # Run
    runner = PipelineRunner()
    await runner.run(task)


# ============================================================================
# Test Configurations
# ============================================================================

# === BASIC CONFIGURATION (1-3) ===


async def test_01_basic_100ms():
    """Test 1: Basic default configuration (100ms)."""
    connection_params = AssemblyAIConnectionParams(
        speech_model="u3-rt-pro",
        min_turn_silence=100,
    )
    await run_bot(connection_params, "Basic Default Configuration (100ms)")


async def test_02_custom_200ms():
    """Test 2: Custom min_end_of_turn_silence (200ms)."""
    connection_params = AssemblyAIConnectionParams(
        speech_model="u3-rt-pro",
        min_turn_silence=200,
    )
    await run_bot(connection_params, "Custom Turn Silence (200ms)")


async def test_03_custom_500ms():
    """Test 3: Longer silence threshold (500ms)."""
    connection_params = AssemblyAIConnectionParams(
        speech_model="u3-rt-pro",
        min_turn_silence=500,
    )
    await run_bot(connection_params, "Longer Turn Silence (500ms)")


# === PROMPTING & WARNINGS (4-7) ===


async def test_04_max_warning():
    """Test 4: max_turn_silence warning (should be overridden)."""
    logger.warning("⚠️ EXPECT WARNING: max_turn_silence will be overridden")
    connection_params = AssemblyAIConnectionParams(
        speech_model="u3-rt-pro",
        max_turn_silence=500,
    )
    await run_bot(connection_params, "max_turn_silence Override Warning")


async def test_05_prompt_warning():
    """Test 5: Custom prompt warning."""
    logger.warning("⚠️ EXPECT WARNING: Custom prompts should be tested carefully")
    connection_params = AssemblyAIConnectionParams(
        speech_model="u3-rt-pro",
        prompt="Transcribe speech accurately with proper punctuation.",
    )
    await run_bot(connection_params, "Custom Prompt Warning Test")


async def test_06_prompt_keyterms_conflict():
    """Test 6: Prompt + keyterms conflict (should error)."""
    logger.error("❌ EXPECT ERROR: Cannot use both prompt and keyterms_prompt")
    try:
        connection_params = AssemblyAIConnectionParams(
            speech_model="u3-rt-pro",
            prompt="Custom prompt",
            keyterms_prompt=["test"],
        )
        await run_bot(connection_params, "Prompt + Keyterms Conflict (ERROR)")
    except ValueError as e:
        logger.error(f"✅ EXPECTED ERROR: {e}")
        input("\nPress Enter to continue...")
        return


async def test_07_keyterms_difficult():
    """Test 7: Keyterms with difficult/unusual names."""
    # Use names that STT wouldn't normally get right
    keyterms = ["Xiomara", "Saoirse", "Krzystof", "Nguyen", "Pipecat", "AssemblyAI"]
    connection_params = AssemblyAIConnectionParams(
        speech_model="u3-rt-pro",
        keyterms_prompt=keyterms,
    )
    logger.info("🎯 Boosted terms: Xiomara, Saoirse, Krzystof, Nguyen, Pipecat, AssemblyAI")
    logger.info("   Try saying these difficult names to test boosting!")
    await run_bot(connection_params, "Keyterms with Difficult Names")


# === DIARIZATION (8-9) ===


async def test_08_diarization_basic():
    """Test 8: Basic diarization (speaker IDs logged)."""
    connection_params = AssemblyAIConnectionParams(
        speech_model="u3-rt-pro",
        speaker_labels=True,
    )
    logger.info("🎤 Diarization enabled - speaker IDs will be logged")
    logger.info("   Try having multiple people speak!")
    await run_bot(connection_params, "Diarization - Basic")


async def test_09_diarization_xml():
    """Test 9: Diarization with XML formatting."""
    connection_params = AssemblyAIConnectionParams(
        speech_model="u3-rt-pro",
        speaker_labels=True,
    )
    logger.info("🎤 Diarization with XML tags")
    logger.info("   Transcripts will include <Speaker X>text</Speaker X>")
    await run_bot(
        connection_params,
        "Diarization - XML Formatting",
        speaker_format="<Speaker {speaker}>{text}</Speaker {speaker}>",
    )


# === DYNAMIC UPDATES - SINGLE PARAMETER (10-13) ===


async def test_10_dynamic_keyterms():
    """Test 10: Dynamic keyterms update with difficult names."""
    connection_params = AssemblyAIConnectionParams(
        speech_model="u3-rt-pro",
    )

    async def dynamic_update(task):
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 1: No keyterms boosting")
        logger.info("  Try saying: Xiomara, Saoirse, Krzystof")
        logger.info("  (May not transcribe correctly)")
        logger.info("=" * 80)
        await asyncio.sleep(15)

        logger.info("\n" + "=" * 80)
        logger.info("🔄 UPDATING: Adding keyterms boost")
        logger.info("=" * 80)
        await task.queue_frame(
            STTUpdateSettingsFrame(
                delta=AssemblyAISTTSettings(
                    connection_params=AssemblyAIConnectionParams(
                        keyterms_prompt=["Xiomara", "Saoirse", "Krzystof", "Nguyen"]
                    )
                )
            )
        )
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2: Keyterms NOW boosted")
        logger.info("  Say the same names again: Xiomara, Saoirse, Krzystof")
        logger.info("  (Should transcribe better now!)")
        logger.info("=" * 80)

    logger.info("🔄 This test has 2 phases:")
    logger.info("   Phase 1 (15s): No boosting - names may be wrong")
    logger.info("   Phase 2: Keyterms added - names should improve")
    await run_bot(
        connection_params,
        "Dynamic Keyterms Update (Before/After)",
        test_dynamic_updates=dynamic_update,
    )


async def test_11_dynamic_silence():
    """Test 11: Dynamic silence parameter update (dramatic change)."""
    connection_params = AssemblyAIConnectionParams(
        speech_model="u3-rt-pro",
        min_turn_silence=100,
    )

    async def dynamic_update(task):
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 1: Quick responses (100ms silence threshold)")
        logger.info("  Speak normally - bot responds quickly")
        logger.info("=" * 80)
        await asyncio.sleep(10)

        logger.info("\n" + "=" * 80)
        logger.info("🔄 UPDATING: Changing silence from 100ms → 3000ms (3 seconds!)")
        logger.info("=" * 80)
        await task.queue_frame(
            STTUpdateSettingsFrame(
                delta=AssemblyAISTTSettings(
                    connection_params=AssemblyAIConnectionParams(min_turn_silence=3000)
                )
            )
        )
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2: Patient responses (3 second silence threshold)")
        logger.info("  Bot will wait 3 full seconds before responding")
        logger.info("  Try pausing mid-sentence - bot should NOT interrupt")
        logger.info("=" * 80)

    logger.info("🔄 Dramatic change: 100ms → 3000ms after 10 seconds")
    await run_bot(
        connection_params,
        "Dynamic Silence Update (100ms → 3s)",
        test_dynamic_updates=dynamic_update,
    )


async def test_12_dynamic_prompt():
    """Test 12: Dynamic prompt update with keyterms in prompt."""
    connection_params = AssemblyAIConnectionParams(
        speech_model="u3-rt-pro",
    )

    async def dynamic_update(task):
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 1: Default prompt (no keyterms)")
        logger.info("  Try saying: Xiomara, Saoirse, Krzystof")
        logger.info("  (May not transcribe correctly)")
        logger.info("=" * 80)
        await asyncio.sleep(15)

        logger.info("\n" + "=" * 80)
        logger.info("🔄 UPDATING: Adding custom prompt with keyterms")
        logger.info("=" * 80)
        custom_prompt = """Transcribe verbatim. Rules:
1) Always include punctuation in output.
2) Use period/question mark ONLY for complete sentences.
3) Use comma for mid-sentence pauses.
4) Use no punctuation for incomplete trailing speech.
5) Filler words (um, uh, so, like) indicate speaker will continue.

Pay special attention to these names and transcribe them exactly: Xiomara, Saoirse, Krzystof, Nguyen."""
        await task.queue_frame(
            STTUpdateSettingsFrame(
                delta=AssemblyAISTTSettings(
                    connection_params=AssemblyAIConnectionParams(prompt=custom_prompt)
                )
            )
        )
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2: Prompt with keyterms NOW active")
        logger.info("  Say the same names again: Xiomara, Saoirse, Krzystof")
        logger.info("  (Should transcribe better now!)")
        logger.info("=" * 80)

    logger.info("🔄 This test has 2 phases:")
    logger.info("   Phase 1 (15s): Default prompt - names may be wrong")
    logger.info("   Phase 2: Custom prompt with keyterms - names should improve")
    await run_bot(
        connection_params,
        "Dynamic Prompt Update (with keyterms)",
        test_dynamic_updates=dynamic_update,
    )


async def test_13_dynamic_clear_keyterms():
    """Test 13: Clear keyterms dynamically."""
    connection_params = AssemblyAIConnectionParams(
        speech_model="u3-rt-pro",
        keyterms_prompt=["Pipecat", "AssemblyAI"],
    )

    async def dynamic_update(task):
        await asyncio.sleep(10)
        logger.info("🔄 UPDATING: Clearing keyterms (empty array)")
        await task.queue_frame(
            STTUpdateSettingsFrame(
                delta=AssemblyAISTTSettings(
                    connection_params=AssemblyAIConnectionParams(keyterms_prompt=[])
                )
            )
        )

    logger.info("🎯 Initial: Pipecat, AssemblyAI boosted")
    logger.info("🔄 After 10s: Keyterms will be cleared")
    await run_bot(
        connection_params,
        "Dynamic Clear Keyterms",
        test_dynamic_updates=dynamic_update,
    )


# === DYNAMIC UPDATES - MULTIPLE PARAMETERS (14-15) ===


async def test_14_multi_param_update():
    """Test 14: Update multiple parameters at once."""
    connection_params = AssemblyAIConnectionParams(
        speech_model="u3-rt-pro",
        min_turn_silence=100,
    )

    async def dynamic_update(task):
        await asyncio.sleep(10)
        logger.info("🔄 UPDATING MULTIPLE: keyterms + silence")
        await task.queue_frame(
            STTUpdateSettingsFrame(
                delta=AssemblyAISTTSettings(
                    connection_params=AssemblyAIConnectionParams(
                        keyterms_prompt=["Xiomara", "Pipecat"],
                        min_turn_silence=250,
                    )
                )
            )
        )

    logger.info("🔄 After 10s: Will update BOTH keyterms AND silence threshold")
    await run_bot(
        connection_params,
        "Multiple Parameter Update",
        test_dynamic_updates=dynamic_update,
    )


async def test_15_complex_sequence():
    """Test 15: Complex multi-stage update sequence."""
    connection_params = AssemblyAIConnectionParams(
        speech_model="u3-rt-pro",
    )

    async def dynamic_update(task):
        logger.info("Stage 1: Initial (10s)")
        await asyncio.sleep(10)

        logger.info("🔄 Stage 2: Add keyterms")
        await task.queue_frame(
            STTUpdateSettingsFrame(
                delta=AssemblyAISTTSettings(
                    connection_params=AssemblyAIConnectionParams(keyterms_prompt=["Pipecat"])
                )
            )
        )
        await asyncio.sleep(10)

        logger.info("🔄 Stage 3: Change silence")
        await task.queue_frame(
            STTUpdateSettingsFrame(
                delta=AssemblyAISTTSettings(
                    connection_params=AssemblyAIConnectionParams(min_turn_silence=200)
                )
            )
        )
        await asyncio.sleep(10)

        logger.info("🔄 Stage 4: Update both")
        await task.queue_frame(
            STTUpdateSettingsFrame(
                delta=AssemblyAISTTSettings(
                    connection_params=AssemblyAIConnectionParams(
                        keyterms_prompt=["AssemblyAI", "OpenAI"],
                        min_turn_silence=150,
                    )
                )
            )
        )

    logger.info("🔄 Multi-stage: 4 configuration changes over 30 seconds")
    await run_bot(
        connection_params,
        "Complex Update Sequence (4 stages)",
        test_dynamic_updates=dynamic_update,
    )


# === MODE COMPARISON (16-17) ===


async def test_16_pipecat_mode():
    """Test 16: Pipecat mode (VAD + Smart Turn controls turns)."""
    connection_params = AssemblyAIConnectionParams(
        speech_model="u3-rt-pro",
        min_turn_silence=100,
    )
    logger.info("🎯 Pipecat Mode: VAD + Smart Turn control turn detection")
    logger.info("   Your min_end_of_turn_silence is sent but ForceEndpoint overrides it")
    await run_bot(
        connection_params,
        "Pipecat Mode (VAD + Smart Turn)",
        vad_force_turn_endpoint=True,
    )


async def test_17_stt_mode():
    """Test 17: STT mode (model controls turns)."""
    connection_params = AssemblyAIConnectionParams(
        speech_model="u3-rt-pro",
        min_turn_silence=100,
    )
    logger.info("🎯 STT Mode: u3-rt-pro model controls turn detection")
    logger.info("   No ForceEndpoint - parameters are respected")
    await run_bot(
        connection_params,
        "STT Mode (Model Turn Detection)",
        vad_force_turn_endpoint=False,
    )


# === STT MODE TIMING EXPERIMENTS (18-20) ===


async def test_18_stt_long_max_short_min():
    """Test 18: STT mode - Long max_turn_silence + Short min (5000ms + 100ms)."""
    connection_params = AssemblyAIConnectionParams(
        speech_model="u3-rt-pro",
        min_turn_silence=100,  # Short - quick confident turns
        max_turn_silence=5000,  # Long - allows pauses up to 5 seconds
    )
    logger.info("🎯 STT Mode: Testing max/min parameter interaction")
    logger.info("   min_turn_silence: 100ms (quick when confident)")
    logger.info("   max_turn_silence: 5000ms (allows up to 5 second pauses)")
    logger.info("   Try: Quick sentences (should respond fast) + Long pauses mid-thought")
    await run_bot(
        connection_params,
        "STT: Long Max (5s) + Short Min (100ms)",
        vad_force_turn_endpoint=False,
    )


async def test_19_stt_long_min():
    """Test 19: STT mode - Long min_turn_silence (3000ms)."""
    connection_params = AssemblyAIConnectionParams(
        speech_model="u3-rt-pro",
        min_turn_silence=3000,  # 3 seconds
        max_turn_silence=5000,  # 5 seconds
    )
    logger.info("🎯 STT Mode: Testing long minimum silence requirement")
    logger.info("   min_turn_silence: 3000ms")
    logger.info("   max_turn_silence: 5000ms")
    logger.info("   Bot will wait 3 full seconds of silence before responding!")
    logger.info("   Try: Speaking with short pauses - bot should NOT interrupt")
    await run_bot(
        connection_params,
        "STT: Long Min (3s)",
        vad_force_turn_endpoint=False,
    )


async def test_20_stt_both_short():
    """Test 20: STT mode - Both short (max=300ms, min=100ms)."""
    connection_params = AssemblyAIConnectionParams(
        speech_model="u3-rt-pro",
        min_turn_silence=100,  # 100ms
        max_turn_silence=300,  # 300ms
    )
    logger.info("🎯 STT Mode: Testing aggressive/quick response timing")
    logger.info("   min_turn_silence: 100ms")
    logger.info("   max_turn_silence: 300ms")
    logger.info("   Bot will respond VERY quickly to any pause!")
    logger.info("   Try: Speaking with natural pauses - expect quick responses")
    await run_bot(
        connection_params,
        "STT: Both Short (300ms/100ms)",
        vad_force_turn_endpoint=False,
    )


# === EDGE CASES (21-23) ===


async def test_21_very_long_silence():
    """Test 21: Very long silence threshold (STT mode only)."""
    connection_params = AssemblyAIConnectionParams(
        speech_model="u3-rt-pro",
        min_turn_silence=10000,  # 10 seconds
    )
    logger.warning("⚠️ STT Mode with 10 second silence threshold")
    logger.info("   Bot will wait 10 seconds of silence before responding!")
    await run_bot(
        connection_params,
        "Very Long Silence (10s) - STT Mode",
        vad_force_turn_endpoint=False,
    )


async def test_22_very_short_silence():
    """Test 22: Very short silence threshold (50ms)."""
    connection_params = AssemblyAIConnectionParams(
        speech_model="u3-rt-pro",
        min_turn_silence=50,
    )
    logger.info("⚡ Very short silence threshold (50ms)")
    logger.info("   Bot will respond very quickly!")
    await run_bot(connection_params, "Very Short Silence (50ms)")


async def test_23_keyterms_plus_diarization():
    """Test 23: Keyterms + Diarization combined."""
    connection_params = AssemblyAIConnectionParams(
        speech_model="u3-rt-pro",
        keyterms_prompt=["Xiomara", "Saoirse", "Pipecat"],
        speaker_labels=True,
    )
    logger.info("🎯 Keyterms + 🎤 Diarization both enabled")
    logger.info("   Try multiple speakers saying difficult names!")
    await run_bot(
        connection_params,
        "Keyterms + Diarization Combined",
        speaker_format="[{speaker}] {text}",
    )


# ============================================================================
# Interactive Menu
# ============================================================================


def show_menu():
    """Display the comprehensive test menu."""
    print("\n" + "=" * 80)
    print("AssemblyAI u3-rt-pro Comprehensive Test Suite")
    print("=" * 80)
    print("\n📋 BASIC CONFIGURATION (1-3)")
    print("  1. Basic Default (100ms)")
    print("  2. Custom Silence (200ms)")
    print("  3. Longer Silence (500ms)")

    print("\n⚠️  PROMPTING & WARNINGS (4-7)")
    print("  4. max_turn_silence Warning")
    print("  5. Custom Prompt Warning")
    print("  6. Prompt + Keyterms Conflict (ERROR)")
    print("  7. Keyterms with Difficult Names")

    print("\n🎤 DIARIZATION (8-9)")
    print("  8. Diarization - Basic")
    print("  9. Diarization - XML Formatting")

    print("\n🔄 DYNAMIC UPDATES - SINGLE (10-13)")
    print(" 10. Dynamic Keyterms (Before/After with difficult names)")
    print(" 11. Dynamic Silence (100ms → 3s DRAMATIC)")
    print(" 12. Dynamic Prompt with Keyterms (Before/After)")
    print(" 13. Dynamic Clear Keyterms")

    print("\n🔄 DYNAMIC UPDATES - MULTIPLE (14-15)")
    print(" 14. Multiple Parameters at Once")
    print(" 15. Complex Update Sequence (4 stages)")

    print("\n⚖️  MODE COMPARISON (16-17)")
    print(" 16. Pipecat Mode (VAD + Smart Turn)")
    print(" 17. STT Mode (Model Turn Detection)")

    print("\n⏱️  STT MODE TIMING EXPERIMENTS (18-20)")
    print(" 18. STT: Long Max (5s) + Short Min (100ms)")
    print(" 19. STT: Long Min (3s)")
    print(" 20. STT: Both Short (300ms/100ms)")

    print("\n🎯 EDGE CASES (21-23)")
    print(" 21. Very Long Silence (10s - STT Mode)")
    print(" 22. Very Short Silence (50ms)")
    print(" 23. Keyterms + Diarization Combined")

    print("\n  0. Exit")
    print("\n" + "=" * 80)


async def main():
    """Main interactive menu."""
    tests = {
        "1": test_01_basic_100ms,
        "2": test_02_custom_200ms,
        "3": test_03_custom_500ms,
        "4": test_04_max_warning,
        "5": test_05_prompt_warning,
        "6": test_06_prompt_keyterms_conflict,
        "7": test_07_keyterms_difficult,
        "8": test_08_diarization_basic,
        "9": test_09_diarization_xml,
        "10": test_10_dynamic_keyterms,
        "11": test_11_dynamic_silence,
        "12": test_12_dynamic_prompt,
        "13": test_13_dynamic_clear_keyterms,
        "14": test_14_multi_param_update,
        "15": test_15_complex_sequence,
        "16": test_16_pipecat_mode,
        "17": test_17_stt_mode,
        "18": test_18_stt_long_max_short_min,
        "19": test_19_stt_long_min,
        "20": test_20_stt_both_short,
        "21": test_21_very_long_silence,
        "22": test_22_very_short_silence,
        "23": test_23_keyterms_plus_diarization,
    }

    while True:
        show_menu()
        choice = input("Enter test number (or 0 to exit): ").strip()

        if choice == "0":
            print("\n👋 Goodbye!")
            break

        if choice in tests:
            try:
                await tests[choice]()
            except KeyboardInterrupt:
                print("\n\n⚠️ Test interrupted by user")
            except Exception as e:
                logger.error(f"Test failed with error: {e}")
                import traceback

                traceback.print_exc()

            input("\n\nPress Enter to return to menu...")
        else:
            print(f"\n❌ Invalid choice: {choice}")
            input("Press Enter to continue...")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
