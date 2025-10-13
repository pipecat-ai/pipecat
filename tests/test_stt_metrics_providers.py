#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""
Pytest-based STT Metrics Comparison Tests

This test suite compares multiple STT providers with the same audio samples
and verifies their metrics (accuracy, speed, cost, quality).

Usage:
    pytest test_stt_metrics_providers.py
    pytest test_stt_metrics_providers.py -v
    pytest test_stt_metrics_providers.py -k "test_deepgram"
    pytest test_stt_metrics_providers.py --html=report.html
"""

import asyncio
import json
import os
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pytest
from loguru import logger

from pipecat.frames.frames import EndFrame, InputAudioRawFrame, MetricsFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask


@dataclass
class STTTestResult:
    """Results from testing a single STT provider."""

    provider: str
    model: str

    # Accuracy
    transcript: str
    word_error_rate: Optional[float] = None

    # Performance
    audio_duration: float = 0.0
    processing_time: Optional[float] = None
    real_time_factor: Optional[float] = None
    words_per_second: Optional[float] = None
    ttft: Optional[float] = None

    # Quality
    average_confidence: Optional[float] = None

    # Cost
    estimated_cost: Optional[float] = None
    cost_per_word: Optional[float] = None

    # Metadata
    requests: Optional[int] = None
    word_count: Optional[int] = None
    character_count: Optional[int] = None


@dataclass
class AudioTestCase:
    """A test case with audio and expected transcript."""

    name: str
    audio_file: str
    ground_truth: str
    duration: float


class STTMetricsCollector(BaseObserver):
    """Collects metrics from STT services."""

    def __init__(self):
        super().__init__()
        self.results: List[STTTestResult] = []
        self.transcriptions: List[str] = []  # Collect all final transcriptions
        self.metrics_data = None
        self.latest_metrics = None  # Track the most recent/complete metrics

    async def on_push_frame(self, data: FramePushed):
        from pipecat.frames.frames import InterimTranscriptionFrame, TranscriptionFrame

        # Capture transcript (only final transcriptions, not interim)
        if isinstance(data.frame, TranscriptionFrame) and not isinstance(
            data.frame, InterimTranscriptionFrame
        ):
            # Store each final transcription separately instead of concatenating
            self.transcriptions.append(data.frame.text)

        # Capture metrics (always update to the latest)
        if isinstance(data.frame, MetricsFrame):
            for metric_data in data.frame.data:
                if hasattr(metric_data, "value") and hasattr(
                    metric_data.value, "audio_duration_seconds"
                ):
                    # Keep updating with the latest metrics
                    self.latest_metrics = metric_data

        # When EndFrame arrives, finalize result with the latest metrics
        if isinstance(data.frame, EndFrame):
            if self.latest_metrics and not self.results:
                self.metrics_data = self.latest_metrics
                self._create_result()

    def _create_result(self):
        """Create result from collected metrics and transcript."""
        if not self.metrics_data:
            return

        # Use the last final transcription (for single-segment audio testing)
        # or join multiple transcriptions if there are multiple segments
        if len(self.transcriptions) == 1:
            final_transcript = self.transcriptions[0].strip()
        elif len(self.transcriptions) > 1:
            # Log warning if multiple transcriptions detected (possible duplicate issue)
            logger.warning(
                f"Multiple final transcriptions detected ({len(self.transcriptions)}): {self.transcriptions}"
            )
            # Use the last one as it's typically the most complete
            final_transcript = self.transcriptions[-1].strip()
        else:
            final_transcript = ""

        usage = self.metrics_data.value
        result = STTTestResult(
            provider=self.metrics_data.processor,
            model=self.metrics_data.model or "unknown",
            transcript=final_transcript,
            word_error_rate=usage.word_error_rate,
            audio_duration=usage.audio_duration_seconds,
            processing_time=usage.processing_time_seconds,
            real_time_factor=usage.real_time_factor,
            words_per_second=usage.words_per_second,
            ttft=usage.time_to_first_transcript,
            average_confidence=usage.average_confidence,
            estimated_cost=usage.estimated_cost,
            cost_per_word=usage.cost_per_word,
            requests=usage.requests,
            word_count=usage.word_count,
            character_count=usage.character_count,
        )
        self.results.append(result)
        # Clear metrics_data so we don't create duplicate results
        self.metrics_data = None


async def run_stt_service_test(
    service_name: str, stt_service, test_case: AudioTestCase, ground_truth: Optional[str] = None
) -> Optional[STTTestResult]:
    """Run a single STT service with a test case and return results."""

    logger.info(f"Testing {service_name} with '{test_case.name}'...")

    # Set ground truth on the service for automatic WER calculation
    if ground_truth:
        stt_service.set_ground_truth(ground_truth)

    # Create observer
    collector = STTMetricsCollector()

    # Create pipeline
    pipeline = Pipeline([stt_service])

    # Create task with metrics enabled
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )
    task.add_observer(collector)

    try:
        # Load audio file
        with wave.open(test_case.audio_file, "rb") as wf:
            sample_rate = wf.getframerate()
            num_channels = wf.getnchannels()
            audio_data = wf.readframes(wf.getnframes())

        # Send VAD frame to indicate user started speaking (for SegmentedSTTService)
        from pipecat.frames.frames import UserStartedSpeakingFrame, UserStoppedSpeakingFrame

        await task.queue_frame(UserStartedSpeakingFrame())

        # Send audio frames
        audio_frame = InputAudioRawFrame(
            audio=audio_data, sample_rate=sample_rate, num_channels=num_channels
        )
        await task.queue_frame(audio_frame)

        # Send VAD frame to indicate user stopped speaking (triggers transcription)
        await task.queue_frame(UserStoppedSpeakingFrame())

        await task.queue_frame(EndFrame())

        # Run pipeline
        runner = PipelineRunner()
        await runner.run(task)

        # Return results
        return collector.results[0] if collector.results else None

    except Exception as e:
        logger.error(f"Error testing {service_name}: {e}")
        return None


async def run_stt_comparison(test_cases: List[AudioTestCase], providers: Dict) -> Dict:
    """Run comparison across all providers and test cases."""

    all_results = {}

    for test_case in test_cases:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Test Case: {test_case.name}")
        logger.info(f"Ground Truth: '{test_case.ground_truth}'")
        logger.info(f"{'=' * 70}\n")

        test_results = []

        for provider_name, stt_service in providers.items():
            try:
                result = await run_stt_service_test(
                    provider_name, stt_service, test_case, test_case.ground_truth
                )
                if result:
                    test_results.append(result)
                    wer_str = (
                        f"WER: {result.word_error_rate:.2f}%" if result.word_error_rate else ""
                    )
                    logger.info(f"✅ {provider_name}: '{result.transcript}' {wer_str}")
            except Exception as e:
                logger.error(f"❌ {provider_name} failed: {e}")

        all_results[test_case.name] = test_results

    return all_results


def generate_comparison_report(results: Dict, output_file: str = "stt_comparison_report.json"):
    """Generate a detailed comparison report."""

    logger.info(f"\n{'=' * 70}")
    logger.info("STT METRICS COMPARISON REPORT")
    logger.info(f"{'=' * 70}\n")

    report = {}

    for test_name, test_results in results.items():
        logger.info(f"\n📊 Test: {test_name}")
        logger.info("-" * 70)

        if not test_results:
            logger.warning("No results for this test case")
            continue

        # Sort by different metrics
        by_accuracy = sorted(
            test_results, key=lambda x: x.word_error_rate if x.word_error_rate is not None else 100
        )
        by_speed = sorted(
            test_results,
            key=lambda x: x.real_time_factor if x.real_time_factor is not None else 999,
        )
        by_cost = sorted(
            test_results, key=lambda x: x.estimated_cost if x.estimated_cost is not None else 999
        )

        logger.info("\n🎯 ACCURACY (by WER - lower is better):")
        for result in by_accuracy:
            wer_str = (
                f"{result.word_error_rate:.2f}%" if result.word_error_rate is not None else "N/A"
            )
            conf_str = (
                f"{result.average_confidence:.2%}"
                if result.average_confidence is not None
                else "N/A"
            )
            logger.info(
                f"  {result.provider:20} WER: {wer_str:8} | Confidence: {conf_str} | '{result.transcript}'"
            )

        logger.info("\n⚡ SPEED (by RTF - lower is better):")
        for result in by_speed:
            rtf_str = (
                f"{result.real_time_factor:.3f}" if result.real_time_factor is not None else "N/A"
            )
            wps_str = (
                f"{result.words_per_second:.1f}" if result.words_per_second is not None else "N/A"
            )
            ttft_str = f"{result.ttft:.3f}s" if result.ttft is not None else "N/A"
            logger.info(
                f"  {result.provider:20} RTF: {rtf_str:8} | WPS: {wps_str:8} | TTFT: {ttft_str}"
            )

        logger.info("\n💰 COST (lower is better):")
        for result in by_cost:
            cost_str = (
                f"${result.estimated_cost:.6f}" if result.estimated_cost is not None else "N/A"
            )
            cpw_str = f"${result.cost_per_word:.6f}" if result.cost_per_word is not None else "N/A"
            logger.info(f"  {result.provider:20} Total: {cost_str:12} | Per word: {cpw_str}")

        # Convert to dict for JSON
        report[test_name] = [asdict(r) for r in test_results]

    # Save to JSON
    output_path = Path(__file__).parent / output_file
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\n📁 Full report saved to: {output_path}")

    # Recommendations
    logger.info(f"\n{'=' * 70}")
    logger.info("🎯 RECOMMENDATIONS")
    logger.info(f"{'=' * 70}")

    # Find best overall
    for test_name, test_results in results.items():
        if not test_results:
            continue

        best_accuracy = min(
            test_results, key=lambda x: x.word_error_rate if x.word_error_rate is not None else 100
        )
        best_speed = min(
            test_results,
            key=lambda x: x.real_time_factor if x.real_time_factor is not None else 999,
        )
        best_cost = min(
            test_results, key=lambda x: x.estimated_cost if x.estimated_cost is not None else 999
        )

        logger.info(f"\nFor '{test_name}':")
        if best_accuracy.word_error_rate is not None:
            logger.info(
                f"  🎯 Best Accuracy: {best_accuracy.provider} (WER: {best_accuracy.word_error_rate:.2f}%)"
            )
        if best_speed.real_time_factor is not None:
            logger.info(
                f"  ⚡ Fastest: {best_speed.provider} (RTF: {best_speed.real_time_factor:.3f})"
            )
        if best_cost.estimated_cost is not None:
            logger.info(f"  💰 Cheapest: {best_cost.provider} (${best_cost.estimated_cost:.6f})")


# ============================================================================
# Pytest Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def test_audio_dir():
    """Get the test_audio directory path."""
    return Path(__file__).parent / "test_audio"


@pytest.fixture(scope="session")
def test_cases(test_audio_dir):
    """Get list of test cases from test_audio directory."""
    test_cases = []

    test_definitions = [
        ("greeting.wav", "Hello, how are you today?", 2.5),
        ("technical.wav", "The API endpoint returns JSON with authentication headers.", 4.0),
        ("numbers.wav", "The meeting is scheduled for March 15th at 3:30 PM.", 3.5),
        ("sample.wav", "You might also want to consider setting up a page on Facebook", 3.0),
    ]

    for audio_file, ground_truth, duration in test_definitions:
        audio_path = test_audio_dir / audio_file
        if audio_path.exists():
            test_cases.append(
                AudioTestCase(
                    name=audio_file.replace(".wav", ""),
                    audio_file=str(audio_path),
                    ground_truth=ground_truth,
                    duration=duration,
                )
            )

    if not test_cases:
        pytest.skip(
            f"No test audio files found in {test_audio_dir}. Please create test audio files first."
        )

    return test_cases


@pytest.fixture(scope="session")
def available_providers():
    """Get dictionary of available STT providers based on environment variables."""
    providers = {}

    # Deepgram
    if os.getenv("DEEPGRAM_API_KEY"):
        try:
            from pipecat.services.deepgram.stt import DeepgramSTTService

            providers["deepgram"] = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
            logger.info("✅ Deepgram configured")
        except ImportError:
            logger.warning("⚠️ Deepgram not available (install pipecat-ai[deepgram])")

    # OpenAI Whisper
    if os.getenv("OPENAI_API_KEY"):
        try:
            from pipecat.services.openai.stt import OpenAISTTService

            providers["openai"] = OpenAISTTService(api_key=os.getenv("OPENAI_API_KEY"))
            logger.info("✅ OpenAI configured")
        except ImportError:
            logger.warning("⚠️ OpenAI not available (install pipecat-ai[openai])")

    # Azure
    if os.getenv("AZURE_SPEECH_API_KEY") and os.getenv("AZURE_SPEECH_REGION"):
        try:
            from pipecat.services.azure.stt import AzureSTTService

            providers["azure"] = AzureSTTService(
                api_key=os.getenv("AZURE_SPEECH_API_KEY"),
                region=os.getenv("AZURE_SPEECH_REGION"),
            )
            logger.info("✅ Azure configured")
        except ImportError:
            logger.warning("⚠️ Azure not available (install pipecat-ai[azure])")

    # Google
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        try:
            from pipecat.services.google.stt import GoogleSTTService

            providers["google"] = GoogleSTTService()
            logger.info("✅ Google configured")
        except ImportError:
            logger.warning("⚠️ Google not available (install pipecat-ai[google])")

    # Groq Whisper
    if os.getenv("GROQ_API_KEY"):
        try:
            from pipecat.services.groq.stt import GroqSTTService

            providers["groq"] = GroqSTTService(api_key=os.getenv("GROQ_API_KEY"))
            logger.info("✅ Groq configured")
        except ImportError:
            logger.warning("⚠️ Groq not available (install pipecat-ai[groq])")

    # AssemblyAI
    if os.getenv("ASSEMBLYAI_API_KEY"):
        try:
            from pipecat.services.assemblyai.stt import AssemblyAISTTService

            providers["assemblyai"] = AssemblyAISTTService(api_key=os.getenv("ASSEMBLYAI_API_KEY"))
            logger.info("✅ AssemblyAI configured")
        except ImportError:
            logger.warning("⚠️ AssemblyAI not available (install pipecat-ai[assemblyai])")

    # ElevenLabs
    if os.getenv("ELEVENLABS_API_KEY"):
        try:
            from pipecat.services.elevenlabs.stt import ElevenLabsSTTService

            providers["elevenlabs"] = ElevenLabsSTTService(api_key=os.getenv("ELEVENLABS_API_KEY"))
            logger.info("✅ ElevenLabs configured")
        except ImportError:
            logger.warning("⚠️ ElevenLabs not available (install pipecat-ai[elevenlabs])")

    if not providers:
        pytest.skip("No STT providers configured. Please set API keys as environment variables.")

    return providers


# ============================================================================
# Pytest Test Functions
# ============================================================================


@pytest.mark.asyncio
async def test_stt_service_basic(available_providers, test_cases):
    """Test that at least one STT service works with at least one test case."""
    # This is a basic smoke test
    provider_name = list(available_providers.keys())[0]
    stt_service = available_providers[provider_name]
    test_case = test_cases[0]

    result = await run_stt_service_test(
        provider_name, stt_service, test_case, test_case.ground_truth
    )

    assert result is not None, f"Failed to get result from {provider_name}"
    assert result.transcript, "Transcript should not be empty"
    # The provider name from metrics includes the service class name, just verify it's not empty
    assert result.provider, "Provider name should not be empty"
    logger.info(f"✅ Test passed for {provider_name}: '{result.transcript}'")


def get_comprehensive_test_params(available_providers, test_cases):
    """Generate parameters for comprehensive testing (provider x test_case combinations)."""
    params = []
    ids = []
    for provider_name, provider_service in available_providers.items():
        for test_case in test_cases:
            params.append((provider_name, provider_service, test_case))
            ids.append(f"{provider_name}-{test_case.name}")
    return {"argnames": "provider_name,provider_service,test_case", "argvalues": params, "ids": ids}


@pytest.fixture(scope="module")
def comprehensive_params(available_providers, test_cases):
    """Fixture to provide comprehensive test parameters."""
    return get_comprehensive_test_params(available_providers, test_cases)


@pytest.mark.asyncio
async def test_stt_providers_comprehensive(available_providers, test_cases):
    """Test each STT provider with each test case comprehensively (all combinations)."""
    # Test each provider with each test case
    for provider_name, provider_service in available_providers.items():
        for test_case in test_cases:
            logger.info(f"\n🧪 Testing {provider_name} with '{test_case.name}'...")

            result = await run_stt_service_test(
                provider_name, provider_service, test_case, test_case.ground_truth
            )

            # Assertions
            assert result is not None, f"Failed to get result from {provider_name}"
            assert result.transcript, f"Transcript should not be empty for {provider_name}"
            assert result.provider, "Provider name should not be empty"

            # Log results
            logger.info(f"✅ {provider_name} - {test_case.name}: '{result.transcript}'")
            if result.word_error_rate is not None:
                logger.info(f"   WER: {result.word_error_rate:.2f}%")
                assert result.word_error_rate >= 0, "WER should be non-negative"

            if result.real_time_factor is not None:
                logger.info(f"   RTF: {result.real_time_factor:.3f}")
                assert result.real_time_factor > 0, "RTF should be positive"


@pytest.mark.asyncio
async def test_stt_comparison_all(available_providers, test_cases, tmp_path):
    """Run full comparison across all providers and generate report."""
    logger.info("🎤 Running comprehensive STT comparison")
    logger.info(f"Providers: {list(available_providers.keys())}")
    logger.info(f"Test cases: {[tc.name for tc in test_cases]}")

    # Run comparison
    results = await run_stt_comparison(test_cases, available_providers)

    # Verify results
    assert results, "Should have results"
    for test_name, test_results in results.items():
        assert test_results, f"Should have results for {test_name}"
        for result in test_results:
            assert result.transcript, f"Transcript should not be empty for {result.provider}"

    # Generate report
    output_file = tmp_path / "stt_comparison_report.json"

    # Save report
    report = {}
    for test_name, test_results in results.items():
        report[test_name] = [asdict(r) for r in test_results]

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"📁 Report saved to: {output_file}")

    # Verify report was created
    assert output_file.exists(), "Report file should be created"

    # Display summary
    for test_name, test_results in results.items():
        logger.info(f"\n📊 Test: {test_name}")
        for result in test_results:
            wer_str = f"WER: {result.word_error_rate:.2f}%" if result.word_error_rate else "N/A"
            logger.info(f"  {result.provider}: '{result.transcript}' ({wer_str})")
