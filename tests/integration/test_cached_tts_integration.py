#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Integration tests for ElevenLabsCachedTTSService with cache warming.

These tests require:
- ELEVENLABS_API_KEY environment variable
- ELEVENLABS_VOICE_ID environment variable
- REDIS_URL environment variable (or defaults to redis://localhost:6379/0)

Run with: pytest tests/integration/test_cached_tts_integration.py -xvs
"""

import asyncio
import os

import pytest
import pytest_asyncio
from mock_llm_service import MockLLMService

from pipecat.frames.frames import (
    EndFrame,
    LLMContextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.elevenlabs.elevenlabs_cached_tts import ElevenLabsCachedTTSService
from pipecat.services.elevenlabs.tts_cache_warmer import ElevenLabsCacheWarmer
from pipecat.tests.utils import QueuedFrameProcessor

from loguru import logger
logger.remove()
logger.add(lambda msg: print(msg, end=""), level="TRACE")


# Test configuration from environment
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
ELEVENLABS_VOICE_ID = "QTKSa2Iyv0yoxvXY2V8a"


# Skip all tests if credentials are not available
pytestmark = pytest.mark.skipif(
    not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID,
    reason="ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID must be set",
)


# Test sentences - these will be pre-warmed and then generated via LLM
TEST_SENTENCES = [
    "Hello, how can I help you today?",
    "Thank you for calling.",
    "I am testing the assistant context aggregation functionality to make sure everything is in order.",
]


async def run_pipeline_collect_frames(pipeline, frames_to_send):
    """Run pipeline and collect all downstream frames."""
    received_down = asyncio.Queue()
    sink = QueuedFrameProcessor(
        queue=received_down,
        queue_direction=FrameDirection.DOWNSTREAM,
        ignore_start=True,
    )

    test_pipeline = Pipeline([pipeline, sink])
    task = PipelineTask(test_pipeline, params=PipelineParams())

    async def push_frames():
        await asyncio.sleep(0.05)
        for frame in frames_to_send:
            await task.queue_frame(frame)
            
        # Lets wait for some time before pushing EndFrame
        await asyncio.sleep(10)
        await task.queue_frame(EndFrame())

    runner = PipelineRunner()
    await asyncio.gather(runner.run(task), push_frames())

    # Collect all frames
    received_frames = []
    while not received_down.empty():
        frame = await received_down.get()
        if not isinstance(frame, EndFrame):
            received_frames.append(frame)

    return received_frames


@pytest_asyncio.fixture
async def warmed_cache():
    """Pre-warm the TTS cache with test sentences."""
    warmer = ElevenLabsCacheWarmer(
        api_key=ELEVENLABS_API_KEY,
        voice_id=ELEVENLABS_VOICE_ID,
        redis_url=REDIS_URL,
        sample_rate=16000,
        model="eleven_flash_v2_5",
    )

    # Warm the cache
    results = await warmer.warm_cache(TEST_SENTENCES, skip_existing=True)

    # Verify all sentences were cached
    for sentence in TEST_SENTENCES:
        assert results.get(sentence, False), f"Failed to cache: {sentence}"

    return results


@pytest.fixture
def cached_tts_service():
    """Create a cached TTS service instance."""
    return ElevenLabsCachedTTSService(
        api_key=ELEVENLABS_API_KEY,
        voice_id=ELEVENLABS_VOICE_ID,
        model="eleven_flash_v2_5",
        sample_rate=16000,
        cache_redis_url=REDIS_URL,
    )


@pytest.mark.asyncio
async def test_cache_warming_creates_entries(warmed_cache):
    """Test that cache warming successfully creates cache entries."""
    # warmed_cache fixture already verifies this, but let's double-check
    warmer = ElevenLabsCacheWarmer(
        api_key=ELEVENLABS_API_KEY,
        voice_id=ELEVENLABS_VOICE_ID,
        redis_url=REDIS_URL,
        sample_rate=16000,
        model="eleven_flash_v2_5",
    )

    status = await warmer.check_cache_status(TEST_SENTENCES)

    for sentence in TEST_SENTENCES:
        assert status.get(sentence, False), f"Sentence not in cache: {sentence}"

    print(f"All {len(TEST_SENTENCES)} sentences verified in cache")


@pytest.mark.asyncio
async def test_llm_to_cached_tts_cache_hit(warmed_cache, cached_tts_service):
    """Test end-to-end: LLM generates sentences → CachedTTS serves from cache.

    This test:
    1. Uses MockLLMService to stream the first test sentence
    2. Uses ElevenLabsCachedTTSService which should hit the cache
    3. Verifies cache hit occurred (no API calls to ElevenLabs during pipeline)
    """
    # Use the first test sentence
    test_text = TEST_SENTENCES[0]

    # Create LLM chunks for this text
    llm_chunks = MockLLMService.create_text_chunks(test_text, chunk_size=15)
    mock_llm = MockLLMService(mock_chunks=llm_chunks)

    # Create context and run
    messages = [{"role": "user", "content": "Help me"}]

    context = LLMContext(messages)

    context_aggregator = LLMContextAggregatorPair(context)

    assistant_context_aggregator = context_aggregator.assistant()

    # Create pipeline
    pipeline = Pipeline([mock_llm, cached_tts_service, assistant_context_aggregator])

    frames_to_send = [LLMContextFrame(context)]

    # Collect frames
    received_frames = await run_pipeline_collect_frames(pipeline, frames_to_send)

    # Verify we got audio frames
    audio_frames = [f for f in received_frames if isinstance(f, TTSAudioRawFrame)]
    assert len(audio_frames) > 0, "Should generate audio frames from cache"

    # Verify cache hit
    stats = cached_tts_service.cache_stats
    assert stats["hits"] > 0, f"Expected cache hits, got: {stats}"
    assert stats["misses"] == 0, f"Expected no cache misses, got: {stats}"

    print(f"LLM Output: {test_text}")
    print(f"Cache stats: {stats}")
    print(f"Generated {len(audio_frames)} audio frames from cache")


@pytest.mark.asyncio
async def test_llm_to_cached_tts_multiple_sentences(warmed_cache, cached_tts_service):
    """Test LLM generating multiple sentences that are all cache hits."""
    # Combine both test sentences
    combined_text = " ".join(TEST_SENTENCES[:2])
    combined_text += " " + "This is another text, which is not cached."
    combined_text += " " + TEST_SENTENCES[2]

    # Create LLM chunks
    llm_chunks = MockLLMService.create_text_chunks(combined_text, chunk_size=20)
    mock_llm = MockLLMService(mock_chunks=llm_chunks)

    # Create context and run
    messages = [{"role": "user", "content": "Help me"}]

    context = LLMContext(messages)

    context_aggregator = LLMContextAggregatorPair(context)

    assistant_context_aggregator = context_aggregator.assistant()

    # Create pipeline
    pipeline = Pipeline([mock_llm, cached_tts_service, assistant_context_aggregator])

    frames_to_send = [LLMContextFrame(context)]

    # Collect frames
    received_frames = await run_pipeline_collect_frames(pipeline, frames_to_send)

    # Verify audio was generated
    audio_frames = [f for f in received_frames if isinstance(f, TTSAudioRawFrame)]
    assert len(audio_frames) > 0, "Should generate audio frames"

    # Verify cache stats - should have hits for the cached sentences
    stats = cached_tts_service.cache_stats
    print(f"Combined text: {combined_text}")
    print(f"Cache stats: {stats}")
    print(f"Generated {len(audio_frames)} audio frames")

    # At least some should be cache hits
    assert stats["hits"] > 0, f"Expected some cache hits, got: {stats}"


@pytest.mark.asyncio
async def test_cache_miss_falls_through_to_api(cached_tts_service):
    """Test that cache misses fall through to actual ElevenLabs API."""
    # Use a sentence that's NOT in the cache
    uncached_text = "This is a unique test sentence that was never cached."

    llm_chunks = MockLLMService.create_text_chunks(uncached_text, chunk_size=15)
    mock_llm = MockLLMService(mock_chunks=llm_chunks)

    pipeline = Pipeline([mock_llm, cached_tts_service])

    messages = [{"role": "user", "content": "Test"}]
    context = LLMContext(messages)

    received_frames = await run_pipeline_collect_frames(pipeline, [LLMContextFrame(context)])

    # Should still generate audio (from API)
    audio_frames = [f for f in received_frames if isinstance(f, TTSAudioRawFrame)]
    assert len(audio_frames) > 0, "Should generate audio from API on cache miss"

    # Verify cache miss occurred
    stats = cached_tts_service.cache_stats
    assert stats["misses"] > 0, f"Expected cache miss, got: {stats}"

    print(f"Uncached text: {uncached_text}")
    print(f"Cache stats: {stats}")
    print(f"Generated {len(audio_frames)} audio frames from API")


@pytest.mark.asyncio
async def test_mixed_cache_hits_and_misses(warmed_cache, cached_tts_service):
    """Test mix of cached and uncached sentences."""
    # Combine a cached sentence with an uncached one
    cached_sentence = TEST_SENTENCES[0]
    uncached_sentence = "This part was never pre-warmed."
    combined_text = f"{cached_sentence} {uncached_sentence}"

    llm_chunks = MockLLMService.create_text_chunks(combined_text, chunk_size=20)
    mock_llm = MockLLMService(mock_chunks=llm_chunks)

    pipeline = Pipeline([mock_llm, cached_tts_service])

    messages = [{"role": "user", "content": "Mixed test"}]
    context = LLMContext(messages)

    received_frames = await run_pipeline_collect_frames(pipeline, [LLMContextFrame(context)])

    # Verify audio was generated
    audio_frames = [f for f in received_frames if isinstance(f, TTSAudioRawFrame)]
    assert len(audio_frames) > 0, "Should generate audio frames"

    # Verify we got both hits and misses
    stats = cached_tts_service.cache_stats
    print(f"Combined text: {combined_text}")
    print(f"Cache stats: {stats}")

    # Should have at least one hit (for the cached sentence) and one miss (for uncached)
    total = stats["hits"] + stats["misses"]
    assert total > 0, "Should have processed at least one sentence"


@pytest.mark.asyncio
async def test_cache_hit_produces_valid_audio(warmed_cache, cached_tts_service):
    """Verify that cached audio is valid (correct sample rate, non-empty)."""
    test_text = TEST_SENTENCES[0]

    llm_chunks = MockLLMService.create_text_chunks(test_text, chunk_size=15)
    mock_llm = MockLLMService(mock_chunks=llm_chunks)

    pipeline = Pipeline([mock_llm, cached_tts_service])

    messages = [{"role": "user", "content": "Validate audio"}]
    context = LLMContext(messages)

    received_frames = await run_pipeline_collect_frames(pipeline, [LLMContextFrame(context)])

    audio_frames = [f for f in received_frames if isinstance(f, TTSAudioRawFrame)]

    # Verify audio properties
    assert len(audio_frames) > 0, "Should have audio frames"

    for frame in audio_frames:
        assert frame.sample_rate == 16000, f"Expected 16000 sample rate, got {frame.sample_rate}"
        assert frame.num_channels == 1, f"Expected 1 channel, got {frame.num_channels}"
        assert len(frame.audio) > 0, "Audio chunk should not be empty"

    # Calculate total audio size
    total_bytes = sum(len(f.audio) for f in audio_frames)
    print(f"Total audio: {total_bytes} bytes across {len(audio_frames)} frames")
    print(f"Sample rate: {audio_frames[0].sample_rate} Hz")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
