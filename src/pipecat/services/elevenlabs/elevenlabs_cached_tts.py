#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Cache-aware ElevenLabs TTS service.

This module provides an ElevenLabs TTS service that checks a pre-warmed cache
before generating audio. The cache is READ-ONLY during pipeline execution -
no caching happens during the pipeline run.

Pre-warm the cache using TTSCacheWarmer before running the pipeline.
"""

import uuid
from typing import Any, AsyncGenerator, Dict, Optional

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.dograh.tts_cache import CachedTTSData, TTSCacheManager
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService


class ElevenLabsCachedTTSService(ElevenLabsTTSService):
    """ElevenLabs TTS service with read-only cache support.

    This service extends ElevenLabsTTSService to check a pre-warmed Redis cache
    before generating audio via the ElevenLabs API. The cache is READ-ONLY:

    - Cache is pre-warmed using TTSCacheWarmer BEFORE the pipeline runs
    - During pipeline execution, only cache READS happen (no writes)
    - If cache hit: replay cached audio + word timestamps
    - If cache miss: fall through to normal ElevenLabs generation

    This approach ensures proper 1:1 correlation between text and audio because
    the cache warmer sends exactly ONE sentence per context.

    Example usage:
        tts = ElevenLabsCachedTTSService(
            api_key="your-api-key",
            voice_id="voice-id",
            cache_redis_url="redis://localhost:6379/0",
        )

    Pre-warm the cache before pipeline:
        warmer = TTSCacheWarmer(
            api_key="your-api-key",
            voice_id="voice-id",
            redis_url="redis://localhost:6379/0",
        )
        await warmer.warm_cache(["Hello!", "Goodbye!"])
    """

    def __init__(
        self,
        *,
        cache_redis_url: Optional[str] = None,
        cache_ttl_seconds: int = 86400,
        **kwargs,
    ):
        """Initialize the cached ElevenLabs TTS service.

        Args:
            cache_redis_url: Redis URL for cache lookups. If None, caching is disabled.
            cache_ttl_seconds: Cache TTL in seconds (for cache key generation consistency)
            **kwargs: Arguments passed to ElevenLabsTTSService
        """
        super().__init__(**kwargs)

        self._cache_enabled = cache_redis_url is not None
        self._cache_manager: Optional[TTSCacheManager] = None

        if self._cache_enabled:
            self._cache_manager = TTSCacheManager(
                redis_url=cache_redis_url,
                cache_ttl_seconds=cache_ttl_seconds,
            )

        # Stats for monitoring
        self._cache_hits = 0
        self._cache_misses = 0

    async def start(self, frame):
        """Start the TTS service and connect to cache."""
        await super().start(frame)

        if self._cache_enabled and self._cache_manager:
            try:
                await self._cache_manager.connect()
                logger.info("TTS cache connected")
            except Exception as e:
                logger.warning(f"Failed to connect to TTS cache: {e}")
                self._cache_enabled = False

    async def stop(self, frame):
        """Stop the TTS service and disconnect from cache."""
        await super().stop(frame)

        if self._cache_manager:
            try:
                await self._cache_manager.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting from TTS cache: {e}")

        # Log cache stats
        total = self._cache_hits + self._cache_misses
        if total > 0:
            hit_rate = (self._cache_hits / total) * 100
            logger.info(
                f"TTS cache stats: {self._cache_hits} hits, {self._cache_misses} misses "
                f"({hit_rate:.1f}% hit rate)"
            )

    async def cancel(self, frame):
        """Cancel the TTS service."""
        await super().cancel(frame)

        if self._cache_manager:
            try:
                await self._cache_manager.disconnect()
            except Exception:
                pass

    def _get_cache_settings(self) -> Dict[str, Any]:
        """Get settings that affect TTS output for cache key generation."""
        return {
            "language": self._settings.get("language"),
            "speed": self._settings.get("speed"),
            "stability": self._settings.get("stability"),
            "similarity_boost": self._settings.get("similarity_boost"),
        }

    async def _check_cache(self, text: str) -> Optional[CachedTTSData]:
        """Check if the text is in cache.

        Args:
            text: The text to look up

        Returns:
            CachedTTSData if found, None otherwise
        """
        if not self._cache_enabled or not self._cache_manager:
            return None

        try:
            cache_key = self._cache_manager.generate_cache_key(
                text=text,
                voice_id=self._voice_id,
                model=self.model_name,
                sample_rate=self.sample_rate,
                settings=self._get_cache_settings(),
            )

            logger.debug(f"cache_key for text: {text} cache_key: {cache_key}")

            return await self._cache_manager.get(cache_key)
        except Exception as e:
            logger.warning(f"Error checking TTS cache: {e}")
            return None

    async def _replay_cached_audio(self, cached_data: CachedTTSData):
        """Replay cached audio through the audio context system.

        Args:
            cached_data: The cached TTS data to replay
        """
        # Create a temporary audio context for replaying
        context_id = str(uuid.uuid4())
        await self.create_audio_context(context_id)

        try:
            self.start_word_timestamps()

            # Push all audio chunks
            for audio_chunk in cached_data.audio_chunks:
                frame = TTSAudioRawFrame(
                    audio=audio_chunk,
                    sample_rate=cached_data.sample_rate,
                    num_channels=cached_data.num_channels,
                )
                await self.append_to_audio_context(context_id, frame)

            # Add word timestamps
            if cached_data.word_timestamps:
                await self.add_word_timestamps(cached_data.word_timestamps)

            # Add stop signal
            await self.add_word_timestamps([("TTSStoppedFrame", 0), ("Reset", 0)])

        finally:
            await self.remove_audio_context(context_id)

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text, checking cache first.

        If the text is in cache, replays cached audio.
        Otherwise, falls through to normal ElevenLabs generation.

        Args:
            text: The text to synthesize

        Yields:
            TTS frames
        """
        # Check cache first
        cached_data = await self._check_cache(text)

        if cached_data:
            self._cache_hits += 1
            logger.debug(f"TTS cache hit: {text[:50]}...")

            yield TTSStartedFrame()
            await self._replay_cached_audio(cached_data)
            yield TTSStoppedFrame()
            return

        # Cache miss - use normal ElevenLabs generation
        self._cache_misses += 1
        logger.debug(f"TTS cache miss: {text[:50]}...")

        async for frame in super().run_tts(text):
            yield frame

    @property
    def cache_stats(self) -> Dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary with 'hits' and 'misses' counts
        """
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
        }
