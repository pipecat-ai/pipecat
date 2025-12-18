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

import json
import uuid
from typing import Any, AsyncGenerator, Dict, Optional

from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.tts_cache import CachedTTSData, TTSCacheManager

try:
    from websockets.protocol import State
except ModuleNotFoundError:
    pass  # Already handled by parent import


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

    async def _ensure_context_initialized(self):
        """Ensure WebSocket context is initialized.

        This creates a WebSocket context if one doesn't exist, ensuring it's
        properly registered with the ElevenLabs server. This allows:
        1. Cache hits to use a valid context for audio replay
        2. Proper context closure on interruption
        3. Seamless switching between cached and non-cached responses
        """
        if not self._websocket or self._websocket.state is State.CLOSED:
            await self._connect()

        if not self._started:
            await self.start_ttfb_metrics()
            self._started = True
            self._cumulative_time = 0
            self._partial_word = ""
            self._partial_word_start_time = 0.0

            # Create context ID and audio context
            if not self._context_id:
                self._context_id = str(uuid.uuid4())
            if not self.audio_context_available(self._context_id):
                await self.create_audio_context(self._context_id)

            # Register context with WebSocket server (with voice settings but no text)
            # This ensures the context is valid for both cached replay and API generation
            msg = {"text": " ", "context_id": self._context_id}
            if self._voice_settings:
                msg["voice_settings"] = self._voice_settings
            if self._pronunciation_dictionary_locators:
                msg["pronunciation_dictionary_locators"] = [
                    locator.model_dump()
                    for locator in self._pronunciation_dictionary_locators
                ]
            await self._websocket.send(json.dumps(msg))
            logger.trace(f"Initialized WebSocket context {self._context_id}")

            return True  # New context was created
        return False  # Existing context reused

    async def _replay_cached_audio(self, cached_data: CachedTTSData):
        """Replay cached audio through the audio context system.

        Uses self._context_id which is managed by run_tts and registered with
        the WebSocket server. This ensures proper context management and
        allows parent's interruption handling to work correctly.

        Args:
            cached_data: The cached TTS data to replay
        """
        await self.stop_ttfb_metrics()
        self.start_word_timestamps()

        # Push all audio chunks using the persistent context
        for audio_chunk in cached_data.audio_chunks:
            frame = TTSAudioRawFrame(
                audio=audio_chunk,
                sample_rate=cached_data.sample_rate,
                num_channels=cached_data.num_channels,
            )
            await self.append_to_audio_context(self._context_id, frame)

        # Add word timestamps with cumulative time offset for proper sequencing
        if cached_data.word_timestamps:
            offset_timestamps = [
                (word, time + self._cumulative_time)
                for word, time in cached_data.word_timestamps
            ]
            await self.add_word_timestamps(offset_timestamps)

            # Update cumulative time based on the maximum timestamp
            # This ensures proper sequencing across multiple cached replays
            max_time = max(t for _, t in cached_data.word_timestamps)
            self._cumulative_time += max_time

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text, checking cache first.

        Always initializes a WebSocket context (if not already initialized) to ensure:
        1. Context is properly registered with ElevenLabs server
        2. Parent's interruption handling works correctly
        3. Seamless switching between cached and non-cached responses

        For cache hits: replays cached audio without sending text to API
        For cache misses: sends text to API for generation

        Args:
            text: The text to synthesize

        Yields:
            TTS frames
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            # Always ensure WebSocket context is initialized
            # This registers the context with ElevenLabs server
            is_new_context = await self._ensure_context_initialized()
            if is_new_context:
                yield TTSStartedFrame()

            # Check cache
            cached_data = await self._check_cache(text)

            if cached_data:
                self._cache_hits += 1
                logger.debug(f"TTS cache hit: {text[:50]}...")

                # Replay cached audio (don't send text to WebSocket API)
                await self._replay_cached_audio(cached_data)
                await self.start_tts_usage_metrics(text)
            else:
                # Cache miss - send text to WebSocket API for generation
                self._cache_misses += 1
                logger.debug(f"TTS cache miss: {text[:50]}...")

                await self._send_text(text)
                await self.start_tts_usage_metrics(text)

            yield None

        except Exception as e:
            logger.error(f"Error in TTS generation: {e}")
            yield TTSStoppedFrame()
            yield ErrorFrame(error=f"Error in TTS generation: {e}")
            self._started = False

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
