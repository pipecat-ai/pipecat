#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""TTS Cache Warmer for pre-generating and caching common sentences.

This module provides a utility to pre-warm the TTS cache with common sentences
before the pipeline runs. It sends ONE sentence per context to ElevenLabs to
ensure proper 1:1 correlation between text and audio.

Usage:
    warmer = ElevenLabsCacheWarmer(
        api_key="sk_d20234bcf70d772236b8cfe9c972a8da741279af84d0f2b0",
        voice_id="QTKSa2Iyv0yoxvXY2V8a",
        redis_url=REDIS_URL,
        sample_rate=16000,
        voice_settings={
            'speed': 0.94,
            'stability': 0.8,
            'similarity_boost': 0.75
        }
    )

    sentences = [
        "Hi."
    ]

    await warmer.warm_cache(sentences)
"""

import asyncio
import base64
import json
from typing import Any, Dict, List, Mapping, Optional, Tuple

from loguru import logger

from pipecat.services.tts_cache import CachedTTSData, TTSCacheManager

try:
    import websockets
    from websockets.asyncio.client import connect as websocket_connect
except ModuleNotFoundError:
    logger.error("websockets not installed. TTS cache warmer will not work.")
    websockets = None


def calculate_word_times(
    alignment_info: Mapping[str, Any],
    cumulative_time: float,
    partial_word: str = "",
    partial_word_start_time: float = 0.0,
) -> tuple[List[Tuple[str, float]], str, float]:
    """Calculate word timestamps from character alignment information.

    Args:
        alignment_info: Character alignment data from ElevenLabs API.
        cumulative_time: Base time offset for this chunk.
        partial_word: Partial word carried over from previous chunk.
        partial_word_start_time: Start time of the partial word.

    Returns:
        Tuple of (word_times, new_partial_word, new_partial_word_start_time)
    """
    chars = alignment_info.get("chars", [])
    char_start_times_ms = alignment_info.get("charStartTimesMs", [])

    if len(chars) != len(char_start_times_ms):
        logger.error(
            f"calculate_word_times: length mismatch - chars={len(chars)}, times={len(char_start_times_ms)}"
        )
        return ([], partial_word, partial_word_start_time)

    words = []
    word_start_times = []
    current_word = partial_word
    word_start_time = partial_word_start_time if partial_word else None

    for i, char in enumerate(chars):
        if char == " ":
            if current_word:
                words.append(current_word)
                word_start_times.append(word_start_time)
                current_word = ""
                word_start_time = None
        else:
            if word_start_time is None:
                word_start_time = cumulative_time + (char_start_times_ms[i] / 1000.0)
            current_word += char

    word_times = list(zip(words, word_start_times))
    new_partial_word = current_word if current_word else ""
    new_partial_word_start_time = word_start_time if word_start_time is not None else 0.0

    return (word_times, new_partial_word, new_partial_word_start_time)


class ElevenLabsCacheWarmer:
    """Pre-warms TTS cache by generating audio for common sentences.

    This class connects directly to ElevenLabs WebSocket API and generates
    audio for a list of sentences, storing the results in Redis cache.

    Each sentence is sent in its own context to ensure 1:1 correlation
    between text and audio.
    """

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        model: str = "eleven_flash_v2_5",
        url: str = "wss://api.elevenlabs.io",
        sample_rate: int = 8000,
        redis_url: str,
        cache_ttl_seconds: int = 86400,
        voice_settings: Optional[Dict[str, Any]] = None,
        language: Optional[str] = None,
    ):
        """Initialize the cache warmer.

        Args:
            api_key: ElevenLabs API key
            voice_id: Voice identifier to use
            model: TTS model name
            url: ElevenLabs WebSocket URL
            sample_rate: Output audio sample rate
            redis_url: Redis connection URL
            cache_ttl_seconds: Cache TTL in seconds
            voice_settings: Optional voice settings (stability, similarity_boost, speed)
            language: Optional language code
        """
        self._api_key = api_key
        self._voice_id = voice_id
        self._model = model
        self._url = url
        self._sample_rate = sample_rate
        self._voice_settings = voice_settings
        self._language = language

        self._cache_manager = TTSCacheManager(
            redis_url=redis_url,
            cache_ttl_seconds=cache_ttl_seconds,
        )

        # Output format based on sample rate
        self._output_format = self._get_output_format(sample_rate)

    def _get_output_format(self, sample_rate: int) -> str:
        """Get ElevenLabs output format for sample rate."""
        formats = {
            8000: "pcm_8000",
            16000: "pcm_16000",
            22050: "pcm_22050",
            24000: "pcm_24000",
            44100: "pcm_44100",
        }
        return formats.get(sample_rate, "pcm_24000")

    def _get_cache_relevant_settings(self) -> Dict[str, Any]:
        """Get settings that affect TTS output for cache key generation.

        Returns dict with all 4 keys to match ElevenLabsCachedTTSService format:
        - language
        - speed
        - stability
        - similarity_boost
        """
        return {
            "language": self._language,
            "speed": self._voice_settings.get("speed") if self._voice_settings else None,
            "stability": self._voice_settings.get("stability") if self._voice_settings else None,
            "similarity_boost": self._voice_settings.get("similarity_boost")
            if self._voice_settings
            else None,
        }

    async def warm_cache(
        self,
        sentences: List[str],
        skip_existing: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, bool]:
        """Pre-warm cache with the given sentences.

        Args:
            sentences: List of sentences to cache
            skip_existing: If True, skip sentences already in cache
            progress_callback: Optional callback for progress updates
                Signature: callback(sentence: str, index: int, total: int, success: bool)

        Returns:
            Dictionary mapping sentence to success status
        """
        await self._cache_manager.connect()

        results = {}
        total = len(sentences)

        for i, sentence in enumerate(sentences):
            # Generate cache key
            cache_key = self._cache_manager.generate_cache_key(
                text=sentence,
                voice_id=self._voice_id,
                model=self._model,
                sample_rate=self._sample_rate,
                settings=self._get_cache_relevant_settings(),
            )

            # Check if already cached
            if skip_existing and await self._cache_manager.exists(cache_key):
                logger.info(f"[{i + 1}/{total}] Already cached: {sentence[:50]}...")
                results[sentence] = True
                if progress_callback:
                    progress_callback(sentence, i, total, True)
                continue

            # Generate and cache
            logger.info(f"[{i + 1}/{total}] Generating: {sentence[:50]}...")
            try:
                cached_data = await self._generate_single_sentence(sentence)
                if cached_data:
                    success = await self._cache_manager.set(cache_key, cached_data)
                    results[sentence] = success
                    if success:
                        logger.info(
                            f"[{i + 1}/{total}] Cached: {sentence[:50]}... "
                            f"({len(cached_data.audio_chunks)} chunks, "
                            f"{cached_data.total_duration_seconds:.2f}s)"
                        )
                else:
                    results[sentence] = False
                    logger.error(f"[{i + 1}/{total}] Failed to generate: {sentence[:50]}...")

            except Exception as e:
                logger.error(f"[{i + 1}/{total}] Error: {e}")
                results[sentence] = False

            if progress_callback:
                progress_callback(sentence, i, total, results[sentence])

        await self._cache_manager.disconnect()
        return results

    async def _generate_single_sentence(self, text: str) -> Optional[CachedTTSData]:
        """Generate TTS for a single sentence using ElevenLabs WebSocket.

        This sends ONE sentence in ONE context to ensure 1:1 correlation.

        Args:
            text: The sentence to generate audio for

        Returns:
            CachedTTSData if successful, None otherwise
        """
        if websockets is None:
            raise RuntimeError("websockets package not installed")

        # Build WebSocket URL
        url = (
            f"{self._url}/v1/text-to-speech/{self._voice_id}/stream-input"
            f"?model_id={self._model}"
            f"&output_format={self._output_format}"
        )

        if self._language:
            url += f"&language_code={self._language}"

        audio_chunks: List[bytes] = []
        word_timestamps: List[Tuple[str, float]] = []
        cumulative_time = 0.0
        partial_word = ""
        partial_word_start_time = 0.0
        total_duration = 0.0

        try:
            async with websocket_connect(
                url,
                max_size=16 * 1024 * 1024,
                additional_headers={"xi-api-key": self._api_key},
            ) as ws:
                # Send initial message with voice settings
                init_msg: Dict[str, Any] = {
                    "text": " ",
                    "try_trigger_generation": True,
                    "generation_config": {"chunk_length_schedule": [50]},
                }

                if self._voice_settings:
                    init_msg["voice_settings"] = self._voice_settings

                await ws.send(json.dumps(init_msg))

                # Send the actual text
                text_msg = {
                    "text": text,
                    "try_trigger_generation": True,
                }
                await ws.send(json.dumps(text_msg))

                # Send end of stream
                end_msg = {"text": ""}
                await ws.send(json.dumps(end_msg))

                # Receive all responses
                async for message in ws:
                    msg = json.loads(message)

                    if msg.get("audio"):
                        audio_data = base64.b64decode(msg["audio"])
                        audio_chunks.append(audio_data)

                    if msg.get("alignment"):
                        alignment = msg["alignment"]
                        word_times, partial_word, partial_word_start_time = calculate_word_times(
                            alignment,
                            cumulative_time,
                            partial_word,
                            partial_word_start_time,
                        )

                        if word_times:
                            word_timestamps.extend(word_times)

                        # Update cumulative time
                        char_start_times_ms = alignment.get("charStartTimesMs", [])
                        char_durations_ms = alignment.get("charDurationsMs", [])

                        if char_start_times_ms and char_durations_ms:
                            chunk_end_time_ms = char_start_times_ms[-1] + char_durations_ms[-1]
                            cumulative_time += chunk_end_time_ms / 1000.0
                            total_duration = cumulative_time

                    if msg.get("isFinal"):
                        break

                # Add any remaining partial word
                if partial_word:
                    word_timestamps.append((partial_word, partial_word_start_time))

            if audio_chunks:
                return CachedTTSData(
                    audio_chunks=audio_chunks,
                    sample_rate=self._sample_rate,
                    num_channels=1,
                    word_timestamps=word_timestamps,
                    total_duration_seconds=total_duration,
                )

            return None

        except Exception as e:
            logger.error(f"Error generating TTS: {e}")
            return None

    async def check_cache_status(self, sentences: List[str]) -> Dict[str, bool]:
        """Check which sentences are already cached.

        Args:
            sentences: List of sentences to check

        Returns:
            Dictionary mapping sentence to cached status
        """
        await self._cache_manager.connect()

        results = {}
        for sentence in sentences:
            cache_key = self._cache_manager.generate_cache_key(
                text=sentence,
                voice_id=self._voice_id,
                model=self._model,
                sample_rate=self._sample_rate,
                settings=self._get_cache_relevant_settings(),
            )
            results[sentence] = await self._cache_manager.exists(cache_key)

        await self._cache_manager.disconnect()
        return results


async def warm_cache_cli():
    """CLI entry point for cache warming.

    Usage:
        python -m pipecat.services.dograh.tts_cache_warmer

    Environment variables:
        ELEVENLABS_API_KEY: ElevenLabs API key
        ELEVENLABS_VOICE_ID: Voice ID to use
        REDIS_URL: Redis connection URL
    """
    import os

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    voice_id = os.environ.get("ELEVENLABS_VOICE_ID")
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

    if not api_key or not voice_id:
        print("Error: ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID must be set")
        return

    # Example sentences - in practice, these would come from a file or database
    sentences = [
        "Hello, how can I help you today?",
        "Thank you for calling.",
        "Is there anything else I can help you with?",
        "Please hold while I look that up for you.",
        "I understand your concern.",
        "Let me transfer you to the right department.",
    ]

    warmer = ElevenLabsCacheWarmer(
        api_key=api_key,
        voice_id=voice_id,
        redis_url=redis_url,
    )

    results = await warmer.warm_cache(sentences)

    # Summary
    cached = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    print(f"\nCache warming complete: {cached} cached, {failed} failed")


if __name__ == "__main__":
    asyncio.run(warm_cache_cli())
