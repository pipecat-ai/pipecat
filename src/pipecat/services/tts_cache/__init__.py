#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""TTS caching module for reducing API costs on repeated phrases."""

from pipecat.services.tts_cache.backends.base import CacheBackend
from pipecat.services.tts_cache.backends.memory import MemoryCacheBackend
from pipecat.services.tts_cache.key_generator import generate_cache_key
from pipecat.services.tts_cache.mixin import TTSCacheMixin
from pipecat.services.tts_cache.models import (
    CachedAudioChunk,
    CachedTTSResponse,
    CachedWordTimestamp,
)

__all__ = [
    "TTSCacheMixin",
    "CacheBackend",
    "MemoryCacheBackend",
    "CachedAudioChunk",
    "CachedWordTimestamp",
    "CachedTTSResponse",
    "generate_cache_key",
]
