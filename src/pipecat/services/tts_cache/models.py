#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Data models for TTS caching."""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CachedAudioChunk:
    """Single chunk of cached TTS audio."""

    audio: bytes
    sample_rate: int
    num_channels: int
    pts: Optional[int] = None


@dataclass
class CachedWordTimestamp:
    """Word with timing information for replay."""

    word: str
    timestamp: float


@dataclass
class CachedTTSResponse:
    """Complete cached TTS response."""

    audio_chunks: List[CachedAudioChunk]
    sample_rate: int
    num_channels: int
    word_timestamps: Optional[List[CachedWordTimestamp]] = None
    total_duration_s: float = 0.0
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_audio_bytes(self) -> int:
        """Calculate total audio size in bytes."""
        return sum(len(chunk.audio) for chunk in self.audio_chunks)
