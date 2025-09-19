#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Silence audio mixer implementation."""

from pipecat.audio.mixers.base_audio_mixer import BaseAudioMixer
from pipecat.frames.frames import MixerControlFrame


class SilenceAudioMixer(BaseAudioMixer):
    """A simple audio mixer that just passes through audio unchanged.

    This mixer is used to enable continuous audio streaming in the output transport.
    When a mixer is present, the BaseOutputTransport will continuously send audio
    frames even when there's no TTS audio, by mixing silence.
    """

    def __init__(self):
        """Initialize the silence audio mixer."""
        self._enabled = True

    async def start(self, sample_rate: int):
        """Initialize the mixer with the output sample rate."""
        self._sample_rate = sample_rate

    async def stop(self):
        """Stop the mixer."""
        pass

    async def process_frame(self, frame: MixerControlFrame):
        """Process mixer control frames."""
        # Could handle enable/disable here if needed
        pass

    async def mix(self, audio: bytes) -> bytes:
        """Mix audio - in this case, just pass through unchanged."""
        return audio
