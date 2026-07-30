#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A no-op audio mixer (pass-through) used when ambient noise is disabled.

It exists so the output transport streams audio continuously (mixing silence)
even when there's no TTS audio. It must implement the full BaseAudioMixer
interface — start(sample_rate), stop(), process_frame(), mix() — otherwise it
stays abstract and the pipeline crashes with
"Can't instantiate abstract class SilenceAudioMixer".
"""

from pipecat.audio.mixers.base_audio_mixer import BaseAudioMixer
from pipecat.frames.frames import MixerControlFrame


class SilenceAudioMixer(BaseAudioMixer):
    """Audio mixer that produces silence — pass-through with no extra mixing.

    Note:
        This mixer is a candidate for ``is_passthrough = True``: it returns its
        input unchanged, so the output transport's continuous send path (a
        full-rate synthesize/mix/write loop on every leg, even an idle one) buys
        nothing here. Deliberately NOT enabled yet — flipping it switches every
        leg that installs this mixer from the mixer path to the no-mixer path,
        which changes whether silence is continuously transmitted, and some
        RTP/telephony configurations rely on that to keep a stream alive. Gate it
        behind a low-concurrency validation run of its own; do not bundle it with
        other changes.
    """

    def __init__(self):
        """Initialize the silence audio mixer."""
        self._sample_rate = 0

    async def start(self, sample_rate: int):
        """Initialize the mixer with the output transport sample rate."""
        self._sample_rate = sample_rate

    async def stop(self):
        """Clean up the mixer when the output transport stops."""
        pass

    async def process_frame(self, frame: MixerControlFrame):
        """Process mixer control frames (no-op for the silence mixer)."""
        pass

    async def mix(self, audio: bytes) -> bytes:
        """Pass transport audio through unchanged."""
        return audio
