#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.audio.dtmf import utils
from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.audio.dtmf.utils import load_dtmf_audio


class TestLoadDTMFAudio(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # The module caches decoded/resampled audio in process-global state.
        # Reset it so each test starts from a clean cache.
        utils.__DTMF_AUDIO__.clear()
        utils.__DTMF_RESAMPLER__ = None

    async def test_cache_is_keyed_by_sample_rate(self):
        """Different sample rates for the same button must not collide in the cache."""
        audio_8k = await load_dtmf_audio(KeypadEntry.ONE, sample_rate=8000)
        audio_16k = await load_dtmf_audio(KeypadEntry.ONE, sample_rate=16000)

        # 16 kHz PCM of the same tone has roughly twice the bytes of 8 kHz. With a
        # button-only cache key the 8 kHz audio was returned for the 16 kHz request,
        # so the lengths matched and this assertion failed.
        self.assertGreater(len(audio_16k), len(audio_8k) * 1.8)

    async def test_cached_rate_returns_identical_bytes(self):
        """Re-requesting a previously loaded sample rate returns the same bytes."""
        first = await load_dtmf_audio(KeypadEntry.ONE, sample_rate=8000)
        # Load a different rate in between; it must not clobber the 8 kHz entry.
        await load_dtmf_audio(KeypadEntry.ONE, sample_rate=16000)
        again = await load_dtmf_audio(KeypadEntry.ONE, sample_rate=8000)
        self.assertEqual(first, again)
