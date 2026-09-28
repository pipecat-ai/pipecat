#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import struct
import unittest

from pipecat.audio.utils import is_silence


class TestIsSilencePcmRange(unittest.TestCase):
    def test_full_negative_scale_is_not_silence(self):
        for samples in ([-32768], [0, -32768, 0], [-20, -32768, 20]):
            with self.subTest(samples=samples):
                pcm = struct.pack(f"<{len(samples)}h", *samples)
                self.assertFalse(is_silence(pcm))

    def test_threshold_and_other_signed_samples(self):
        for samples, expected in (
            ([0, 0], True),
            ([-20, 20], True),
            ([-21, 0], False),
            ([0, 21], False),
            ([-32767, 0], False),
            ([0, 32767], False),
        ):
            with self.subTest(samples=samples):
                pcm = struct.pack(f"<{len(samples)}h", *samples)
                self.assertEqual(is_silence(pcm), expected)


if __name__ == "__main__":
    unittest.main()
