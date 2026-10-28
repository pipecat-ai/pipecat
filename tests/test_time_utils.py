#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.utils.time import nanoseconds_to_str


class TestNanosecondsToStr(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(nanoseconds_to_str(0), "0:00:00.000000")

    def test_keeps_exact_microseconds(self):
        # Values that float seconds can't represent exactly.
        self.assertEqual(nanoseconds_to_str(2_300_000_000), "0:00:02.300000")
        self.assertEqual(nanoseconds_to_str(1_000_001_000), "0:00:01.000001")
        self.assertEqual(nanoseconds_to_str(3_600_000_001_000), "1:00:00.000001")

    def test_truncates_sub_microsecond_part(self):
        self.assertEqual(nanoseconds_to_str(1_234_567_891), "0:00:01.234567")
        self.assertEqual(nanoseconds_to_str(999), "0:00:00.000000")

    def test_hours_minutes_seconds(self):
        ns = ((2 * 3600) + (5 * 60) + 7) * 1_000_000_000 + 89_000
        self.assertEqual(nanoseconds_to_str(ns), "2:05:07.000089")

    def test_long_durations(self):
        self.assertEqual(nanoseconds_to_str(100 * 3600 * 1_000_000_000), "100:00:00.000000")


if __name__ == "__main__":
    unittest.main()
