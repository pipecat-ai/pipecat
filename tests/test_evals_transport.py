#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the eval transport's per-connection query flags."""

import types
import unittest

from pipecat.evals.transport import (
    CAPTURE_AUDIO_QUERY_PARAM,
    SKIP_TTS_QUERY_PARAM,
    _query_flag,
)


def _ws(path=None, request_path=None):
    """A minimal stand-in for a websockets connection object."""
    request = types.SimpleNamespace(path=request_path) if request_path is not None else None
    return types.SimpleNamespace(path=path, request=request)


class TestQueryFlag(unittest.TestCase):
    def test_true_via_legacy_path(self):
        self.assertTrue(_query_flag(_ws(path="/?skip_tts=true"), SKIP_TTS_QUERY_PARAM))

    def test_true_via_request_path(self):
        self.assertTrue(
            _query_flag(_ws(path=None, request_path="/?skip_tts=1"), SKIP_TTS_QUERY_PARAM)
        )

    def test_accepts_yes_and_mixed_case(self):
        self.assertTrue(_query_flag(_ws(path="/?skip_tts=YES"), SKIP_TTS_QUERY_PARAM))

    def test_capture_audio_flag(self):
        self.assertTrue(_query_flag(_ws(path="/?capture_audio=true"), CAPTURE_AUDIO_QUERY_PARAM))
        self.assertFalse(_query_flag(_ws(path="/?skip_tts=true"), CAPTURE_AUDIO_QUERY_PARAM))

    def test_false_when_absent(self):
        self.assertFalse(_query_flag(_ws(path="/"), SKIP_TTS_QUERY_PARAM))

    def test_false_when_falsey_value(self):
        self.assertFalse(_query_flag(_ws(path="/?skip_tts=false"), SKIP_TTS_QUERY_PARAM))

    def test_false_when_no_path_at_all(self):
        self.assertFalse(_query_flag(_ws(), SKIP_TTS_QUERY_PARAM))


if __name__ == "__main__":
    unittest.main()
