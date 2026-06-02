#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the eval transport's skip-TTS connection signal."""

import types
import unittest

from pipecat.evals.transport import _connection_wants_skip_tts


def _ws(path=None, request_path=None):
    """A minimal stand-in for a websockets connection object."""
    request = types.SimpleNamespace(path=request_path) if request_path is not None else None
    return types.SimpleNamespace(path=path, request=request)


class TestConnectionWantsSkipTTS(unittest.TestCase):
    def test_true_via_legacy_path(self):
        self.assertTrue(_connection_wants_skip_tts(_ws(path="/?skip_tts=true")))

    def test_true_via_request_path(self):
        self.assertTrue(_connection_wants_skip_tts(_ws(path=None, request_path="/?skip_tts=1")))

    def test_accepts_yes_and_mixed_case(self):
        self.assertTrue(_connection_wants_skip_tts(_ws(path="/?skip_tts=YES")))

    def test_false_when_absent(self):
        self.assertFalse(_connection_wants_skip_tts(_ws(path="/")))

    def test_false_when_falsey_value(self):
        self.assertFalse(_connection_wants_skip_tts(_ws(path="/?skip_tts=false")))

    def test_false_when_no_path_at_all(self):
        self.assertFalse(_connection_wants_skip_tts(_ws()))


if __name__ == "__main__":
    unittest.main()
