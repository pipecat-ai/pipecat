#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the eval transport's per-connection query flags and virtual mic."""

import asyncio
import time
import types
import unittest

from pipecat.evals.transport import (
    AUDIO_CHUNK_MS,
    CAPTURE_AUDIO_QUERY_PARAM,
    RECORD_QUERY_PARAM,
    SKIP_TTS_QUERY_PARAM,
    EvalMicrophone,
    _query_flag,
    _query_value,
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
        self.assertTrue(
            _query_flag(_ws(path="/?capture_bot_audio=true"), CAPTURE_AUDIO_QUERY_PARAM)
        )
        self.assertFalse(_query_flag(_ws(path="/?skip_tts=true"), CAPTURE_AUDIO_QUERY_PARAM))

    def test_false_when_absent(self):
        self.assertFalse(_query_flag(_ws(path="/"), SKIP_TTS_QUERY_PARAM))

    def test_false_when_falsey_value(self):
        self.assertFalse(_query_flag(_ws(path="/?skip_tts=false"), SKIP_TTS_QUERY_PARAM))

    def test_false_when_no_path_at_all(self):
        self.assertFalse(_query_flag(_ws(), SKIP_TTS_QUERY_PARAM))


class TestEvalMicrophone(unittest.IsolatedAsyncioTestCase):
    """The virtual mic paces queued utterances at real time, silence otherwise."""

    SR = 16000
    CHUNK_BYTES = (SR * AUDIO_CHUNK_MS // 1000) * 2  # 20ms of 16kHz 16-bit mono

    async def _run_mic(self, mic, seconds):
        task = asyncio.create_task(mic.run(self.SR))
        try:
            await asyncio.sleep(seconds)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def test_paces_utterance_in_chunks_with_silence_around(self):
        pushed: list[tuple[bytes, int]] = []

        async def push(pcm, rate):
            pushed.append((pcm, rate))

        mic = EvalMicrophone(push)
        utterance = b"\x01\x02" * (self.SR // 10)  # 100ms -> five 20ms chunks
        mic.add_audio(utterance, self.SR)

        start = time.monotonic()
        await self._run_mic(mic, 0.3)
        elapsed = time.monotonic() - start

        speech = [pcm for pcm, _ in pushed if pcm != b"\x00" * len(pcm)]
        self.assertEqual(b"".join(speech), utterance)  # full utterance, in order
        self.assertTrue(all(len(pcm) == self.CHUNK_BYTES for pcm in speech))
        silence = b"\x00\x00" * (self.SR * AUDIO_CHUNK_MS // 1000)
        self.assertIn(silence, [pcm for pcm, _ in pushed])  # silence keeps flowing
        # Real-time pacing: ~300ms of run time emits ~15 frames, not hundreds.
        self.assertLess(len(pushed), int(elapsed / (AUDIO_CHUNK_MS / 1000)) + 5)

    async def test_reset_drops_queued_audio(self):
        pushed: list[bytes] = []

        async def push(pcm, rate):
            pushed.append(pcm)

        mic = EvalMicrophone(push)
        mic.add_audio(b"\x01\x02" * self.SR, self.SR)  # 1s queued
        mic.reset()
        await self._run_mic(mic, 0.1)

        self.assertTrue(all(pcm == b"\x00" * len(pcm) for pcm in pushed))  # only silence


class TestQueryValue(unittest.TestCase):
    def test_reads_url_decoded_value(self):
        ws = _ws(path="/?record=%2Ftmp%2Frec%2Fcap.wav")
        self.assertEqual(_query_value(ws, RECORD_QUERY_PARAM), "/tmp/rec/cap.wav")

    def test_none_when_absent(self):
        self.assertIsNone(_query_value(_ws(path="/?skip_tts=true"), RECORD_QUERY_PARAM))

    def test_none_when_empty(self):
        self.assertIsNone(_query_value(_ws(path="/?record="), RECORD_QUERY_PARAM))


if __name__ == "__main__":
    unittest.main()
