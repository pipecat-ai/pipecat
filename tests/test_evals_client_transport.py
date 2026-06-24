#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the eval harness's mic-like output transport."""

import asyncio
import types
import unittest

from pipecat.evals.client_transport import MIC_FRAME_S, EvalMicOutputTransport
from pipecat.transports.websocket.client import WebsocketClientParams


def _fake_session():
    """A session stand-in that reports an open connection."""
    return types.SimpleNamespace(is_closing=False, is_connected=True)


class TestEvalMicOutput(unittest.IsolatedAsyncioTestCase):
    """The output streams queued audio at real time, silence otherwise."""

    SR = 16000
    CHUNK_BYTES = int(SR * MIC_FRAME_S) * 2  # one ~20ms frame, 16-bit mono

    async def _run(self, out, seconds):
        task = asyncio.create_task(out._mic_task_handler())
        try:
            await asyncio.sleep(seconds)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def test_paces_queued_audio_then_silence(self):
        out = EvalMicOutputTransport(
            None, _fake_session(), WebsocketClientParams(audio_out_enabled=True)
        )
        out._sample_rate = self.SR  # set by start(); skip the transport lifecycle

        sent: list[bytes] = []

        async def capture(frame):  # capture frames instead of serializing/sending
            sent.append(frame.audio)

        async def noop(*args, **kwargs):  # the unlinked processor has no downstream
            pass

        out._send_mic_frame = capture
        out.push_frame = noop

        utterance = b"\x01\x02" * (self.SR // 10)  # 100ms -> five 20ms chunks
        out._mic_pcm.extend(utterance)

        await self._run(out, 0.25)

        speech = [pcm for pcm in sent if pcm != b"\x00" * len(pcm)]
        self.assertEqual(b"".join(speech), utterance)  # full utterance, in order
        self.assertTrue(all(len(pcm) == self.CHUNK_BYTES for pcm in speech))
        self.assertIn(b"\x00" * self.CHUNK_BYTES, sent)  # silence keeps flowing
        # Real-time pacing: ~0.25s emits ~12 frames, not hundreds.
        self.assertLess(len(sent), int(0.25 / MIC_FRAME_S) + 5)


if __name__ == "__main__":
    unittest.main()
