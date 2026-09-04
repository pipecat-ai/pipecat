#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the eval harness's client output transport."""

import asyncio
import types
import unittest
from unittest.mock import patch

from pipecat.evals import client_transport
from pipecat.evals.client_transport import FRAME_S, EvalHarnessOutputTransport, _RecorderTrack
from pipecat.transports.websocket.client import WebsocketClientParams


def _fake_session():
    """A session stand-in that reports an open connection."""
    return types.SimpleNamespace(is_closing=False, is_connected=True)


class TestEvalHarnessOutput(unittest.IsolatedAsyncioTestCase):
    """The output streams queued audio at real time, silence otherwise."""

    SR = 16000
    CHUNK_BYTES = int(SR * FRAME_S) * 2  # one ~40ms frame, 16-bit mono

    async def _run(self, out, seconds):
        task = asyncio.create_task(out._send_task_handler())
        try:
            await asyncio.sleep(seconds)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def test_paces_queued_audio_then_silence(self):
        out = EvalHarnessOutputTransport(
            None, _fake_session(), WebsocketClientParams(audio_out_enabled=True)
        )
        out._sample_rate = self.SR  # set by start(); skip the transport lifecycle

        sent: list[bytes] = []

        async def capture(frame):  # capture frames instead of serializing/sending
            sent.append(frame.audio)

        async def noop(*args, **kwargs):  # the unlinked processor has no downstream
            pass

        out._send_frame = capture
        out.push_frame = noop

        utterance = b"\x01\x02" * (self.SR * 12 // 100)  # 120ms -> three 40ms chunks
        out._pending.extend(utterance)

        await self._run(out, 0.25)

        speech = [pcm for pcm in sent if pcm != b"\x00" * len(pcm)]
        self.assertEqual(b"".join(speech), utterance)  # full utterance, in order
        self.assertTrue(all(len(pcm) == self.CHUNK_BYTES for pcm in speech))
        self.assertIn(b"\x00" * self.CHUNK_BYTES, sent)  # silence keeps flowing
        # Real-time pacing: ~0.25s emits ~6 frames at 40ms, not hundreds.
        self.assertLess(len(sent), int(0.25 / FRAME_S) + 5)


class TestRecorderTrack(unittest.IsolatedAsyncioTestCase):
    """A track lays its chunks out on a playout timeline; silence only for real pauses."""

    SR = 16000
    CHUNK_S = 0.04
    CHUNK = b"\x01\x00" * int(SR * CHUNK_S)  # 40ms of non-silent samples

    def _track(self, arrivals, *, paced=False) -> _RecorderTrack:
        """A track fed one CHUNK at each arrival time (seconds)."""
        track = _RecorderTrack(paced=paced)
        with patch.object(client_transport, "time") as fake_time:
            for at in arrivals:
                fake_time.monotonic.return_value = at
                track.add(self.CHUNK, self.SR)
        return track

    @staticmethod
    def _paced(start: float, count: int, step: float = 0.04) -> list[float]:
        return [start + i * step for i in range(count)]

    async def _silence_s(self, track) -> float:
        """Seconds of inserted silence: the output beyond the chunks themselves."""
        out = await track.rendered(self.SR, track.first)
        return (len(out) - len(track._chunks) * len(self.CHUNK)) / (self.SR * 2)

    async def test_real_time_stream_is_contiguous(self):
        jitter = [0, 0.01, -0.005, 0.008, 0, -0.01, 0.012, 0]
        arrivals = [t + j for t, j in zip(self._paced(10.0, 8), jitter)]
        out = await self._track(arrivals, paced=True).rendered(self.SR, 10.0)
        self.assertEqual(out, self.CHUNK * 8)

    async def test_pause_between_turns_is_silence(self):
        arrivals = self._paced(10.0, 5) + self._paced(10.0 + 5 * 0.04 + 1.0, 5)
        for paced in (False, True):
            self.assertAlmostEqual(
                await self._silence_s(self._track(arrivals, paced=paced)), 1.0, delta=0.001
            )

    async def test_receiver_stall_on_paced_track_stays_gapless(self):
        # 0.5s hole, then the held-up chunks arrive in a burst, then real time again.
        burst = [10.7 + i * 0.001 for i in range(12)]  # 0.48s of audio in 12ms
        arrivals = self._paced(10.0, 5) + burst + self._paced(burst[-1] + 0.04, 5)
        track = self._track(arrivals, paced=True)
        self.assertEqual(await track.rendered(self.SR, 10.0), self.CHUNK * len(arrivals))

    async def test_stall_leaves_only_what_the_burst_does_not_recover(self):
        # The source paused 0.52s (resuming at 10.72) and the receiver stalled until
        # 11.2: the 0.48s of audio sent meanwhile arrives in a burst, and real time
        # resumes from the end of the burst. Only the pause is silence.
        burst = [11.2 + i * 0.0005 for i in range(12)]
        arrivals = self._paced(10.0, 5) + burst + self._paced(burst[-1] + 0.001, 5)
        self.assertAlmostEqual(
            await self._silence_s(self._track(arrivals, paced=True)), 0.52, delta=0.01
        )

    async def test_unpaced_pause_counts_from_the_end_of_playout(self):
        # A fast source (the user TTS) produces a turn in a burst; its playout
        # still takes the audio's duration, and the pause runs from that end.
        arrivals = [10.0 + i * 0.001 for i in range(10)] + [20.0]
        self.assertAlmostEqual(
            await self._silence_s(self._track(arrivals)), 10.0 - 10 * 0.04, delta=0.001
        )

    async def test_lead_silence_aligns_to_the_recording_start(self):
        track = self._track(self._paced(12.0, 2))
        out = await track.rendered(self.SR, 10.0)
        self.assertEqual(out, b"\x00" * (2 * self.SR * 2) + self.CHUNK * 2)


if __name__ == "__main__":
    unittest.main()
