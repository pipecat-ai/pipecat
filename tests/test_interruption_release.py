#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

import numpy as np

from pipecat.audio.utils import apply_fade_out_to_pcm16
from pipecat.frames.frames import OutputAudioRawFrame
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import InterruptionReleaseMode, TransportParams
from pipecat.utils.frame_queue import FrameQueue


class TestInterruptionReleaseParams(unittest.TestCase):
    def test_default_transport_params(self):
        p = TransportParams()
        self.assertIs(p.interruption_release_mode, InterruptionReleaseMode.IMMEDIATE_CUT)
        self.assertEqual(p.interruption_max_release_drain_ms, 60)
        self.assertEqual(p.interruption_fade_out_ms, 20)

    def test_bounded_drain_mode(self):
        p = TransportParams(
            interruption_release_mode=InterruptionReleaseMode.BOUNDED_DRAIN,
            interruption_max_release_drain_ms=40,
            interruption_fade_out_ms=10,
        )
        self.assertIs(p.interruption_release_mode, InterruptionReleaseMode.BOUNDED_DRAIN)
        self.assertEqual(p.interruption_max_release_drain_ms, 40)
        self.assertEqual(p.interruption_fade_out_ms, 10)


class TestApplyFadeOutPcm16(unittest.TestCase):
    def test_fade_out_lowers_end_of_tail(self):
        sample_rate = 8000
        n = 200
        pcm = (np.ones(n, dtype=np.int16) * 10000).tobytes()
        out = apply_fade_out_to_pcm16(
            pcm,
            sample_rate=sample_rate,
            num_channels=1,
            fade_out_ms=20,
        )
        self.assertEqual(len(out), len(pcm))
        tail = np.frombuffer(out[-4:], dtype=np.int16)
        head = np.frombuffer(out[:4], dtype=np.int16)
        self.assertLess(np.abs(tail).max(), np.abs(head).max())


class TestCollectBoundedDrainPcm(unittest.TestCase):
    def test_respects_max_bytes_buffer_then_queue(self):
        params = TransportParams()
        t = BaseOutputTransport(params=params, name="test-out")
        t._sample_rate = 16000
        t._audio_chunk_size = 320
        ms = BaseOutputTransport.MediaSender(
            t, destination=None, sample_rate=16000, audio_chunk_size=320, params=params
        )
        old_buf = bytearray(b"abcdefghij")
        q = FrameQueue()
        q.put_nowait(
            OutputAudioRawFrame(
                audio=b"klmnopqrst",
                sample_rate=16000,
                num_channels=1,
            )
        )
        out = ms._collect_bounded_drain_pcm(old_buf, q, max_bytes=15)
        self.assertEqual(out, b"abcdefghijklmno")
        self.assertTrue(q.empty())

    def test_empty_buffer_and_queue(self):
        params = TransportParams()
        t = BaseOutputTransport(params=params, name="test-out-2")
        t._sample_rate = 16000
        t._audio_chunk_size = 320
        ms = BaseOutputTransport.MediaSender(
            t, destination=None, sample_rate=16000, audio_chunk_size=320, params=params
        )
        out = ms._collect_bounded_drain_pcm(bytearray(), FrameQueue(), max_bytes=10)
        self.assertEqual(out, b"")
