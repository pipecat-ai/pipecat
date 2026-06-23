#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the paced audio sender and ``audio_out_send_lead_secs`` in
:class:`BaseOutputTransport`."""

import unittest
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from pipecat.frames.frames import BotStoppedSpeakingFrame, StartFrame
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams

_INTERVAL = 0.04  # 16 kHz, 4x10ms chunks -> 40ms real playback per chunk


class _FakeClock:
    """Deterministic monotonic clock; ``sleep`` advances it (ideal sleep)."""

    def __init__(self, t: float = 1000.0):
        self.t = t
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.t

    async def sleep(self, duration: float):
        self.sleeps.append(duration)
        self.t += duration


def _make_transport(lead: float) -> BaseOutputTransport:
    transport = BaseOutputTransport(
        TransportParams(audio_out_enabled=True, audio_out_send_lead_secs=lead)
    )
    transport._send_interval = _INTERVAL
    transport._next_send_time = 0.0
    return transport


def _make_sender(lead: float):
    transport = _make_transport(lead)
    transport.push_frame = AsyncMock()
    sender = BaseOutputTransport.MediaSender(
        transport,
        destination=None,
        sample_rate=16000,
        audio_chunk_size=1280,
        params=transport._params,
    )
    return transport, sender


class TestSendLeadValidation(unittest.TestCase):
    def test_zero_is_allowed(self):
        self.assertEqual(
            TransportParams(audio_out_send_lead_secs=0.0).audio_out_send_lead_secs, 0.0
        )

    def test_positive_is_allowed(self):
        self.assertEqual(
            TransportParams(audio_out_send_lead_secs=1.5).audio_out_send_lead_secs, 1.5
        )

    def test_negative_is_rejected(self):
        with pytest.raises(ValidationError):
            TransportParams(audio_out_send_lead_secs=-0.1)


class TestSendIntervalComputation(unittest.IsolatedAsyncioTestCase):
    async def test_send_interval_is_real_chunk_duration(self):
        # audio_chunk_size = (16000/100)*1*2 * 4 = 1280 bytes; as 16-bit samples
        # that is 40ms of audio, so the send interval must be 40ms.
        transport = BaseOutputTransport(TransportParams(audio_out_enabled=True))
        await transport.start(StartFrame(audio_out_sample_rate=16000))
        self.assertAlmostEqual(transport._send_interval, _INTERVAL)

    async def test_send_interval_guarded_against_zero_sample_rate(self):
        # A zero sample rate must not raise ZeroDivisionError; the interval
        # simply stays at its default.
        transport = BaseOutputTransport(TransportParams(audio_out_enabled=True))
        await transport.start(StartFrame(audio_out_sample_rate=0))
        self.assertEqual(transport._send_interval, 0.0)


class TestPacing(unittest.IsolatedAsyncioTestCase):
    async def _run(self, transport, ticks, clock, stalls=None):
        stalls = stalls or {}
        with (
            patch("pipecat.transports.base_output.time.monotonic", clock.monotonic),
            patch("pipecat.transports.base_output.asyncio.sleep", clock.sleep),
        ):
            for i in range(ticks):
                if i in stalls:
                    clock.t += stalls[i]
                await transport._write_audio_sleep()

    async def test_default_no_lead_paces_on_grid_without_drift(self):
        transport = _make_transport(lead=0.0)
        clock = _FakeClock()
        await self._run(transport, ticks=5, clock=clock)

        # First chunk goes out immediately, the rest pace at exactly one
        # interval each (absolute grid, no drift).
        self.assertAlmostEqual(clock.sleeps[0], 0.0)
        for slept in clock.sleeps[1:]:
            self.assertAlmostEqual(slept, _INTERVAL)

    async def test_lead_bursts_to_build_buffer_then_paces(self):
        lead = 0.1
        transport = _make_transport(lead=lead)
        clock = _FakeClock()
        await self._run(transport, ticks=10, clock=clock)

        # The opening burst sends several chunks with no sleep to build the
        # lead, then settles into paced sends.
        self.assertAlmostEqual(clock.sleeps[0], 0.0)
        self.assertGreaterEqual(sum(1 for s in clock.sleeps if s == 0.0), 2)
        self.assertGreater(clock.sleeps[-1], 0.0)

        # In steady state the buffer sits within [lead, lead+interval).
        buffered = transport._next_send_time - clock.t
        self.assertGreaterEqual(buffered, lead - 1e-9)
        self.assertLess(buffered, lead + _INTERVAL)

    async def test_stall_shorter_than_lead_is_absorbed_without_reset(self):
        lead = 0.1
        transport = _make_transport(lead=lead)
        clock = _FakeClock()
        # Build a steady lead first.
        await self._run(transport, ticks=8, clock=clock)
        deadline_before = transport._next_send_time

        # An 80ms stall (< 100ms lead) right before the next send.
        await self._run(transport, ticks=1, clock=clock, stalls={0: 0.08})

        # The schedule advanced on the grid (deadline += interval), i.e. it was
        # NOT re-anchored to the wall clock: the lead absorbed the stall.
        self.assertAlmostEqual(transport._next_send_time, deadline_before + _INTERVAL)

    async def test_stall_longer_than_lead_resets_schedule(self):
        lead = 0.1
        transport = _make_transport(lead=lead)
        clock = _FakeClock()
        await self._run(transport, ticks=8, clock=clock)

        # A 200ms stall (> 100ms lead): the buffer underran, so the schedule
        # re-anchors to the wall clock and the lead is lost.
        await self._run(transport, ticks=1, clock=clock, stalls={0: 0.2})
        self.assertAlmostEqual(transport._next_send_time, clock.t + _INTERVAL)


class TestDrainAndCompensation(unittest.IsolatedAsyncioTestCase):
    async def test_drain_waits_for_buffered_audio(self):
        transport, sender = _make_sender(lead=0.1)
        clock = _FakeClock()
        transport._next_send_time = clock.t + 0.07  # 70ms still buffered
        with (
            patch("pipecat.transports.base_output.time.monotonic", clock.monotonic),
            patch("pipecat.transports.base_output.asyncio.sleep", clock.sleep),
        ):
            await sender._drain_send_lead()
        self.assertEqual(len(clock.sleeps), 1)
        self.assertAlmostEqual(clock.sleeps[0], 0.07)

    async def test_drain_is_noop_without_lead(self):
        transport, sender = _make_sender(lead=0.0)
        clock = _FakeClock()
        transport._next_send_time = clock.t + 0.07
        with (
            patch("pipecat.transports.base_output.time.monotonic", clock.monotonic),
            patch("pipecat.transports.base_output.asyncio.sleep", clock.sleep),
        ):
            await sender._drain_send_lead()
        self.assertEqual(clock.sleeps, [])

    async def test_drain_is_noop_when_buffer_already_empty(self):
        transport, sender = _make_sender(lead=0.1)
        clock = _FakeClock()
        transport._next_send_time = clock.t - 0.5  # deadline already passed
        with (
            patch("pipecat.transports.base_output.time.monotonic", clock.monotonic),
            patch("pipecat.transports.base_output.asyncio.sleep", clock.sleep),
        ):
            await sender._drain_send_lead()
        self.assertEqual(clock.sleeps, [])

    async def test_bot_stopped_speaking_drains_lead_before_signalling(self):
        transport, sender = _make_sender(lead=0.1)
        sender._bot_speaking = True
        clock = _FakeClock()
        transport._next_send_time = clock.t + 0.07
        with (
            patch("pipecat.transports.base_output.time.monotonic", clock.monotonic),
            patch("pipecat.transports.base_output.asyncio.sleep", clock.sleep),
        ):
            await sender._bot_stopped_speaking()

        # It drained the residual lead, then emitted bot-stopped.
        self.assertAlmostEqual(clock.sleeps[0], 0.07)
        self.assertFalse(sender._bot_speaking)
        pushed = [c.args[0] for c in transport.push_frame.call_args_list]
        self.assertTrue(any(isinstance(f, BotStoppedSpeakingFrame) for f in pushed))

    async def test_bot_stopped_speaking_no_drain_without_lead(self):
        transport, sender = _make_sender(lead=0.0)
        sender._bot_speaking = True
        clock = _FakeClock()
        transport._next_send_time = clock.t + 0.07
        with (
            patch("pipecat.transports.base_output.time.monotonic", clock.monotonic),
            patch("pipecat.transports.base_output.asyncio.sleep", clock.sleep),
        ):
            await sender._bot_stopped_speaking()
        self.assertEqual(clock.sleeps, [])
        self.assertFalse(sender._bot_speaking)


if __name__ == "__main__":
    unittest.main()
