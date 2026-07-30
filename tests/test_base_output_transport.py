#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for interruption handling in :class:`BaseOutputTransport`."""

import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from pipecat.audio.mixers.base_audio_mixer import BaseAudioMixer
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    CancelFrame,
    HeartbeatFrame,
    InterruptionFrame,
    MixerControlFrame,
    OutputAudioRawFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams


class _PassthroughMixer(BaseAudioMixer):
    """Minimal mixer that returns the input audio unchanged."""

    async def start(self, sample_rate: int):
        pass

    async def stop(self):
        pass

    async def process_frame(self, frame: MixerControlFrame):
        pass

    async def mix(self, audio: bytes) -> bytes:
        return audio


async def _make_transport(mixer: BaseAudioMixer | None = None) -> BaseOutputTransport:
    params = TransportParams(audio_out_enabled=True, audio_out_mixer=mixer)
    transport = BaseOutputTransport(params)
    transport.push_frame = AsyncMock()
    transport.write_audio_frame = AsyncMock(return_value=True)

    task_manager = TaskManager()
    task_manager.setup(TaskManagerParams(loop=asyncio.get_event_loop()))
    await transport.setup(
        FrameProcessorSetup(
            clock=SystemClock(),
            task_manager=task_manager,
            pipeline_worker=SimpleNamespace(app_resources=None),  # type: ignore[arg-type]
        )
    )
    start_frame = StartFrame(audio_out_sample_rate=16000)
    await transport.process_frame(start_frame, FrameDirection.DOWNSTREAM)
    await transport.set_transport_ready(start_frame)
    return transport


class TestBaseOutputTransportInterruptions(unittest.IsolatedAsyncioTestCase):
    async def _make_transport(self, mixer: BaseAudioMixer | None = None) -> BaseOutputTransport:
        return await _make_transport(mixer)

    async def test_interruption_with_mixer_keeps_audio_task_and_mixer_output(self):
        transport = await self._make_transport(mixer=_PassthroughMixer())
        try:
            sender = transport._media_senders[None]
            task_before = sender._audio_task
            self.assertIsNotNone(task_before)

            # Mixer-only frames flow while the queue is empty.
            await asyncio.sleep(0.1)
            self.assertGreater(transport.write_audio_frame.call_count, 0)

            await transport.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)

            # Same task object: not cancelled and not recreated.
            self.assertIs(sender._audio_task, task_before)
            self.assertFalse(task_before.cancelled())

            # Mixer frames keep flowing across the interruption.
            count_after_interruption = transport.write_audio_frame.call_count
            await asyncio.sleep(0.1)
            self.assertGreater(transport.write_audio_frame.call_count, count_after_interruption)
        finally:
            await transport.cancel(CancelFrame())

    async def test_interruption_without_mixer_recreates_audio_task(self):
        transport = await self._make_transport(mixer=None)
        try:
            sender = transport._media_senders[None]
            task_before = sender._audio_task
            self.assertIsNotNone(task_before)

            await transport.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)

            self.assertIsNot(sender._audio_task, task_before)
            self.assertIsNotNone(sender._audio_task)
        finally:
            await transport.cancel(CancelFrame())

    async def test_interruption_with_mixer_still_discards_queued_bot_audio(self):
        transport = await self._make_transport(mixer=_PassthroughMixer())
        try:
            sender = transport._media_senders[None]

            # Pause the audio task by patching write_audio_frame with a slow
            # write, so the queued bot audio can't be consumed before the
            # interruption arrives.
            write_started = asyncio.Event()
            release_write = asyncio.Event()

            async def slow_write(frame):
                write_started.set()
                await release_write.wait()
                return True

            transport.write_audio_frame = AsyncMock(side_effect=slow_write)
            # A timeout here IS the assertion: if the sender ever stops writing
            # continuously this must fail, not hang the suite (there is no
            # global pytest timeout in this repo).
            await asyncio.wait_for(write_started.wait(), timeout=5)

            # Queue bot audio (one full chunk) behind the in-flight write.
            bot_audio = OutputAudioRawFrame(
                audio=b"\x01\x02" * (sender.audio_chunk_size // 2),
                sample_rate=sender.sample_rate,
                num_channels=1,
            )
            await transport.process_frame(bot_audio, FrameDirection.DOWNSTREAM)
            self.assertFalse(sender._audio_queue.empty())

            await transport.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)

            # The queued bot audio was dropped by the reset.
            self.assertTrue(sender._audio_queue.empty())
            release_write.set()
        finally:
            await transport.cancel(CancelFrame())


class TestBaseOutputTransportHeartbeatRouting(unittest.IsolatedAsyncioTestCase):
    """Heartbeats must not be routed through the paced media path.

    Regression coverage for the run-60 false-stall signal: ``HeartbeatFrame`` is
    a plain ``ControlFrame``, so before this fix it fell through to
    ``MediaSender.handle_sync_frame`` and sat in the audio queue behind every
    queued bot-audio chunk. Since that queue is consumed at 1x realtime, the
    heartbeat's measured traversal latency was the playout backlog: a single ~9s
    utterance was enough to trip a 10s heartbeat monitor on a perfectly healthy
    pipeline.
    """

    async def _fill_audio_queue(self, transport: BaseOutputTransport, chunks: int) -> None:
        sender = transport._media_senders[None]
        for _ in range(chunks):
            await transport.process_frame(
                OutputAudioRawFrame(
                    audio=b"\x01\x02" * (sender.audio_chunk_size // 2),
                    sample_rate=sender.sample_rate,
                    num_channels=1,
                ),
                FrameDirection.DOWNSTREAM,
            )

    async def test_heartbeat_bypasses_backlogged_audio_queue(self):
        transport = await _make_transport(mixer=_PassthroughMixer())
        try:
            sender = transport._media_senders[None]

            # Block the audio task mid-write so nothing drains: this is the
            # backlog condition that used to delay heartbeats.
            release_write = asyncio.Event()
            write_started = asyncio.Event()

            async def blocked_write(frame):
                write_started.set()
                await release_write.wait()
                return True

            transport.write_audio_frame = AsyncMock(side_effect=blocked_write)
            # A timeout here IS the assertion: if the sender ever stops writing
            # continuously this must fail, not hang the suite (there is no
            # global pytest timeout in this repo).
            await asyncio.wait_for(write_started.wait(), timeout=5)

            await self._fill_audio_queue(transport, chunks=50)
            queued_before = sender._audio_queue.qsize()
            self.assertGreaterEqual(queued_before, 50)

            heartbeat = HeartbeatFrame(timestamp=0)
            started = time.monotonic()
            await transport.process_frame(heartbeat, FrameDirection.DOWNSTREAM)
            elapsed = time.monotonic() - started

            # Delivered straight to the sink, ahead of the whole backlog.
            pushed = [call.args[0] for call in transport.push_frame.call_args_list]
            self.assertIn(heartbeat, pushed)
            self.assertLess(elapsed, 0.1)

            # ...and it never entered the paced queue, which is still backlogged.
            self.assertFalse(sender._audio_queue.has_frame(HeartbeatFrame))
            self.assertGreaterEqual(sender._audio_queue.qsize(), queued_before)

            release_write.set()
        finally:
            await transport.cancel(CancelFrame())

    async def test_heartbeat_never_reaches_handle_sync_frame(self):
        transport = await _make_transport(mixer=_PassthroughMixer())
        try:
            sender = transport._media_senders[None]
            sync_frames = []
            original_handle_sync_frame = sender.handle_sync_frame

            async def spy(frame):
                sync_frames.append(frame)
                await original_handle_sync_frame(frame)

            sender.handle_sync_frame = spy

            for _ in range(3):
                await transport.process_frame(
                    HeartbeatFrame(timestamp=0), FrameDirection.DOWNSTREAM
                )

            self.assertEqual(sync_frames, [])
        finally:
            await transport.cancel(CancelFrame())

    async def test_upstream_heartbeat_is_still_forwarded(self):
        transport = await _make_transport(mixer=_PassthroughMixer())
        try:
            heartbeat = HeartbeatFrame(timestamp=0)
            await transport.process_frame(heartbeat, FrameDirection.UPSTREAM)
            pushed = [(call.args[0], call.args[1]) for call in transport.push_frame.call_args_list]
            self.assertIn((heartbeat, FrameDirection.UPSTREAM), pushed)
        finally:
            await transport.cancel(CancelFrame())


class TestBaseOutputTransportWriteWatchdog(unittest.IsolatedAsyncioTestCase):
    """`seconds_since_last_output_write` replaces what the heartbeat used to prove.

    Routing heartbeats around the media path (above) means they no longer prove
    the output audio task is alive. This watchdog covers that gap without the
    false positives, because a mixer-backed sender writes on a fixed cadence
    whether or not anyone is speaking.
    """

    async def test_watchdog_is_fresh_while_the_mixer_sender_writes(self):
        transport = await _make_transport(mixer=_PassthroughMixer())
        try:
            await asyncio.sleep(0.1)
            self.assertGreater(transport.write_audio_frame.call_count, 0)
            self.assertLess(transport.seconds_since_last_output_write, 0.5)
        finally:
            await transport.cancel(CancelFrame())

    async def test_watchdog_grows_when_the_audio_task_wedges(self):
        transport = await _make_transport(mixer=_PassthroughMixer())
        try:
            await asyncio.sleep(0.05)

            release_write = asyncio.Event()

            async def wedged_write(frame):
                await release_write.wait()
                return True

            transport.write_audio_frame = AsyncMock(side_effect=wedged_write)

            await asyncio.sleep(0.05)
            first = transport.seconds_since_last_output_write
            await asyncio.sleep(0.25)
            second = transport.seconds_since_last_output_write

            self.assertGreater(second, first)
            self.assertGreater(second, 0.2)

            release_write.set()
        finally:
            await transport.cancel(CancelFrame())

    async def test_watchdog_reports_zero_without_a_continuous_writer(self):
        # No mixer: the audio task blocks on an empty queue by design, so an
        # idle sender must not be reported as stale.
        transport = await _make_transport(mixer=None)
        try:
            await asyncio.sleep(0.15)
            self.assertEqual(transport.seconds_since_last_output_write, 0.0)
        finally:
            await transport.cancel(CancelFrame())

    async def test_watchdog_ignores_failed_writes(self):
        transport = await _make_transport(mixer=_PassthroughMixer())
        try:
            await asyncio.sleep(0.05)
            transport.write_audio_frame = AsyncMock(return_value=False)
            await asyncio.sleep(0.2)
            # Writes are being attempted but none succeed: that is a wedge.
            self.assertGreater(transport.seconds_since_last_output_write, 0.15)
        finally:
            await transport.cancel(CancelFrame())


class _SilenceOnlyMixer(_PassthroughMixer):
    """A mixer that declares it contributes nothing to the outgoing audio."""

    @property
    def is_passthrough(self) -> bool:
        return True


class TestBaseOutputTransportPassthroughMixer(unittest.IsolatedAsyncioTestCase):
    """A mixer that adds nothing should not force the continuous send path.

    Configuring any mixer puts the sender on a full-rate synthesize/mix/write
    loop and makes every interruption drain the audio queue in place. That is
    right for a mixer that generates audio, but a silence mixer installed
    unconditionally when ambient audio is off pays it on every idle leg.
    """

    async def test_passthrough_mixer_does_not_write_while_idle(self):
        transport = await _make_transport(mixer=_SilenceOnlyMixer())
        try:
            await asyncio.sleep(0.15)
            self.assertEqual(transport.write_audio_frame.call_count, 0)
        finally:
            await transport.cancel(CancelFrame())

    async def test_audio_generating_mixer_still_writes_while_idle(self):
        transport = await _make_transport(mixer=_PassthroughMixer())
        try:
            await asyncio.sleep(0.15)
            self.assertGreater(transport.write_audio_frame.call_count, 0)
        finally:
            await transport.cancel(CancelFrame())

    async def test_passthrough_mixer_interruption_recreates_the_audio_task(self):
        transport = await _make_transport(mixer=_SilenceOnlyMixer())
        try:
            sender = transport._media_senders[None]
            task_before = sender._audio_task

            await transport.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)

            # Same branch as the no-mixer case: cancel and recreate, rather than
            # resetting the queue to keep mixer-only output flowing.
            self.assertIsNot(sender._audio_task, task_before)
            self.assertIsNotNone(sender._audio_task)
        finally:
            await transport.cancel(CancelFrame())

    async def test_passthrough_mixer_still_delivers_queued_audio(self):
        transport = await _make_transport(mixer=_SilenceOnlyMixer())
        try:
            sender = transport._media_senders[None]
            await transport.process_frame(
                OutputAudioRawFrame(
                    audio=b"\x01\x02" * (sender.audio_chunk_size // 2),
                    sample_rate=sender.sample_rate,
                    num_channels=1,
                ),
                FrameDirection.DOWNSTREAM,
            )
            await asyncio.sleep(0.1)
            self.assertGreater(transport.write_audio_frame.call_count, 0)
        finally:
            await transport.cancel(CancelFrame())

    async def test_watchdog_is_inert_for_a_passthrough_mixer(self):
        transport = await _make_transport(mixer=_SilenceOnlyMixer())
        try:
            await asyncio.sleep(0.15)
            # No continuous writer, so no wedge evidence either way.
            self.assertEqual(transport.seconds_since_last_output_write, 0.0)
        finally:
            await transport.cancel(CancelFrame())
