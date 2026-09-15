#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the SmallWebRTC transport client.

Covers app-message delivery in `SmallWebRTCClient.send_message` /
`SmallWebRTCConnection.send_app_message`:

1. **Pre-open buffering** — messages sent before the data channel is open
   (including before the peer connection is established) are queued and
   flushed, in order, once the channel opens. A channel created by the
   remote peer arrives from aiortc already open, so the flush must fire on
   channel arrival, not only on the "open" event.

2. **Closing discard** — messages sent while the connection is closing are
   discarded.

And the `MediaStreamError` handling in
`SmallWebRTCClient.read_audio_frame` and `read_video_frame`:

1. **Park on dead track** — when the underlying aiortc track is permanently
   raising `MediaStreamError`, the iterator must stop calling `recv()` on it
   (clear the track reference) so we don't busy-loop a CPU core. Without the
   fix, the loop hits `recv()` ~100 times per second indefinitely.

2. **Renegotiation resumes** — after the dead track is replaced by a fresh
   one (the same mechanism `_handle_client_connected` uses), the iterator
   must pick up frames from the new track. A plain `break` on
   `MediaStreamError` would terminate the iterator and regress this path.

And interruption release in `RawAudioTrack`:

1. **RawAudioTrack queue remainder** — only the oldest queued write is faded, with
   no added samples and unchanged timing/future ownership.

2. **Rapid interruptions** — newer writes behind an existing release tail
   are discarded so stale audio cannot accumulate across task restarts.
"""

import asyncio
import fractions
import json
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

# The `webrtc` extra is optional; skip the whole module when it (and its
# transitive `av` dependency) is unavailable, matching the default CI unit
# test environment which does not install extras.
pytest.importorskip("aiortc")
pytest.importorskip("av")

from aiortc.mediastreams import MediaStreamError  # noqa: E402
from av import AudioFrame, VideoFrame  # noqa: E402

from pipecat.audio.utils import _apply_half_hann_fade_out  # noqa: E402
from pipecat.clocks.system_clock import SystemClock  # noqa: E402
from pipecat.frames.frames import (  # noqa: E402
    CancelFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import (  # noqa: E402
    FrameDirection,
    FrameProcessorSetup,
)
from pipecat.transports.base_transport import TransportParams  # noqa: E402
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection  # noqa: E402
from pipecat.transports.smallwebrtc.transport import (  # noqa: E402
    CAM_VIDEO_SOURCE,
    SCREEN_VIDEO_SOURCE,
    RawAudioTrack,
    SmallWebRTCCallbacks,
    SmallWebRTCClient,
    SmallWebRTCOutputTransport,
)
from pipecat.utils.asyncio.task_manager import TaskManager  # noqa: E402


class FakeDataChannel:
    """Stands in for an aiortc `RTCDataChannel` received from the remote peer."""

    def __init__(self, ready_state="open"):
        self.readyState = ready_state
        self.sent = []
        self._handlers = {}

    def send(self, message):
        self.sent.append(message)

    def on(self, event):
        def register(handler):
            self._handlers[event] = handler
            return handler

        return register

    async def fire(self, event):
        await self._handlers[event]()

    @property
    def sent_types(self):
        return [json.loads(m)["type"] for m in self.sent]


async def _noop(*args):
    pass


def _make_client():
    connection = SmallWebRTCConnection()
    callbacks = SmallWebRTCCallbacks(
        on_app_message=_noop, on_client_connected=_noop, on_client_disconnected=_noop
    )
    return SmallWebRTCClient(connection, callbacks), connection


def _message(message_type):
    return OutputTransportMessageUrgentFrame(message={"type": message_type})


def _queued_audio(track: RawAudioTrack) -> bytes:
    return b"".join(chunk for chunk, _ in track._chunk_queue)


def _audio_frame_samples(frame: AudioFrame) -> np.ndarray:
    return frame.to_ndarray().reshape(-1)


class TestRawAudioTrackInterruptionFade(unittest.IsolatedAsyncioTestCase):
    async def test_single_chunk_keeps_shape_future_and_playout_timing(self):
        track = RawAudioTrack(sample_rate=16_000)
        samples = np.full(track._samples_per_10ms, 12_000, dtype=np.int16)
        future = track.add_audio_bytes(samples.tobytes())
        original_shape = [
            (len(chunk), queued_future) for chunk, queued_future in track._chunk_queue
        ]

        track._release_interrupted_audio()

        expected = _apply_half_hann_fade_out(samples.tobytes())
        self.assertEqual(_queued_audio(track), expected)
        self.assertEqual(
            [(len(chunk), queued_future) for chunk, queued_future in track._chunk_queue],
            original_shape,
        )
        self.assertFalse(future.done())

        track._start = 0
        frame = await track.recv()
        self.assertEqual(frame.pts, 0)
        np.testing.assert_array_equal(
            _audio_frame_samples(frame), np.frombuffer(expected, np.int16)
        )
        self.assertTrue(future.result())

        silence = await track.recv()
        self.assertEqual(silence.pts, track._samples_per_10ms)
        self.assertFalse(np.any(_audio_frame_samples(silence)))

    async def test_fades_one_global_remainder_after_playout_boundary(self):
        track = RawAudioTrack(sample_rate=16_000)
        sample_count = track._samples_per_10ms * 4
        positions = np.arange(sample_count)
        samples = np.rint(
            12_000 * np.sin(2 * np.pi * 440 * positions / track._sample_rate + 0.4)
        ).astype(np.int16)
        future = track.add_audio_bytes(samples.tobytes())

        first_frame = await track.recv()
        first_chunk = _audio_frame_samples(first_frame)
        remaining = samples[track._samples_per_10ms :]
        original_shape = [
            (len(chunk), queued_future) for chunk, queued_future in track._chunk_queue
        ]

        track._release_interrupted_audio()

        expected = np.frombuffer(_apply_half_hann_fade_out(remaining.tobytes()), dtype=np.int16)
        faded = np.frombuffer(_queued_audio(track), dtype=np.int16)
        np.testing.assert_array_equal(faded, expected)
        self.assertEqual(
            [(len(chunk), queued_future) for chunk, queued_future in track._chunk_queue],
            original_shape,
        )
        self.assertIs(track._chunk_queue[-1][1], future)
        self.assertEqual(int(faded[0]), int(remaining[0]))
        self.assertEqual(int(faded[-1]), 0)
        self.assertEqual(
            int(faded[0]) - int(first_chunk[-1]),
            int(samples[track._samples_per_10ms]) - int(samples[track._samples_per_10ms - 1]),
        )

        track._start = 0
        emitted = []
        for expected_pts in (160, 320):
            frame = await track.recv()
            self.assertEqual(frame.pts, expected_pts)
            emitted.append(_audio_frame_samples(frame))
            self.assertFalse(future.done())
        final_frame = await track.recv()
        self.assertEqual(final_frame.pts, 480)
        emitted.append(_audio_frame_samples(final_frame))

        self.assertTrue(future.done())
        self.assertTrue(future.result())
        np.testing.assert_array_equal(np.concatenate(emitted), expected)

        silence = await track.recv()
        self.assertEqual(silence.pts, 640)
        silence_samples = _audio_frame_samples(silence)
        self.assertEqual(int(emitted[-1][-1]), int(silence_samples[0]))
        self.assertFalse(np.any(silence_samples))

    async def test_rapid_interruption_discards_newer_writes(self):
        track = RawAudioTrack(sample_rate=16_000)
        oldest = np.full(track._samples_per_10ms * 2, 10_000, dtype=np.int16)
        oldest_future = track.add_audio_bytes(oldest.tobytes())

        track._release_interrupted_audio()
        oldest_future.cancel()
        first_release = _queued_audio(track)

        newer = np.full(track._samples_per_10ms * 2, 20_000, dtype=np.int16)
        newer_future = track.add_audio_bytes(newer.tobytes())
        self.assertEqual(len(track._chunk_queue), 4)

        track._release_interrupted_audio()

        self.assertEqual(len(track._chunk_queue), 2)
        self.assertEqual(_queued_audio(track), _apply_half_hann_fade_out(first_release))
        self.assertTrue(oldest_future.cancelled())
        self.assertTrue(newer_future.cancelled())

        latest = np.full(track._samples_per_10ms * 4, 30_000, dtype=np.int16)
        latest_future = track.add_audio_bytes(latest.tobytes())
        track._release_interrupted_audio()
        self.assertEqual(len(track._chunk_queue), 2)
        self.assertTrue(latest_future.cancelled())

    async def test_cancelled_write_future_can_be_faded_and_drained(self):
        track = RawAudioTrack(sample_rate=16_000)
        samples = np.full(track._samples_per_10ms * 4, -12_000, dtype=np.int16)
        future = track.add_audio_bytes(samples.tobytes())
        future.cancel()

        track._release_interrupted_audio()
        track._start = 0
        for _ in range(4):
            await track.recv()

        self.assertTrue(future.cancelled())

    async def test_empty_queue_is_unchanged(self):
        track = RawAudioTrack(sample_rate=16_000)

        track._release_interrupted_audio()

        self.assertEqual(track._chunk_queue, deque())


class TestSmallWebRTCAudioInterruption(unittest.IsolatedAsyncioTestCase):
    async def test_full_interruption_lifecycle_fades_and_drains_cancelled_write(self):
        track = RawAudioTrack(sample_rate=16_000)
        write_started = asyncio.Event()
        write_future = None

        async def write_audio(frame):
            nonlocal write_future
            write_future = track.add_audio_bytes(frame.audio)
            write_started.set()
            await write_future
            return True

        client = MagicMock()
        client.setup = AsyncMock()
        client.connect = AsyncMock()
        client.disconnect = AsyncMock()
        client.write_audio_frame = AsyncMock(side_effect=write_audio)
        client._release_interrupted_audio = MagicMock(side_effect=track._release_interrupted_audio)
        transport = SmallWebRTCOutputTransport(
            client=client,
            params=TransportParams(audio_out_enabled=True, audio_out_sample_rate=16_000),
        )
        transport.push_frame = AsyncMock()
        await transport.setup(
            FrameProcessorSetup(
                clock=SystemClock(),
                task_manager=TaskManager(),
                pipeline_worker=SimpleNamespace(app_resources=None),  # type: ignore[arg-type]
                audio_out_sample_rate=16_000,
            )
        )

        try:
            await transport.process_frame(
                StartFrame(audio_out_sample_rate=16_000), FrameDirection.DOWNSTREAM
            )
            sender = transport._media_senders[None]
            samples = np.full(sender.audio_chunk_size // 2, 12_000, dtype=np.int16)
            await transport.process_frame(
                OutputAudioRawFrame(audio=samples.tobytes(), sample_rate=16_000, num_channels=1),
                FrameDirection.DOWNSTREAM,
            )
            await write_started.wait()
            expected = _apply_half_hann_fade_out(samples.tobytes())

            await transport.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)

            self.assertIsNotNone(write_future)
            self.assertTrue(write_future.cancelled())
            self.assertEqual(_queued_audio(track), expected)
            client._release_interrupted_audio.assert_called_once_with()

            track._start = 0
            emitted = []
            for _ in range(4):
                emitted.append(_audio_frame_samples(await track.recv()))
            np.testing.assert_array_equal(
                np.concatenate(emitted), np.frombuffer(expected, dtype=np.int16)
            )
        finally:
            await transport.cancel(CancelFrame())

    async def test_client_delegates_fade_to_output_track(self):
        client, connection = _make_client()
        try:
            client._audio_output_track = MagicMock()

            client._release_interrupted_audio()

            client._audio_output_track._release_interrupted_audio.assert_called_once_with()
        finally:
            await connection._pc.close()

    async def test_client_without_output_track_is_unchanged(self):
        client, connection = _make_client()
        try:
            client._release_interrupted_audio()
            self.assertIsNone(client._audio_output_track)
        finally:
            await connection._pc.close()

    async def test_output_transport_delegates_to_client(self):
        client = MagicMock()
        transport = SmallWebRTCOutputTransport(client=client, params=TransportParams())

        cancel_immediately = transport._prepare_audio_interruption(None)

        client._release_interrupted_audio.assert_called_once_with()
        self.assertTrue(cancel_immediately)

    async def test_output_transport_skips_shared_track_with_named_destinations(self):
        for params in (
            TransportParams(audio_out_destinations=["secondary"]),
            TransportParams(video_out_destinations=["secondary"]),
        ):
            with self.subTest(params=params):
                client = MagicMock()
                transport = SmallWebRTCOutputTransport(client=client, params=params)

                cancel_immediately = transport._prepare_audio_interruption(None)

                client._release_interrupted_audio.assert_not_called()
                self.assertFalse(cancel_immediately)


class TestSendMessage(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client, self.connection = _make_client()

    async def asyncTearDown(self):
        await self.connection._pc.close()

    async def test_queues_before_connection_and_flushes_on_channel_arrival(self):
        """Messages sent pre-connect are buffered and flushed in order.

        The data channel is created by the remote peer, so aiortc emits
        "datachannel" with the channel already open and no "open" event
        follows — the flush must happen on arrival.
        """
        for message_type in ("user-mute-started", "metrics", "bot-ready"):
            await self.client.send_message(_message(message_type))
        self.assertEqual(len(self.connection._outgoing_messages_queue), 3)

        channel = FakeDataChannel()
        self.connection._pc.emit("datachannel", channel)

        self.assertEqual(channel.sent_types, ["user-mute-started", "metrics", "bot-ready"])
        self.assertEqual(self.connection._outgoing_messages_queue, [])

    async def test_flushes_on_open_event_when_channel_arrives_connecting(self):
        """A channel that arrives before opening flushes when "open" fires."""
        await self.client.send_message(_message("user-mute-started"))

        channel = FakeDataChannel(ready_state="connecting")
        self.connection._pc.emit("datachannel", channel)
        self.assertEqual(channel.sent, [])

        channel.readyState = "open"
        await channel.fire("open")
        self.assertEqual(channel.sent_types, ["user-mute-started"])

    async def test_sends_directly_when_channel_open(self):
        channel = FakeDataChannel()
        self.connection._pc.emit("datachannel", channel)

        await self.client.send_message(_message("server-message"))
        self.assertEqual(channel.sent_types, ["server-message"])
        self.assertEqual(self.connection._outgoing_messages_queue, [])

    async def test_discards_when_closing(self):
        channel = FakeDataChannel()
        self.connection._pc.emit("datachannel", channel)
        self.client._closing = True

        await self.client.send_message(_message("server-message"))
        self.assertEqual(channel.sent, [])
        self.assertEqual(self.connection._outgoing_messages_queue, [])


def _make_audio_self(track):
    fake = MagicMock()
    fake._audio_input_track = track
    fake._webrtc_connection = MagicMock()
    fake._webrtc_connection.is_connected.return_value = True
    fake._in_sample_rate = 16_000
    fake._audio_in_channels = 1
    # Passthrough resampler.
    fake._audio_in_resampler.resample.side_effect = lambda f: [f]
    return fake


def _make_video_self(video_track=None, screen_track=None):
    fake = MagicMock()
    fake._video_input_track = video_track
    fake._screen_video_track = screen_track
    fake._webrtc_connection = MagicMock()
    fake._webrtc_connection.is_connected.return_value = True
    fake._webrtc_connection.pc_id = "test-pc"
    fake._convert_frame.side_effect = lambda arr, fmt: arr
    return fake


def _good_audio_frame():
    samples = 320  # 20 ms @ 16 kHz
    arr = np.zeros((1, samples), dtype=np.int16)
    f = AudioFrame.from_ndarray(arr, format="s16", layout="mono")
    f.sample_rate = 16_000
    f.pts = 0
    f.time_base = fractions.Fraction(1, 16_000)
    return f


def _good_video_frame():
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    f = VideoFrame.from_ndarray(arr, format="rgb24")
    f.pts = 0
    return f


class TestReadAudioFrameMediaStreamError(unittest.IsolatedAsyncioTestCase):
    async def test_parks_on_dead_track(self):
        """Dead track: iterator must null the track ref and stop calling recv().

        Without the fix this loop calls `track.recv()` ~100Hz forever, pinning
        a CPU core. With the fix, `_audio_input_track` is set to None on the
        first `MediaStreamError` and the loop parks on the `is None` gate.
        """
        track = MagicMock()
        track.recv = AsyncMock(side_effect=MediaStreamError("track ended"))
        fake = _make_audio_self(track)

        async def consume():
            async for _ in SmallWebRTCClient.read_audio_frame(fake):
                pass

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.2)
        task.cancel()
        try:
            await task
        except BaseException:
            pass

        # Exactly one recv() call: after MediaStreamError, the track ref is
        # cleared and the loop sleeps on `is None` instead of re-calling recv.
        self.assertEqual(track.recv.await_count, 1)
        self.assertIsNone(fake._audio_input_track)

    async def test_renegotiation_resumes(self):
        """After the dead track is replaced, the iterator must yield frames.

        This is the renegotiation path: a plain `break` on `MediaStreamError`
        would terminate the generator. The track-nulling fix lets the
        existing `is None: sleep; continue` gate wait for a fresh track from
        `_handle_client_connected`.
        """
        dead = MagicMock()
        dead.recv = AsyncMock(side_effect=MediaStreamError("track ended"))
        fresh = MagicMock()
        fresh.recv = AsyncMock(return_value=_good_audio_frame())
        fake = _make_audio_self(dead)

        yielded = 0

        async def consume():
            nonlocal yielded
            async for _ in SmallWebRTCClient.read_audio_frame(fake):
                yielded += 1
                if yielded >= 3:
                    break

        task = asyncio.create_task(consume())
        # Let the dead track raise + the loop park on `is None`.
        await asyncio.sleep(0.05)
        # Simulate _handle_client_connected reassigning a fresh track.
        fake._audio_input_track = fresh
        await asyncio.wait_for(task, timeout=1.0)

        self.assertEqual(dead.recv.await_count, 1)
        self.assertGreaterEqual(yielded, 3)


class TestReadVideoFrameMediaStreamError(unittest.IsolatedAsyncioTestCase):
    async def test_camera_parks_on_dead_track(self):
        track = MagicMock()
        track.recv = AsyncMock(side_effect=MediaStreamError("track ended"))
        fake = _make_video_self(video_track=track)

        async def consume():
            async for _ in SmallWebRTCClient.read_video_frame(fake, CAM_VIDEO_SOURCE):
                pass

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.2)
        task.cancel()
        try:
            await task
        except BaseException:
            pass

        self.assertEqual(track.recv.await_count, 1)
        self.assertIsNone(fake._video_input_track)

    async def test_screen_parks_on_dead_track(self):
        """Screen-share uses a separate track reference."""
        track = MagicMock()
        track.recv = AsyncMock(side_effect=MediaStreamError("track ended"))
        fake = _make_video_self(screen_track=track)

        async def consume():
            async for _ in SmallWebRTCClient.read_video_frame(fake, SCREEN_VIDEO_SOURCE):
                pass

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.2)
        task.cancel()
        try:
            await task
        except BaseException:
            pass

        self.assertEqual(track.recv.await_count, 1)
        self.assertIsNone(fake._screen_video_track)

    async def test_camera_renegotiation_resumes(self):
        dead = MagicMock()
        dead.recv = AsyncMock(side_effect=MediaStreamError("track ended"))
        fresh = MagicMock()
        fresh.recv = AsyncMock(return_value=_good_video_frame())
        fake = _make_video_self(video_track=dead)

        yielded = 0

        async def consume():
            nonlocal yielded
            async for _ in SmallWebRTCClient.read_video_frame(fake, CAM_VIDEO_SOURCE):
                yielded += 1
                if yielded >= 2:
                    break

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        fake._video_input_track = fresh
        await asyncio.wait_for(task, timeout=1.0)

        self.assertEqual(dead.recv.await_count, 1)
        self.assertGreaterEqual(yielded, 2)


if __name__ == "__main__":
    unittest.main()
