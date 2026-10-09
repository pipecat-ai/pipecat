#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for LiveKit transport video stream handling.

Regression tests for issue #3116: Memory leak when video_in_enabled=False
but video tracks are subscribed. The fix ensures video stream processing
only starts when there is a consumer for the frames.
"""

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

try:
    from livekit import rtc

    from pipecat.frames.frames import (
        ImageRawFrame,
        InputAudioRawFrame,
        OutputImageRawFrame,
        UserAudioRawFrame,
        UserImageRequestFrame,
    )
    from pipecat.transports.base_transport import VideoInSourceParams
    from pipecat.transports.livekit.transport import (
        LiveKitCallbacks,
        LiveKitInputTransport,
        LiveKitOutputTransport,
        LiveKitParams,
        LiveKitTransport,
        LiveKitTransportClient,
        _ParticipantAudioMixer,
    )

    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False


@unittest.skipUnless(LIVEKIT_AVAILABLE, "livekit package not installed")
class TestLiveKitVideoStreamMemoryLeak(unittest.IsolatedAsyncioTestCase):
    """Regression tests for video queue memory leak (#3116).

    The bug: When video_in_enabled=False, subscribing to a video track would
    start a producer that fills _video_queue, but no consumer would drain it,
    causing unbounded memory growth (~3GB/min).

    The fix: Only start video stream processing when video_in_enabled=True.
    """

    def _create_client(self, video_in_enabled: bool) -> LiveKitTransportClient:
        """Create a client with the specified video input setting."""
        params = LiveKitParams(video_in_enabled=video_in_enabled)
        callbacks = LiveKitCallbacks(
            on_connected=AsyncMock(),
            on_disconnected=AsyncMock(),
            on_before_disconnect=AsyncMock(),
            on_participant_connected=AsyncMock(),
            on_participant_disconnected=AsyncMock(),
            on_audio_track_subscribed=AsyncMock(),
            on_audio_track_unsubscribed=AsyncMock(),
            on_video_track_subscribed=AsyncMock(),
            on_video_track_unsubscribed=AsyncMock(),
            on_data_received=AsyncMock(),
            on_first_participant_joined=AsyncMock(),
            on_dtmf_event=AsyncMock(),
            on_active_speaker_changed=AsyncMock(),
            on_video_track_muted=AsyncMock(),
        )
        client = LiveKitTransportClient(
            url="wss://test.livekit.cloud",
            token="test-token",
            room_name="test-room",
            params=params,
            callbacks=callbacks,
            transport_name="test-transport",
        )
        client._task_manager = MagicMock()
        # Close the stream coroutines instead of running them.
        client._task_manager.create_task.side_effect = lambda coro, name: coro.close()
        return client

    def _create_mock_video_track(self):
        """Create a mock video track subscription event."""
        track = MagicMock()
        track.kind = rtc.TrackKind.KIND_VIDEO
        track.sid = "video-track-123"
        publication = MagicMock()
        participant = MagicMock()
        participant.identity = "participant-456"
        return track, publication, participant

    async def test_disabled_video_input_does_not_start_queue_producer(self):
        """When video input is disabled, no producer should fill the queue.

        This prevents the memory leak where frames accumulate with no consumer.
        """
        client = self._create_client(video_in_enabled=False)
        track, publication, participant = self._create_mock_video_track()

        await client._async_on_track_subscribed(track, publication, participant)

        # Verify no video processing task was started
        task_names = [call[0][1] for call in client._task_manager.create_task.call_args_list]
        video_tasks = [name for name in task_names if "video" in name.lower()]
        self.assertEqual(video_tasks, [], "No video processing task should be started")

        # Queue should remain empty
        self.assertEqual(client._video_queue.qsize(), 0)

        # Track metadata should still be recorded
        self.assertIn((participant.identity, "camera"), client._video_tracks)

        # Callback should still fire for user code
        client._callbacks.on_video_track_subscribed.assert_called_once()

    async def test_enabled_video_input_starts_queue_producer(self):
        """When video input is enabled, the producer should start."""
        client = self._create_client(video_in_enabled=True)
        track, publication, participant = self._create_mock_video_track()

        with patch.object(rtc, "VideoStream"):
            await client._async_on_track_subscribed(track, publication, participant)

        # Verify video processing task was started
        task_names = [call[0][1] for call in client._task_manager.create_task.call_args_list]
        video_tasks = [name for name in task_names if "video" in name.lower()]
        self.assertEqual(len(video_tasks), 1, "Video processing task should be started")

        # Track metadata should be recorded
        self.assertIn((participant.identity, "camera"), client._video_tracks)

        # Callback should fire
        client._callbacks.on_video_track_subscribed.assert_called_once()


@unittest.skipUnless(LIVEKIT_AVAILABLE, "livekit package not installed")
class TestLiveKitAudioStreamLeakOnUnsubscribe(unittest.IsolatedAsyncioTestCase):
    """Regression tests for AudioStream leak on track unsubscribe.

    The bug: ``_async_on_track_subscribed`` creates an owned ``rtc.AudioStream``
    plus a ``_process_audio_stream`` task feeding the shared ``_audio_queue``,
    but only saves the track. ``_async_on_track_unsubscribed`` never closes the
    stream or cancels the task. Per livekit-rtc ``audio_stream.py``, an owned
    ``AudioStream._run`` loops over the FFI queue and exits only on a native
    ``eos`` event emitted by ``aclose()`` → ``_ffi_handle.dispose()``. So when a
    participant republishes their mic (e.g. mute/unmute), the previous stream
    keeps pushing frames forever; N republishes → N concurrent producers
    interleave audio into the shared queue and downstream STT receives garbage.

    The fix: store ``(stream, task)`` per ``participant.identity`` in
    ``_audio_streams`` on subscribe, then ``aclose()`` + cancel on unsubscribe
    and again on a re-subscribe for the same identity (to handle missed
    unsubscribe). Symmetric for video.
    """

    def _create_client(self, video_in_enabled: bool = False) -> LiveKitTransportClient:
        params = LiveKitParams(video_in_enabled=video_in_enabled)
        callbacks = LiveKitCallbacks(
            on_connected=AsyncMock(),
            on_disconnected=AsyncMock(),
            on_before_disconnect=AsyncMock(),
            on_participant_connected=AsyncMock(),
            on_participant_disconnected=AsyncMock(),
            on_audio_track_subscribed=AsyncMock(),
            on_audio_track_unsubscribed=AsyncMock(),
            on_video_track_subscribed=AsyncMock(),
            on_video_track_unsubscribed=AsyncMock(),
            on_data_received=AsyncMock(),
            on_first_participant_joined=AsyncMock(),
            on_dtmf_event=AsyncMock(),
            on_active_speaker_changed=AsyncMock(),
            on_video_track_muted=AsyncMock(),
        )
        client = LiveKitTransportClient(
            url="wss://test.livekit.cloud",
            token="test-token",
            room_name="test-room",
            params=params,
            callbacks=callbacks,
            transport_name="test-transport",
        )
        client._task_manager = MagicMock()

        # Return a real (mockable) Task so the cleanup path can call ``.done()``
        # and ``.cancel()`` on it without blowing up.
        def _make_task(coro, name):
            coro.close()  # we never run the producer in the unit test
            t = MagicMock()
            t.done.return_value = False
            t.cancel = MagicMock()
            return t

        client._task_manager.create_task.side_effect = _make_task
        return client

    def _audio_track(self, sid: str = "audio-track-1", participant_identity: str = "p-1"):
        track = MagicMock()
        track.kind = rtc.TrackKind.KIND_AUDIO
        track.sid = sid
        publication = MagicMock()
        publication.sid = sid
        participant = MagicMock()
        participant.identity = participant_identity
        return track, publication, participant

    def _video_track(self, sid: str = "video-track-1", participant_identity: str = "p-1"):
        track = MagicMock()
        track.kind = rtc.TrackKind.KIND_VIDEO
        track.sid = sid
        publication = MagicMock()
        publication.sid = sid
        participant = MagicMock()
        participant.identity = participant_identity
        return track, publication, participant

    async def test_audio_stream_registered_on_subscribe(self):
        """Subscribing an audio track registers ``(stream, task)`` for the sid."""
        client = self._create_client()
        track, pub, participant = self._audio_track()

        mock_stream = MagicMock()
        mock_stream.aclose = AsyncMock()
        with patch.object(rtc, "AudioStream", return_value=mock_stream):
            await client._async_on_track_subscribed(track, pub, participant)

        self.assertIn(participant.identity, client._audio_streams)
        stream, task = client._audio_streams[participant.identity]
        self.assertIs(stream, mock_stream)
        self.assertIsNotNone(task)

    async def test_audio_stream_closed_and_task_cancelled_on_unsubscribe(self):
        """Unsubscribing closes the stream, cancels the task, clears the registry."""
        client = self._create_client()
        track, pub, participant = self._audio_track()

        mock_stream = MagicMock()
        mock_stream.aclose = AsyncMock()
        with patch.object(rtc, "AudioStream", return_value=mock_stream):
            await client._async_on_track_subscribed(track, pub, participant)
        _, task = client._audio_streams[participant.identity]

        await client._async_on_track_unsubscribed(track, pub, participant)

        mock_stream.aclose.assert_awaited_once()
        task.cancel.assert_called_once()
        self.assertNotIn(participant.identity, client._audio_streams)
        client._callbacks.on_audio_track_unsubscribed.assert_called_once()

    async def test_resubscribe_closes_previous_audio_stream(self):
        """Re-subscribing the same sid (mic republish) closes the prior stream."""
        client = self._create_client()
        track, pub, participant = self._audio_track()

        first = MagicMock()
        first.aclose = AsyncMock()
        second = MagicMock()
        second.aclose = AsyncMock()

        with patch.object(rtc, "AudioStream", return_value=first):
            await client._async_on_track_subscribed(track, pub, participant)
        first_task = client._audio_streams[participant.identity][1]

        # Republish without an explicit unsubscribe in between.
        with patch.object(rtc, "AudioStream", return_value=second):
            await client._async_on_track_subscribed(track, pub, participant)

        first.aclose.assert_awaited_once()
        first_task.cancel.assert_called_once()
        self.assertIs(client._audio_streams[participant.identity][0], second)

    async def test_unsubscribe_without_subscribe_is_noop(self):
        """Unsubscribe for an unknown sid does not raise."""
        client = self._create_client()
        track, pub, participant = self._audio_track()
        # No subscribe before this call.
        await client._async_on_track_unsubscribed(track, pub, participant)
        client._callbacks.on_audio_track_unsubscribed.assert_called_once()

    async def test_video_stream_closed_on_unsubscribe(self):
        """Symmetric behaviour for video when ``video_in_enabled=True``."""
        client = self._create_client(video_in_enabled=True)
        track, pub, participant = self._video_track()

        mock_stream = MagicMock()
        mock_stream.aclose = AsyncMock()
        with patch.object(rtc, "VideoStream", return_value=mock_stream):
            await client._async_on_track_subscribed(track, pub, participant)
        self.assertIn((participant.identity, "camera"), client._video_streams)

        await client._async_on_track_unsubscribed(track, pub, participant)
        mock_stream.aclose.assert_awaited_once()
        self.assertNotIn((participant.identity, "camera"), client._video_streams)


@unittest.skipUnless(LIVEKIT_AVAILABLE, "livekit package not installed")
class TestLiveKitInputResamplesEachParticipantSeparately(unittest.IsolatedAsyncioTestCase):
    """Audio from participants speaking at the same time must not bleed together.

    The bug: every participant's 48 kHz audio went through one shared stream
    resampler. A stream resampler keeps filter state between calls, so when two
    participants' frames are interleaved, each output frame is built partly from
    the other participant's samples, and speech recognition hears both voices.
    """

    RATE_IN, RATE_OUT, TONES = 48000, 16000, {"alice": 440.0, "bob": 1000.0}

    def _frame(self, hz: float, index: int) -> "rtc.AudioFrameEvent":
        n = self.RATE_IN // 100  # 10 ms
        t = (np.arange(n) + index * n) / self.RATE_IN
        pcm = (8000 * np.sin(2 * np.pi * hz * t)).astype(np.int16).tobytes()
        return rtc.AudioFrameEvent(frame=rtc.AudioFrame(pcm, self.RATE_IN, 1, n))

    def _magnitude(self, pcm: bytes, hz: float) -> float:
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
        spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
        return float(spectrum[int(round(hz * len(samples) / self.RATE_OUT))])

    async def test_each_participant_keeps_only_their_own_audio(self):
        frames = [(self._frame(hz, i), who) for i in range(100) for who, hz in self.TONES.items()]

        async def interleaved():  # two participants talking at once, 1 s each
            for frame in frames:
                yield frame

        client = MagicMock()
        client.get_next_audio_frame = interleaved
        transport = LiveKitInputTransport(MagicMock(), client, LiveKitParams())
        transport._sample_rate = self.RATE_OUT
        heard = {who: b"" for who in self.TONES}

        async def collect(frame):
            heard[frame.user_id] += frame.audio

        transport.push_audio_frame = collect
        await transport._audio_in_task_handler()

        for who, other in (("alice", "bob"), ("bob", "alice")):
            pcm = heard[who][len(heard[who]) // 4 :]  # past the resampler's start-up
            leaked = self._magnitude(pcm, self.TONES[other]) / self._magnitude(pcm, self.TONES[who])
            self.assertLess(leaked, 0.01, f"{who}'s audio carries {other}'s voice ({leaked:.0%})")

    async def test_a_participant_whose_audio_track_is_gone_is_released(self):
        transport = LiveKitTransport(url="wss://test.livekit.cloud", token="t", room_name="r")
        transport.input()._resamplers["alice"] = MagicMock()

        await transport._on_audio_track_unsubscribed("alice")

        self.assertNotIn("alice", transport.input()._resamplers)


@unittest.skipUnless(LIVEKIT_AVAILABLE, "livekit package not installed")
class TestLiveKitSipDtmfInput(unittest.IsolatedAsyncioTestCase):
    """Inbound SIP DTMF should surface as InputDTMFFrame (#4436)."""

    def _create_client(self) -> LiveKitTransportClient:
        params = LiveKitParams()
        callbacks = LiveKitCallbacks(
            on_connected=AsyncMock(),
            on_disconnected=AsyncMock(),
            on_before_disconnect=AsyncMock(),
            on_participant_connected=AsyncMock(),
            on_participant_disconnected=AsyncMock(),
            on_audio_track_subscribed=AsyncMock(),
            on_audio_track_unsubscribed=AsyncMock(),
            on_video_track_subscribed=AsyncMock(),
            on_video_track_unsubscribed=AsyncMock(),
            on_data_received=AsyncMock(),
            on_first_participant_joined=AsyncMock(),
            on_dtmf_event=AsyncMock(),
            on_active_speaker_changed=AsyncMock(),
            on_video_track_muted=AsyncMock(),
        )
        client = LiveKitTransportClient(
            url="wss://test.livekit.cloud",
            token="test-token",
            room_name="test-room",
            params=params,
            callbacks=callbacks,
            transport_name="test-transport",
        )
        client._task_manager = MagicMock()
        return client

    async def test_sip_dtmf_forwards_digit_to_callback(self):
        """Room sip_dtmf_received events are normalized and forwarded."""
        client = self._create_client()
        participant = MagicMock()
        participant.identity = "sip-participant-1"
        dtmf = MagicMock()
        dtmf.digit = "5"
        dtmf.code = 5
        dtmf.participant = participant

        await client._async_on_sip_dtmf_received(dtmf)

        client._callbacks.on_dtmf_event.assert_awaited_once_with(
            {
                "tone": "5",
                "digit": "5",
                "code": 5,
                "participant_id": "sip-participant-1",
            }
        )

    async def test_transport_dtmf_event_pushes_input_frame(self):
        """Transport pushes InputDTMFFrame so DTMFAggregator can consume digits."""
        from pipecat.audio.dtmf.types import KeypadEntry
        from pipecat.frames.frames import InputDTMFFrame
        from pipecat.transports.livekit.transport import LiveKitTransport

        transport = LiveKitTransport(
            url="wss://test.livekit.cloud",
            token="test-token",
            room_name="test-room",
        )
        transport._input = MagicMock()
        transport._input.push_frame = AsyncMock()
        transport._call_event_handler = AsyncMock()

        data = {
            "tone": "1",
            "digit": "1",
            "code": 1,
            "participant_id": "sip-participant-1",
        }
        await transport._on_dtmf_event(data)

        transport._call_event_handler.assert_awaited_once_with("on_dtmf_event", data)
        transport._input.push_frame.assert_awaited_once()
        frame = transport._input.push_frame.await_args.args[0]
        self.assertIsInstance(frame, InputDTMFFrame)
        self.assertEqual(frame.button, KeypadEntry.ONE)

    async def test_transport_ignores_unsupported_dtmf_tone(self):
        """Unsupported tones are logged and do not push a frame."""
        from pipecat.transports.livekit.transport import LiveKitTransport

        transport = LiveKitTransport(
            url="wss://test.livekit.cloud",
            token="test-token",
            room_name="test-room",
        )
        transport._input = MagicMock()
        transport._input.push_frame = AsyncMock()
        transport._call_event_handler = AsyncMock()

        await transport._on_dtmf_event(
            {"tone": "A", "digit": "A", "code": 12, "participant_id": "p1"}
        )

        transport._call_event_handler.assert_awaited_once()
        transport._input.push_frame.assert_not_awaited()


@unittest.skipUnless(LIVEKIT_AVAILABLE, "livekit package not installed")
class TestLiveKitActiveSpeaker(unittest.IsolatedAsyncioTestCase):
    """The room's loudest speaker surfaces as on_active_speaker_changed, as on Daily.

    LiveKit's active_speakers_changed lists only speakers whose level changed, quietest
    first, so an empty list doesn't mean silence.
    """

    def _client(self):
        client = TestLiveKitSipDtmfInput._create_client(self)
        client._callbacks.on_active_speaker_changed = AsyncMock()
        return client

    async def _run_scheduled(self, client):
        for call in client._task_manager.create_task.call_args_list:
            await call.args[0]

    async def test_the_loudest_speaker_is_reported(self):
        client = self._client()
        client._on_active_speakers_changed_wrapper(
            [MagicMock(identity="bob"), MagicMock(identity="alice")]
        )
        await self._run_scheduled(client)
        client._callbacks.on_active_speaker_changed.assert_awaited_once_with("alice")

    async def test_only_a_change_of_speaker_is_reported(self):
        client = self._client()
        alice, bob = MagicMock(identity="alice"), MagicMock(identity="bob")
        for speakers in ([alice], [bob, alice], [], [alice], [bob]):
            client._on_active_speakers_changed_wrapper(speakers)
        await self._run_scheduled(client)
        self.assertEqual(
            [c.args[0] for c in client._callbacks.on_active_speaker_changed.await_args_list],
            ["alice", "bob"],
        )

    async def test_connect_forgets_the_last_speaker(self):
        client = self._client()
        client._active_speaker_id = "alice"
        client._out_sample_rate = 16000
        client._room = MagicMock()
        client._room.connect = AsyncMock()
        client._room.remote_participants = {}
        client._room.local_participant.publish_track = AsyncMock()
        with (
            patch("pipecat.transports.livekit.transport.rtc.AudioSource"),
            patch("pipecat.transports.livekit.transport.rtc.LocalAudioTrack"),
        ):
            await client.connect()
        self.assertIsNone(client._active_speaker_id)

    async def test_setup_listens_for_the_room_event(self):
        client = TestLiveKitSipDtmfInput._create_client(self)
        client._task_manager = None
        setup = MagicMock(audio_out_sample_rate=16000)

        with patch("pipecat.transports.livekit.transport.rtc.Room") as room_class:
            await client.setup(setup)

        on = room_class.return_value.on  # room.on(event)(handler)
        i = [c.args[0] for c in on.call_args_list].index("active_speakers_changed")
        self.assertEqual(
            on.return_value.call_args_list[i].args[0], client._on_active_speakers_changed_wrapper
        )

    async def test_transport_emits_dailys_event(self):
        from pipecat.transports.livekit.transport import LiveKitTransport

        transport = LiveKitTransport(
            url="wss://test.livekit.cloud", token="test-token", room_name="test-room"
        )
        transport._call_event_handler = AsyncMock()

        await transport._on_active_speaker_changed("alice")

        transport._call_event_handler.assert_awaited_once_with(
            "on_active_speaker_changed", {"id": "alice"}
        )


@unittest.skipUnless(LIVEKIT_AVAILABLE, "livekit package not installed")
class TestLiveKitAppMessageInput(unittest.IsolatedAsyncioTestCase):
    """Inbound JSON data messages (RTVI's wire channel) are parsed and
    broadcast both directions as an ``InputTransportMessageFrame`` so
    ``RTVIProcessor`` sees them wherever it sits in the pipeline. Non-object
    JSON and non-JSON data are not pushed into the pipeline, but still fire
    ``on_data_received`` for backwards compatibility.
    """

    def _make_transport(self):
        from pipecat.transports.livekit.transport import LiveKitTransport

        transport = LiveKitTransport(
            url="wss://test.livekit.cloud",
            token="test-token",
            room_name="test-room",
        )
        input_transport = transport.input()
        input_transport.push_frame = AsyncMock()
        transport._call_event_handler = AsyncMock()
        return transport, input_transport

    async def test_data_received_broadcasts_parsed_input_message_frame(self):
        """A JSON data message is parsed and broadcast as an InputTransportMessageFrame."""
        from pipecat.frames.frames import InputTransportMessageFrame
        from pipecat.processors.frame_processor import FrameDirection

        transport, input_transport = self._make_transport()

        rtvi_message = {"label": "rtvi-ai", "type": "client-ready", "id": "1", "data": {}}
        await transport._on_data_received(json.dumps(rtvi_message).encode(), "participant-1")

        self.assertEqual(input_transport.push_frame.await_count, 2)
        directions = set()
        for call in input_transport.push_frame.await_args_list:
            frame = call.args[0]
            direction = call.args[1] if len(call.args) > 1 else FrameDirection.DOWNSTREAM
            self.assertIsInstance(frame, InputTransportMessageFrame)
            self.assertEqual(frame.message, rtvi_message)
            self.assertEqual(frame.participant_id, "participant-1")
            directions.add(direction)
        # Broadcast both ways so RTVIProcessor sees it regardless of where it
        # sits relative to the transport in the pipeline.
        self.assertEqual(directions, {FrameDirection.DOWNSTREAM, FrameDirection.UPSTREAM})

        transport._call_event_handler.assert_any_call(
            "on_app_message", rtvi_message, "participant-1"
        )

    async def test_non_json_data_is_not_pushed_but_reported_for_compat(self):
        """Non-JSON data doesn't crash, isn't pushed, but still reports on_data_received."""
        transport, input_transport = self._make_transport()

        await transport._on_data_received(b"not json", "participant-1")

        input_transport.push_frame.assert_not_awaited()
        transport._call_event_handler.assert_awaited_once_with(
            "on_data_received", b"not json", "participant-1"
        )

    async def test_non_object_json_is_not_pushed_but_reported_for_compat(self):
        """A JSON value that isn't an object (str/number/bool/list) is ignored.

        ``RTVIProcessor`` calls ``.get("label")`` on the parsed message, so
        pushing a non-dict would raise ``AttributeError`` deep in the pipeline.
        """
        transport, input_transport = self._make_transport()

        await transport._on_data_received(json.dumps("hi").encode(), "participant-1")

        input_transport.push_frame.assert_not_awaited()
        transport._call_event_handler.assert_awaited_once_with(
            "on_data_received", json.dumps("hi").encode(), "participant-1"
        )


@unittest.skipUnless(LIVEKIT_AVAILABLE, "livekit package not installed")
class TestLiveKitClientConnectedAlias(unittest.IsolatedAsyncioTestCase):
    """on_client_connected/on_client_disconnected match Daily's dict shape.

    Drop-in bot templates read ``client["id"]`` off these aliases, so the
    payload needs to be a mapping, not a bare participant id string.
    """

    def _make_transport(self):
        from pipecat.transports.livekit.transport import LiveKitTransport

        transport = LiveKitTransport(
            url="wss://test.livekit.cloud",
            token="test-token",
            room_name="test-room",
        )
        transport._input = MagicMock()
        transport._input.push_frame = AsyncMock()
        transport._input.remove_participant_video = AsyncMock()
        transport._call_event_handler = AsyncMock()
        return transport

    async def test_on_client_connected_receives_dict(self):
        transport = self._make_transport()

        await transport._on_participant_connected("participant-1")

        transport._call_event_handler.assert_any_call(
            "on_client_connected", {"id": "participant-1"}
        )

    async def test_on_client_disconnected_receives_dict(self):
        transport = self._make_transport()

        await transport._on_participant_disconnected("participant-1")

        transport._call_event_handler.assert_any_call(
            "on_client_disconnected", {"id": "participant-1"}
        )


@unittest.skipUnless(LIVEKIT_AVAILABLE, "livekit package not installed")
class TestLiveKitParticipantIdentity(unittest.IsolatedAsyncioTestCase):
    """The participant_id this transport hands out must be the LiveKit
    identity, not the SID.

    Regression test (pipecat-ai/pipecat#5218): ``room.remote_participants``
    is keyed by identity and ``destination_identities`` expects identities,
    but the transport used to emit ``participant.sid`` everywhere. Callers
    couldn't feed the ``get_participants()``/event ``participant_id`` back
    into ``get_participant_metadata``/``mute_participant``/
    ``unmute_participant``/targeted ``send_message`` — the lookup would
    silently fail (``room.remote_participants.get(sid)`` returns ``None``).
    """

    def _create_client(self) -> LiveKitTransportClient:
        params = LiveKitParams()
        callbacks = LiveKitCallbacks(
            on_connected=AsyncMock(),
            on_disconnected=AsyncMock(),
            on_before_disconnect=AsyncMock(),
            on_participant_connected=AsyncMock(),
            on_participant_disconnected=AsyncMock(),
            on_audio_track_subscribed=AsyncMock(),
            on_audio_track_unsubscribed=AsyncMock(),
            on_video_track_subscribed=AsyncMock(),
            on_video_track_unsubscribed=AsyncMock(),
            on_data_received=AsyncMock(),
            on_first_participant_joined=AsyncMock(),
            on_dtmf_event=AsyncMock(),
            on_active_speaker_changed=AsyncMock(),
            on_video_track_muted=AsyncMock(),
        )
        client = LiveKitTransportClient(
            url="wss://test.livekit.cloud",
            token="test-token",
            room_name="test-room",
            params=params,
            callbacks=callbacks,
            transport_name="test-transport",
        )
        client._task_manager = MagicMock()
        return client

    def _mock_room_with_participant(
        self, client: LiveKitTransportClient, *, sid: str, identity: str
    ):
        publication = MagicMock()
        publication.kind = rtc.TrackKind.KIND_AUDIO
        publication.set_subscribed = MagicMock()

        participant = MagicMock()
        participant.sid = sid
        participant.identity = identity
        participant.name = "Test User"
        participant.metadata = ""
        participant.track_publications = {"track-1": publication}

        room = MagicMock()
        room.remote_participants = {identity: participant}
        client._room = room
        return participant, publication

    async def test_get_participants_returns_identity_not_sid(self):
        client = self._create_client()
        participant, _ = self._mock_room_with_participant(
            client, sid="PA_serverSid", identity="repro-client"
        )

        self.assertEqual(client.get_participants(), ["repro-client"])

    async def test_get_participant_metadata_resolves_id_from_get_participants(self):
        """The id get_participants() hands out must work as a lookup key."""
        client = self._create_client()
        self._mock_room_with_participant(client, sid="PA_serverSid", identity="repro-client")

        (participant_id,) = client.get_participants()
        metadata = await client.get_participant_metadata(participant_id)

        self.assertEqual(metadata, {"id": "repro-client", "name": "Test User", "metadata": ""})

    async def test_mute_participant_resolves_id_from_get_participants(self):
        client = self._create_client()
        _, publication = self._mock_room_with_participant(
            client, sid="PA_serverSid", identity="repro-client"
        )

        (participant_id,) = client.get_participants()
        await client.mute_participant(participant_id)

        publication.set_subscribed.assert_called_once_with(False)

    async def test_unmute_participant_resolves_id_from_get_participants(self):
        client = self._create_client()
        _, publication = self._mock_room_with_participant(
            client, sid="PA_serverSid", identity="repro-client"
        )

        (participant_id,) = client.get_participants()
        await client.unmute_participant(participant_id)

        publication.set_subscribed.assert_called_once_with(True)

    async def test_participant_connected_callback_receives_identity(self):
        client = self._create_client()
        participant = MagicMock()
        participant.sid = "PA_serverSid"
        participant.identity = "repro-client"

        await client._async_on_participant_connected(participant)

        client._callbacks.on_participant_connected.assert_awaited_once_with("repro-client")


@unittest.skipUnless(LIVEKIT_AVAILABLE, "livekit package not installed")
class TestLiveKitAudioTrackSubscribedHandler(unittest.TestCase):
    """The top-level transport's on_audio/video_track_subscribed handlers
    must not re-derive publications from nonexistent SDK attributes.

    Regression test: these used to look up ``participant.audio_tracks``/
    ``participant.video_tracks`` (removed from the SDK; ``track_publications``
    is the only such attribute now) and re-invoke the subscribe wrapper that
    had already run for this exact track via the room event, redundantly.
    """

    def test_on_audio_track_subscribed_only_fires_event_handler(self):
        import asyncio

        from pipecat.transports.livekit.transport import LiveKitTransport

        transport = LiveKitTransport(
            url="wss://test.livekit.cloud", token="test-token", room_name="test-room"
        )
        transport._call_event_handler = AsyncMock()
        transport._client = MagicMock()

        asyncio.run(transport._on_audio_track_subscribed("participant-1"))

        transport._call_event_handler.assert_awaited_once_with(
            "on_audio_track_subscribed", "participant-1"
        )
        transport._client.room.remote_participants.get.assert_not_called()

    def test_on_video_track_subscribed_only_fires_event_handler(self):
        import asyncio

        from pipecat.transports.livekit.transport import LiveKitTransport

        transport = LiveKitTransport(
            url="wss://test.livekit.cloud", token="test-token", room_name="test-room"
        )
        transport._call_event_handler = AsyncMock()
        transport._client = MagicMock()

        asyncio.run(transport._on_video_track_subscribed("participant-1"))

        transport._call_event_handler.assert_awaited_once_with(
            "on_video_track_subscribed", "participant-1"
        )
        transport._client.room.remote_participants.get.assert_not_called()


@unittest.skipUnless(LIVEKIT_AVAILABLE, "livekit package not installed")
class TestLiveKitVideoOutputPublish(unittest.IsolatedAsyncioTestCase):
    """Video output (publishing) support for the LiveKit transport.

    Mirrors how ``connect()`` always publishes one ``pipecat-audio``
    microphone track: when ``LiveKitParams.video_out_enabled`` is set, it
    should also publish one ``pipecat-video`` camera track using an
    ``rtc.VideoSource``/``rtc.LocalVideoTrack`` pair, the same way audio uses
    ``rtc.AudioSource``/``rtc.LocalAudioTrack``.
    """

    def _create_client(self, video_out_enabled: bool) -> LiveKitTransportClient:
        params = LiveKitParams(video_out_enabled=video_out_enabled)
        callbacks = LiveKitCallbacks(
            on_connected=AsyncMock(),
            on_disconnected=AsyncMock(),
            on_before_disconnect=AsyncMock(),
            on_participant_connected=AsyncMock(),
            on_participant_disconnected=AsyncMock(),
            on_audio_track_subscribed=AsyncMock(),
            on_audio_track_unsubscribed=AsyncMock(),
            on_video_track_subscribed=AsyncMock(),
            on_video_track_unsubscribed=AsyncMock(),
            on_data_received=AsyncMock(),
            on_first_participant_joined=AsyncMock(),
            on_dtmf_event=AsyncMock(),
            on_active_speaker_changed=AsyncMock(),
            on_video_track_muted=AsyncMock(),
        )
        client = LiveKitTransportClient(
            url="wss://test.livekit.cloud",
            token="test-token",
            room_name="test-room",
            params=params,
            callbacks=callbacks,
            transport_name="test-transport",
        )
        client._task_manager = MagicMock()
        # Normally set in setup(); connect() reads this directly.
        client._out_sample_rate = 16000

        mock_room = MagicMock()
        mock_room.connect = AsyncMock()
        mock_room.local_participant.publish_track = AsyncMock()
        mock_room.local_participant.identity = "bot"
        mock_room.remote_participants = {}
        client._room = mock_room
        return client

    async def test_video_out_disabled_publishes_only_audio_track(self):
        """When video output is disabled, only the microphone track is published."""
        client = self._create_client(video_out_enabled=False)

        with (
            patch.object(rtc, "AudioSource", return_value=MagicMock()),
            patch.object(rtc.LocalAudioTrack, "create_audio_track", return_value=MagicMock()),
            patch.object(rtc, "VideoSource") as mock_video_source_cls,
            patch.object(rtc.LocalVideoTrack, "create_video_track") as mock_create_video_track,
        ):
            await client.connect()

            mock_video_source_cls.assert_not_called()
            mock_create_video_track.assert_not_called()

        self.assertIsNone(client._video_source)
        self.assertIsNone(client._video_track)
        client.room.local_participant.publish_track.assert_awaited_once()

    async def test_video_out_enabled_publishes_audio_and_video_tracks(self):
        """When video output is enabled, both a microphone and a camera track are published."""
        client = self._create_client(video_out_enabled=True)

        mock_video_source = MagicMock()
        mock_video_track = MagicMock()

        with (
            patch.object(rtc, "AudioSource", return_value=MagicMock()),
            patch.object(rtc.LocalAudioTrack, "create_audio_track", return_value=MagicMock()),
            patch.object(
                rtc, "VideoSource", return_value=mock_video_source
            ) as mock_video_source_cls,
            patch.object(
                rtc.LocalVideoTrack, "create_video_track", return_value=mock_video_track
            ) as mock_create_video_track,
        ):
            await client.connect()

            mock_video_source_cls.assert_called_once_with(
                client._params.video_out_width, client._params.video_out_height
            )
            mock_create_video_track.assert_called_once_with("pipecat-video", mock_video_source)

        self.assertIs(client._video_source, mock_video_source)
        self.assertIs(client._video_track, mock_video_track)

        # Both the microphone and camera tracks got published.
        self.assertEqual(client.room.local_participant.publish_track.await_count, 2)
        published_tracks = [
            call.args[0] for call in client.room.local_participant.publish_track.await_args_list
        ]
        self.assertIn(mock_video_track, published_tracks)

        published_sources = [
            call.args[1].source
            for call in client.room.local_participant.publish_track.await_args_list
        ]
        self.assertIn(rtc.TrackSource.SOURCE_CAMERA, published_sources)
        self.assertIn(rtc.TrackSource.SOURCE_MICROPHONE, published_sources)

    def _video_publish_options(self, **params) -> "rtc.TrackPublishOptions":
        client = self._create_client(video_out_enabled=True)
        client._params = LiveKitParams(video_out_enabled=True, **params)
        return client._video_publish_options()

    async def test_video_publish_options_default_leaves_encoding_to_livekit(self):
        """Without a bitrate or codec, LiveKit chooses the encoding and codec."""
        options = self._video_publish_options()

        self.assertEqual(options.source, rtc.TrackSource.SOURCE_CAMERA)
        self.assertFalse(options.HasField("video_encoding"))
        self.assertFalse(options.HasField("video_codec"))

    async def test_video_publish_options_apply_bitrate_framerate_and_codec(self):
        """The max bitrate, framerate and codec params reach the publish options."""
        options = self._video_publish_options(
            video_out_max_bitrate=2_000_000, video_out_framerate=24, video_out_codec="h264"
        )

        self.assertEqual(options.video_encoding.max_bitrate, 2_000_000)
        self.assertEqual(options.video_encoding.max_framerate, 24)
        self.assertEqual(options.video_codec, rtc.VideoCodec.H264)

    async def test_video_publish_options_ignore_unknown_codec(self):
        """An unknown codec is ignored so LiveKit falls back to its default."""
        options = self._video_publish_options(video_out_codec="MPEG2")

        self.assertFalse(options.HasField("video_codec"))

    async def test_publish_video_writes_to_video_source(self):
        """``publish_video`` captures the frame on the connected VideoSource."""
        client = self._create_client(video_out_enabled=True)
        client._connected = True
        mock_video_source = MagicMock()
        client._video_source = mock_video_source

        video_frame = MagicMock()
        result = await client.publish_video(video_frame)

        self.assertTrue(result)
        mock_video_source.capture_frame.assert_called_once_with(video_frame)

    async def test_publish_video_without_source_returns_false(self):
        """``publish_video`` is a no-op when no video source has been set up."""
        client = self._create_client(video_out_enabled=False)
        client._connected = True

        result = await client.publish_video(MagicMock())

        self.assertFalse(result)

    async def test_failed_publish_is_rolled_back_so_retry_publishes(self):
        """A publish failure leaves the client disconnected, so a retry publishes again."""
        client = self._create_client(video_out_enabled=True)
        client.room.disconnect = AsyncMock()
        # Fail the video publish on the first attempt only.
        client.room.local_participant.publish_track = AsyncMock(
            side_effect=[None, RuntimeError("publish failed"), None, None]
        )
        audio_source = MagicMock()
        audio_source.aclose = AsyncMock()
        video_source = MagicMock()
        video_source.aclose = AsyncMock()
        # Call the undecorated method so the test controls each attempt.
        connect = LiveKitTransportClient.connect.__wrapped__

        with (
            patch.object(rtc, "AudioSource", return_value=audio_source),
            patch.object(rtc.LocalAudioTrack, "create_audio_track", return_value=MagicMock()),
            patch.object(rtc, "VideoSource", return_value=video_source),
            patch.object(rtc.LocalVideoTrack, "create_video_track", return_value=MagicMock()),
        ):
            with self.assertRaises(RuntimeError):
                await connect(client)

            self.assertFalse(client._connected)
            self.assertEqual(client._disconnect_counter, 0)
            client.room.disconnect.assert_awaited_once()
            audio_source.aclose.assert_awaited_once()
            video_source.aclose.assert_awaited_once()
            client._callbacks.on_connected.assert_not_awaited()

            await connect(client)

        self.assertTrue(client._connected)
        self.assertEqual(client._disconnect_counter, 1)
        self.assertEqual(client.room.local_participant.publish_track.await_count, 4)
        client._callbacks.on_connected.assert_awaited_once()

    async def test_room_disconnect_event_reports_disconnect_of_connected_client(self):
        """A disconnect the client did not start is reported once."""
        client = self._create_client(video_out_enabled=False)
        client._connected = True

        await client._async_on_disconnected()

        self.assertFalse(client._connected)
        client._callbacks.on_disconnected.assert_awaited_once()

    async def test_disconnect_reports_disconnect_once(self):
        """The room's event during disconnect() does not report a second disconnect."""
        client = self._create_client(video_out_enabled=False)
        client._connected = True
        client._disconnect_counter = 1

        async def room_disconnect():
            # LiveKit emits its "disconnected" event while disconnecting.
            await client._async_on_disconnected()

        client.room.disconnect = AsyncMock(side_effect=room_disconnect)

        await client.disconnect()

        client.room.disconnect.assert_awaited_once()
        client._callbacks.on_disconnected.assert_awaited_once()

    async def test_failed_connect_does_not_report_disconnect(self):
        """Rolling back a failed connect does not report a disconnect."""
        client = self._create_client(video_out_enabled=False)
        client.room.local_participant.publish_track = AsyncMock(
            side_effect=RuntimeError("publish failed")
        )

        async def room_disconnect():
            await client._async_on_disconnected()

        client.room.disconnect = AsyncMock(side_effect=room_disconnect)
        audio_source = MagicMock()
        audio_source.aclose = AsyncMock()

        with (
            patch.object(rtc, "AudioSource", return_value=audio_source),
            patch.object(rtc.LocalAudioTrack, "create_audio_track", return_value=MagicMock()),
        ):
            with self.assertRaises(RuntimeError):
                await LiveKitTransportClient.connect.__wrapped__(client)

        client.room.disconnect.assert_awaited_once()
        client._callbacks.on_disconnected.assert_not_awaited()

    async def test_disconnect_closes_output_sources(self):
        """Disconnecting closes the audio and video sources and forgets them."""
        client = self._create_client(video_out_enabled=True)
        client.room.disconnect = AsyncMock()

        audio_source = MagicMock()
        audio_source.aclose = AsyncMock()
        video_source = MagicMock()
        video_source.aclose = AsyncMock()

        with (
            patch.object(rtc, "AudioSource", return_value=audio_source),
            patch.object(rtc.LocalAudioTrack, "create_audio_track", return_value=MagicMock()),
            patch.object(rtc, "VideoSource", return_value=video_source),
            patch.object(rtc.LocalVideoTrack, "create_video_track", return_value=MagicMock()),
        ):
            await client.connect()
        await client.disconnect()

        audio_source.aclose.assert_awaited_once()
        video_source.aclose.assert_awaited_once()
        self.assertIsNone(client._audio_source)
        self.assertIsNone(client._audio_track)
        self.assertIsNone(client._video_source)
        self.assertIsNone(client._video_track)
        self.assertFalse(await client.publish_video(MagicMock()))


@unittest.skipUnless(LIVEKIT_AVAILABLE, "livekit package not installed")
class TestLiveKitOutputTransportWriteVideoFrame(unittest.IsolatedAsyncioTestCase):
    """``LiveKitOutputTransport.write_video_frame`` conversion and dispatch."""

    def _create_output_transport(self, **param_overrides) -> LiveKitOutputTransport:
        params = LiveKitParams(
            video_out_enabled=True,
            video_out_width=4,
            video_out_height=4,
            video_out_color_format="RGB",
            **param_overrides,
        )
        client = MagicMock()
        client.publish_video = AsyncMock(return_value=True)
        transport = MagicMock()
        return LiveKitOutputTransport(transport, client, params)

    async def test_write_video_frame_converts_and_publishes(self):
        """A supported color format is converted to an rtc.VideoFrame and published."""
        output = self._create_output_transport()
        image_bytes = bytes(range(4 * 4 * 3))
        frame = OutputImageRawFrame(image=image_bytes, size=(4, 4), format="RGB")

        result = await output.write_video_frame(frame)

        self.assertTrue(result)
        output._client.publish_video.assert_awaited_once()
        (livekit_frame,) = output._client.publish_video.await_args.args
        self.assertIsInstance(livekit_frame, rtc.VideoFrame)
        self.assertEqual(livekit_frame.width, 4)
        self.assertEqual(livekit_frame.height, 4)
        self.assertEqual(bytes(livekit_frame.data), image_bytes)

    async def test_write_video_frame_maps_four_byte_formats(self):
        """RGBA, BGRA and ARGB images map to the matching LiveKit buffer types."""
        expected_types = {
            "RGBA": rtc.VideoBufferType.RGBA,
            "BGRA": rtc.VideoBufferType.BGRA,
            "ARGB": rtc.VideoBufferType.ARGB,
        }
        for color_format, buffer_type in expected_types.items():
            with self.subTest(color_format=color_format):
                output = self._create_output_transport()
                image_bytes = bytes(range(4 * 4 * 4))
                frame = OutputImageRawFrame(image=image_bytes, size=(4, 4), format=color_format)

                result = await output.write_video_frame(frame)

                self.assertTrue(result)
                (livekit_frame,) = output._client.publish_video.await_args.args
                self.assertEqual(livekit_frame.type, buffer_type)
                self.assertEqual(bytes(livekit_frame.data), image_bytes)

    async def test_write_video_frame_unsupported_format_does_not_publish(self):
        """An unknown color format is rejected without touching the client."""
        output = self._create_output_transport()
        frame = OutputImageRawFrame(image=b"\x00" * 16, size=(4, 4), format="I420")

        result = await output.write_video_frame(frame)

        self.assertFalse(result)
        output._client.publish_video.assert_not_awaited()

    async def test_unsupported_format_error_is_logged_once(self):
        """Repeated frames with the same unsupported format log a single error."""
        output = self._create_output_transport()
        frame = OutputImageRawFrame(image=b"\x00" * 16, size=(4, 4), format="I420")

        with patch("pipecat.transports.livekit.transport.logger") as mock_logger:
            await output.write_video_frame(frame)
            await output.write_video_frame(frame)

        mock_logger.error.assert_called_once()

    async def test_write_video_frame_wrong_length_does_not_publish(self):
        """An image whose length does not match its size and format is rejected."""
        output = self._create_output_transport()
        frame = OutputImageRawFrame(image=b"\x00" * 10, size=(4, 4), format="RGB")

        result = await output.write_video_frame(frame)

        self.assertFalse(result)
        output._client.publish_video.assert_not_awaited()


@unittest.skipUnless(LIVEKIT_AVAILABLE, "livekit package not installed")
class TestLiveKitParticipantsAlreadyInRoom(unittest.IsolatedAsyncioTestCase):
    """Participants already in the room when the bot connects are reported."""

    def _create_client(self, identities: list[str]) -> LiveKitTransportClient:
        callbacks = LiveKitCallbacks(
            **{name: AsyncMock() for name in LiveKitCallbacks.model_fields}
        )
        client = LiveKitTransportClient(
            url="wss://test.livekit.cloud",
            token="test-token",
            room_name="test-room",
            params=LiveKitParams(),
            callbacks=callbacks,
            transport_name="test-transport",
        )
        client._task_manager = MagicMock()
        client._out_sample_rate = 16000

        room = MagicMock()
        room.connect = AsyncMock()
        room.local_participant.publish_track = AsyncMock()
        room.local_participant.identity = "bot"
        room.remote_participants = {
            identity: MagicMock(identity=identity) for identity in identities
        }
        client._room = room
        return client

    async def _connect(self, client):
        with (
            patch.object(rtc, "AudioSource", return_value=MagicMock()),
            patch.object(rtc.LocalAudioTrack, "create_audio_track", return_value=MagicMock()),
        ):
            await client.connect()

    async def test_each_participant_already_in_the_room_is_connected(self):
        client = self._create_client(["alice", "bob"])

        await self._connect(client)

        callbacks = client._callbacks
        self.assertEqual(
            [c.args for c in callbacks.on_participant_connected.await_args_list],
            [("alice",), ("bob",)],
        )
        callbacks.on_first_participant_joined.assert_awaited_once_with("alice")

    async def test_participant_who_joins_later_is_not_first(self):
        client = self._create_client(["alice"])
        await self._connect(client)

        participant = MagicMock()
        participant.identity = "bob"
        await client._async_on_participant_connected(participant)

        client._callbacks.on_first_participant_joined.assert_awaited_once_with("alice")
        client._callbacks.on_participant_connected.assert_awaited_with("bob")


def _video_publication(source: "rtc.TrackSource.ValueType") -> MagicMock:
    publication = MagicMock()
    publication.sid = f"video-{source}"
    publication.source = source
    publication.kind = rtc.TrackKind.KIND_VIDEO
    publication.muted = False
    return publication


@unittest.skipUnless(LIVEKIT_AVAILABLE, "livekit package not installed")
class TestLiveKitVideoSources(unittest.IsolatedAsyncioTestCase):
    """A user's camera and screen share are received side by side."""

    def _create_client(self) -> LiveKitTransportClient:
        callbacks = LiveKitCallbacks(
            **{name: AsyncMock() for name in LiveKitCallbacks.model_fields},
        )
        client = LiveKitTransportClient(
            url="wss://test.livekit.cloud",
            token="test-token",
            room_name="test-room",
            params=LiveKitParams(video_in_enabled=True),
            callbacks=callbacks,
            transport_name="test-transport",
        )
        client._task_manager = MagicMock()
        # Close the stream coroutines instead of running them.
        client._task_manager.create_task.side_effect = lambda coro, name: coro.close()
        return client

    async def _subscribe(self, client, source):
        track = MagicMock()
        track.kind = rtc.TrackKind.KIND_VIDEO
        participant = MagicMock()
        participant.identity = "alice"
        publication = _video_publication(source)
        stream = MagicMock()
        stream.aclose = AsyncMock()
        with patch.object(rtc, "VideoStream", return_value=stream):
            await client._async_on_track_subscribed(track, publication, participant)
        return track, publication, participant, stream

    async def test_camera_and_screen_share_are_received_side_by_side(self):
        client = self._create_client()
        await self._subscribe(client, rtc.TrackSource.SOURCE_CAMERA)
        await self._subscribe(client, rtc.TrackSource.SOURCE_SCREENSHARE)

        self.assertEqual(
            set(client._video_streams), {("alice", "camera"), ("alice", "screenVideo")}
        )
        self.assertTrue(client.video_source_enabled("alice", "screenVideo"))

    async def test_stopping_the_screen_share_keeps_the_camera(self):
        client = self._create_client()
        _, _, _, camera_stream = await self._subscribe(client, rtc.TrackSource.SOURCE_CAMERA)
        screen = await self._subscribe(client, rtc.TrackSource.SOURCE_SCREENSHARE)

        await client._async_on_track_unsubscribed(*screen[:3])

        camera_stream.aclose.assert_not_awaited()
        self.assertEqual(set(client._video_streams), {("alice", "camera")})
        self.assertFalse(client.video_source_enabled("alice", "screenVideo"))

    async def test_a_muted_track_sends_no_video(self):
        client = self._create_client()
        _, publication, participant, _ = await self._subscribe(
            client, rtc.TrackSource.SOURCE_CAMERA
        )

        publication.muted = True
        await client._async_on_track_muted(participant, publication)

        self.assertFalse(client.video_source_enabled("alice", "camera"))
        client._callbacks.on_video_track_muted.assert_awaited_once_with("alice", "camera")

    async def test_video_without_a_source_is_the_camera(self):
        client = self._create_client()
        await self._subscribe(client, rtc.TrackSource.SOURCE_UNKNOWN)

        self.assertTrue(client.video_source_enabled("alice", "camera"))


@unittest.skipUnless(LIVEKIT_AVAILABLE, "livekit package not installed")
class TestLiveKitInputVideoSampling(unittest.IsolatedAsyncioTestCase):
    """Which incoming video frames the input transport passes on."""

    def _input(self, frames, **params):
        async def next_frames():
            for frame in frames:
                yield frame

        client = MagicMock()
        client.get_next_video_frame = next_frames
        client.video_source_enabled = lambda participant_id, video_source: True
        input = LiveKitInputTransport(
            MagicMock(), client, LiveKitParams(video_in_enabled=True, **params)
        )
        input._convert_livekit_video_to_pipecat = AsyncMock(
            return_value=ImageRawFrame(image=b"\x00\x00\x00", size=(1, 1), format="RGB")
        )
        input.pushed = []

        async def collect(frame):
            input.pushed.append(frame)

        input.push_video_frame = collect
        return input

    async def test_every_frame_of_every_source_without_video_in_sources(self):
        frames = [(MagicMock(), "alice", "camera"), (MagicMock(), "alice", "screenVideo")] * 3
        input = self._input(frames)

        await input._video_in_task_handler()

        self.assertEqual(
            [frame.transport_source for frame in input.pushed],
            ["camera", "screenVideo"] * 3,
        )

    async def test_only_listed_sources_with_video_in_sources(self):
        frames = [(MagicMock(), "alice", "camera"), (MagicMock(), "alice", "screenVideo")]
        input = self._input(frames, video_in_sources={"screenVideo": VideoInSourceParams()})

        await input._video_in_task_handler()

        self.assertEqual([frame.transport_source for frame in input.pushed], ["screenVideo"])

    async def test_explicit_capture_takes_precedence(self):
        frames = [(MagicMock(), "alice", "camera")] * 3
        input = self._input(frames, video_in_sources={"camera": VideoInSourceParams()})
        await input.capture_participant_video("alice", video_source="camera", on_request_only=True)

        await input._video_in_task_handler()

        self.assertEqual(input.pushed, [])

    async def test_request_answered_by_the_next_frame(self):
        frames = [(MagicMock(), "alice", "camera")] * 2
        input = self._input(
            frames, video_in_sources={"camera": VideoInSourceParams(on_request_only=True)}
        )
        request = UserImageRequestFrame(user_id="alice", text="What is this?")
        await input.request_participant_image(request)

        await input._video_in_task_handler()

        self.assertEqual(len(input.pushed), 1)
        self.assertIs(input.pushed[0].request, request)
        self.assertEqual(input.pushed[0].text, "What is this?")

    async def test_request_without_a_track_completes_with_an_error(self):
        input = self._input([])
        input._client.video_source_enabled = lambda participant_id, video_source: False
        result_callback = AsyncMock()

        await input.request_participant_image(
            UserImageRequestFrame(
                user_id="alice", video_source="screenVideo", result_callback=result_callback
            )
        )

        result_callback.assert_awaited_once()
        self.assertIn("screenVideo", result_callback.await_args.args[0]["error"])

    async def test_a_participant_who_leaves_is_forgotten(self):
        transport = LiveKitTransport(
            url="wss://test.livekit.cloud",
            token="t",
            room_name="r",
            params=LiveKitParams(video_in_enabled=True),
        )
        await transport.input().capture_participant_video("alice", 1, "camera")

        await transport._on_participant_disconnected("alice")

        self.assertFalse(transport.input()._video_samplers.is_capturing("alice", "camera"))

    async def _waiting_request(self, input, video_source="camera"):
        result_callback = AsyncMock()
        await input.request_participant_image(
            UserImageRequestFrame(
                user_id="alice", video_source=video_source, result_callback=result_callback
            )
        )
        result_callback.assert_not_awaited()
        return result_callback

    async def test_request_after_the_track_stopped_is_answered_with_an_error(self):
        input = self._input([], video_in_sources={"screenVideo": VideoInSourceParams()})
        input._capture_configured_video("alice", "screenVideo")
        input._client.video_source_enabled = lambda participant_id, video_source: False
        result_callback = AsyncMock()

        await input.request_participant_image(
            UserImageRequestFrame(
                user_id="alice", video_source="screenVideo", result_callback=result_callback
            )
        )

        self.assertIn("error", result_callback.await_args.args[0])

    async def test_waiting_request_is_answered_when_its_track_stops(self):
        input = self._input([], video_in_sources={"screenVideo": VideoInSourceParams()})
        result_callback = await self._waiting_request(input, "screenVideo")

        await input.stop_participant_video("alice", "screenVideo")

        self.assertIn("stopped", result_callback.await_args.args[0]["error"])
        self.assertTrue(input._video_samplers.is_capturing("alice", "screenVideo"))

    async def test_waiting_request_is_answered_when_the_participant_leaves(self):
        input = self._input([])
        result_callback = await self._waiting_request(input)

        await input.remove_participant_video("alice")

        self.assertIn("left", result_callback.await_args.args[0]["error"])

    async def test_waiting_requests_are_answered_when_the_room_disconnects(self):
        transport = LiveKitTransport(
            url="wss://test.livekit.cloud",
            token="t",
            room_name="r",
            params=LiveKitParams(video_in_enabled=True),
        )
        input = transport.input()
        input._client = MagicMock(video_source_enabled=lambda participant_id, video_source: True)
        result_callback = await self._waiting_request(input)

        await transport._on_disconnected()

        self.assertIn("error", result_callback.await_args.args[0])

    def test_screen_in_capability(self):
        input = LiveKitInputTransport(
            MagicMock(),
            MagicMock(),
            LiveKitParams(
                video_in_enabled=True, video_in_sources={"screenVideo": VideoInSourceParams()}
            ),
        )
        self.assertTrue(input.capabilities.screen_in)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(LIVEKIT_AVAILABLE, "livekit package not installed")
class TestLiveKitAudioOutQueueSize(unittest.IsolatedAsyncioTestCase):
    """``audio_out_queue_size_ms`` sizes the outgoing ``rtc.AudioSource`` buffer."""

    def _create_client(self, params: LiveKitParams) -> LiveKitTransportClient:
        callbacks = LiveKitCallbacks(
            on_connected=AsyncMock(),
            on_disconnected=AsyncMock(),
            on_before_disconnect=AsyncMock(),
            on_participant_connected=AsyncMock(),
            on_participant_disconnected=AsyncMock(),
            on_audio_track_subscribed=AsyncMock(),
            on_audio_track_unsubscribed=AsyncMock(),
            on_video_track_subscribed=AsyncMock(),
            on_video_track_unsubscribed=AsyncMock(),
            on_data_received=AsyncMock(),
            on_first_participant_joined=AsyncMock(),
            on_dtmf_event=AsyncMock(),
            on_active_speaker_changed=AsyncMock(),
            on_video_track_muted=AsyncMock(),
        )
        client = LiveKitTransportClient(
            url="wss://test.livekit.cloud",
            token="test-token",
            room_name="test-room",
            params=params,
            callbacks=callbacks,
            transport_name="test-transport",
        )
        client._task_manager = MagicMock()
        client._out_sample_rate = 16000
        room = MagicMock()
        room.connect = AsyncMock()
        room.local_participant.identity = "bot"
        room.local_participant.publish_track = AsyncMock()
        room.remote_participants = {}
        client._room = room
        return client

    async def _connect_and_get_audio_source_call(self, params: LiveKitParams):
        client = self._create_client(params)
        with (
            patch.object(rtc, "AudioSource") as audio_source,
            patch.object(rtc.LocalAudioTrack, "create_audio_track"),
        ):
            await client.connect()
        return audio_source.call_args

    async def test_default_matches_livekit_default(self):
        call = await self._connect_and_get_audio_source_call(LiveKitParams())
        self.assertEqual(call.kwargs["queue_size_ms"], 1000)

    async def test_queue_size_is_passed_to_the_audio_source(self):
        call = await self._connect_and_get_audio_source_call(
            LiveKitParams(audio_out_queue_size_ms=200)
        )
        self.assertEqual(call.args, (16000, 1))
        self.assertEqual(call.kwargs["queue_size_ms"], 200)


def _pcm(value: int, ms: int, sample_rate: int = 48000) -> bytes:
    """Constant-valued 16-bit mono PCM."""
    return np.full(sample_rate * ms // 1000, value, dtype=np.int16).tobytes()


def _samples(audio: bytes) -> np.ndarray:
    return np.frombuffer(audio, dtype=np.int16)


@unittest.skipUnless(LIVEKIT_AVAILABLE, "livekit package not installed")
class TestParticipantAudioMixer(unittest.TestCase):
    """Participants' audio is mixed into one stream that keeps their pace."""

    def _mixer(self, *participant_ids: str, **kwargs) -> _ParticipantAudioMixer:
        """A mixer that has already heard from ``participant_ids``."""
        mixer = _ParticipantAudioMixer(sample_rate=48000, num_channels=1, **kwargs)
        for participant_id in participant_ids:
            mixer.add(participant_id, b"")
        return mixer

    def test_a_single_participant_passes_through_in_chunks(self):
        mixer = self._mixer()
        chunks = mixer.add("alice", _pcm(100, 30))
        self.assertEqual(len(chunks), 3)
        for chunk in chunks:
            self.assertEqual(len(chunk), 960)
            self.assertTrue(np.all(_samples(chunk) == 100))

    def test_partial_chunks_are_held_until_complete(self):
        mixer = self._mixer()
        self.assertEqual(mixer.add("alice", _pcm(100, 5)), [])
        self.assertEqual(len(mixer.add("alice", _pcm(100, 5))), 1)

    def test_participants_are_summed_and_keep_real_time(self):
        mixer = self._mixer("alice", "bob")
        chunks = []
        for _ in range(100):  # one second of 10 ms frames from each participant
            chunks += mixer.add("alice", _pcm(1000, 10))
            chunks += mixer.add("bob", _pcm(2000, 10))
        self.assertEqual(len(chunks), 100)
        self.assertTrue(all(np.all(_samples(chunk) == 3000) for chunk in chunks))

    def test_waits_for_every_participant_before_mixing(self):
        mixer = self._mixer("alice", "bob")
        self.assertEqual(mixer.add("alice", _pcm(1000, 30)), [])
        chunks = mixer.add("bob", _pcm(2000, 30))
        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(np.all(_samples(chunk) == 3000) for chunk in chunks))

    def test_a_participant_who_stops_sending_is_left_out(self):
        mixer = self._mixer("alice", "bob", max_wait_ms=50)

        # Bob is quiet: alice's audio waits for him until 50 ms is buffered,
        # then comes out on its own, and keeps pace without him after that.
        self.assertEqual(mixer.add("alice", _pcm(1000, 40)), [])
        chunks = mixer.add("alice", _pcm(1000, 10))
        self.assertEqual(len(chunks), 5)
        self.assertTrue(all(np.all(_samples(chunk) == 1000) for chunk in chunks))
        self.assertEqual(len(mixer.add("alice", _pcm(1000, 10))), 1)

        # Bob's audio is mixed in again as soon as it arrives.
        self.assertEqual(mixer.add("bob", _pcm(2000, 10)), [])
        chunks = mixer.add("alice", _pcm(1000, 10))
        self.assertEqual(len(chunks), 1)
        self.assertTrue(np.all(_samples(chunks[0]) == 3000))

    def test_mixed_samples_are_clipped(self):
        mixer = self._mixer("alice", "bob")
        mixer.add("alice", _pcm(30000, 10))
        (chunk,) = mixer.add("bob", _pcm(30000, 10))
        self.assertTrue(np.all(_samples(chunk) == 32767))


@unittest.skipUnless(LIVEKIT_AVAILABLE, "livekit package not installed")
class TestLiveKitAudioInUserTracks(unittest.IsolatedAsyncioTestCase):
    """How the input transport delivers several participants' audio."""

    async def _push_audio_from_two_participants(self, audio_in_user_tracks: bool):
        from pipecat.transports.livekit.transport import LiveKitTransport

        transport = LiveKitTransport(
            url="wss://test.livekit.cloud",
            token="test-token",
            room_name="test-room",
            params=LiveKitParams(audio_in_enabled=True, audio_in_user_tracks=audio_in_user_tracks),
        )
        input_transport = transport.input()
        input_transport._sample_rate = 16000
        input_transport.push_audio_frame = AsyncMock()

        async def frames():
            for _ in range(100):  # one second of 10 ms frames from each participant
                for participant_id, value in (("alice", 1000), ("bob", 2000)):
                    frame = rtc.AudioFrame(_pcm(value, 10), 48000, 1, 480)
                    yield rtc.AudioFrameEvent(frame=frame), participant_id

        transport._client.get_next_audio_frame = frames
        await input_transport._audio_in_task_handler()
        return [call.args[0] for call in input_transport.push_audio_frame.await_args_list]

    async def test_user_tracks_push_each_participants_audio(self):
        frames = await self._push_audio_from_two_participants(audio_in_user_tracks=True)
        self.assertTrue(all(isinstance(frame, UserAudioRawFrame) for frame in frames))
        self.assertEqual({frame.user_id for frame in frames}, {"alice", "bob"})

    async def test_mixed_audio_is_one_real_time_stream(self):
        frames = await self._push_audio_from_two_participants(audio_in_user_tracks=False)
        self.assertTrue(all(type(frame) is InputAudioRawFrame for frame in frames))
        self.assertTrue(all(frame.sample_rate == 16000 for frame in frames))

        # One second from each of two participants is one second of audio, not
        # two (less what the resampler still holds).
        seconds = sum(len(frame.audio) for frame in frames) / 2 / 16000
        self.assertGreater(seconds, 0.9)
        self.assertLessEqual(seconds, 1.0)

        # The middle of the stream carries both participants' audio.
        middle = _samples(b"".join(frame.audio for frame in frames))[4000:12000]
        self.assertTrue(np.all(np.abs(middle - 3000) <= 3))
