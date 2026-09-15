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

try:
    from livekit import rtc

    from pipecat.frames.frames import OutputImageRawFrame
    from pipecat.transports.livekit.transport import (
        LiveKitCallbacks,
        LiveKitOutputTransport,
        LiveKitParams,
        LiveKitTransportClient,
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
        self.assertIn(participant.identity, client._video_tracks)

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
        self.assertIn(participant.identity, client._video_tracks)

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
        self.assertIn(participant.identity, client._video_streams)

        await client._async_on_track_unsubscribed(track, pub, participant)
        mock_stream.aclose.assert_awaited_once()
        self.assertNotIn(participant.identity, client._video_streams)


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
