#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

from pydantic import ValidationError

from pipecat.frames.frames import UserImageRawFrame, UserImageRequestFrame
from pipecat.transports.base_transport import TransportParams, VideoInSourceParams
from pipecat.transports.video_in_sampler import JITTER_TOLERANCE_SECS, _VideoInSampler


class TestVideoInSourcesParams(unittest.TestCase):
    def test_defaults_to_no_sources(self):
        self.assertEqual(TransportParams().video_in_sources, {})

    def test_sources_keyed_by_source(self):
        params = TransportParams(
            video_in_enabled=True,
            video_in_sources={
                "camera": VideoInSourceParams(on_request_only=True),
                "screenVideo": VideoInSourceParams(framerate=1),
            },
        )
        self.assertTrue(params.video_in_sources["camera"].on_request_only)
        self.assertEqual(params.video_in_sources["screenVideo"].framerate, 1)

    def test_source_params_from_dict(self):
        params = TransportParams(video_in_enabled=True, video_in_sources={"camera": {}})
        self.assertEqual(params.video_in_sources["camera"], VideoInSourceParams())

    def test_default_framerate(self):
        self.assertEqual(VideoInSourceParams().framerate, 30)

    def test_framerate_below_one_rejected(self):
        with self.assertRaises(ValidationError):
            VideoInSourceParams(framerate=-1)

    def test_zero_framerate_points_to_on_request_only(self):
        with self.assertRaisesRegex(ValidationError, "on_request_only=True"):
            VideoInSourceParams(framerate=0)

    def test_framerate_with_on_request_only_rejected(self):
        with self.assertRaises(ValidationError):
            VideoInSourceParams(framerate=5, on_request_only=True)

    def test_sources_require_video_in_enabled(self):
        with self.assertRaises(ValidationError):
            TransportParams(video_in_sources={"camera": VideoInSourceParams()})

    def test_sources_survive_model_dump(self):
        params = TransportParams(
            video_in_enabled=True, video_in_sources={"screenVideo": VideoInSourceParams()}
        )
        self.assertEqual(TransportParams(**params.model_dump()), params)


class TestVideoInSampler(unittest.TestCase):
    def _sample_at(self, sampler: _VideoInSampler, now: float):
        with patch("pipecat.transports.video_in_sampler.time.time", return_value=now):
            return sampler.sample()

    def _passed(self, sampler: _VideoInSampler, times) -> list[float]:
        return [t for t in times if self._sample_at(sampler, t)[0]]

    def test_none_passes_every_frame(self):
        sampler = _VideoInSampler(None)
        self.assertEqual(self._passed(sampler, (1.0, 1.01, 1.02)), [1.0, 1.01, 1.02])

    def test_zero_passes_only_requested_frames(self):
        sampler = _VideoInSampler(0)
        self.assertFalse(self._sample_at(sampler, 1.0)[0])
        request = UserImageRequestFrame(user_id="u")
        sampler.add_request(request)
        self.assertEqual(self._sample_at(sampler, 1.1), (True, request))
        self.assertFalse(self._sample_at(sampler, 1.2)[0])

    def test_framerate_below_input_rate(self):
        # A 30 fps stream sampled at 5 fps for ten seconds.
        times = [10 + i / 30 for i in range(300)]
        passed = self._passed(_VideoInSampler(5), times)
        self.assertAlmostEqual(len(passed) / 10, 5, delta=0.1)
        # Every gap after the first is within a stream frame of the interval.
        gaps = [b - a for a, b in zip(passed[1:], passed[2:])]
        self.assertTrue(all(abs(gap - 0.2) < 1 / 30 for gap in gaps), gaps)

    def test_framerate_at_input_rate_passes_every_frame(self):
        # A slightly slow 30 fps stream with jitter, sampled at 30 fps.
        times, t = [], 10.0
        for i in range(90):
            t += (1 / 29.5) * (0.8 if i % 2 else 1.2)
            times.append(t)
        self.assertEqual(self._passed(_VideoInSampler(30), times), times)

    def test_next_frame_one_interval_after_the_first(self):
        times = [10 + i / 30 for i in range(45)]
        passed = self._passed(_VideoInSampler(1), times)
        self.assertGreaterEqual(passed[1] - passed[0], 1.0 - JITTER_TOLERANCE_SECS)

    def test_no_close_frames_after_a_late_frame(self):
        # At 1 fps, the stream pauses for 1.2 s after a frame is passed on, so
        # the next frame arrives late but not long enough for a stall.
        times = [10 + i / 30 for i in range(31)] + [11.2 + i / 30 for i in range(60)]
        passed = self._passed(_VideoInSampler(1), times)
        gaps = [b - a for a, b in zip(passed, passed[1:])]
        self.assertTrue(all(gap >= 1.0 - JITTER_TOLERANCE_SECS for gap in gaps), gaps)

    def test_request_answered_between_due_frames(self):
        sampler = _VideoInSampler(1)
        self._sample_at(sampler, 10.0)
        request = UserImageRequestFrame(user_id="u")
        sampler.add_request(request)
        self.assertEqual(self._sample_at(sampler, 10.2), (True, request))
        # The schedule restarts from the request: nothing until a second later.
        self.assertFalse(self._sample_at(sampler, 11.0)[0])
        self.assertTrue(self._sample_at(sampler, 11.2)[0])

    def test_no_burst_after_a_stall(self):
        sampler = _VideoInSampler(1)
        times = [10 + i / 30 for i in range(30)] + [20 + i / 30 for i in range(30)]
        passed = self._passed(sampler, times)
        # One frame when the stream resumes, then the next an interval later.
        self.assertEqual(len([t for t in passed if 20 <= t < 20.9]), 1)

    def test_requests_answered_in_order(self):
        sampler = _VideoInSampler(0)
        first, second = UserImageRequestFrame(user_id="a"), UserImageRequestFrame(user_id="b")
        sampler.add_request(first)
        sampler.add_request(second)
        self.assertIs(self._sample_at(sampler, 1.0)[1], first)
        self.assertIs(self._sample_at(sampler, 1.1)[1], second)

    def test_framerate_can_change(self):
        sampler = _VideoInSampler(None)
        sampler.framerate = 0
        self.assertFalse(self._sample_at(sampler, 1.0)[0])


def _image_frame() -> UserImageRawFrame:
    return UserImageRawFrame(user_id="peer", image=b"", size=(1, 1), format="RGB")


class TestDailyVideoInSampling(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        try:
            from pipecat.transports.daily.transport import DailyInputTransport
        except Exception as e:
            self.skipTest(f"Daily transport unavailable: {e}")
        self.transport_cls = DailyInputTransport

    def _fake_input(self, framerate: int):
        fake = MagicMock()
        fake._video_samplers = {"p1": {"camera": _VideoInSampler(framerate)}}
        fake.push_video_frame = AsyncMock()
        return fake

    async def test_request_answered_by_next_frame(self):
        fake = self._fake_input(0)
        video_frame = SimpleNamespace(buffer=b"", width=1, height=1, color_format="RGB")

        await self.transport_cls._on_participant_video_frame(fake, "p1", video_frame, "camera")
        fake.push_video_frame.assert_not_called()

        request = UserImageRequestFrame(user_id="p1", text="what's this?")
        await self.transport_cls.request_participant_image(fake, request)
        await self.transport_cls._on_participant_video_frame(fake, "p1", video_frame, "camera")

        frame = fake.push_video_frame.call_args[0][0]
        self.assertIs(frame.request, request)
        self.assertEqual(frame.text, "what's this?")
        self.assertEqual(frame.transport_source, "camera")

    async def test_request_for_uncaptured_source_is_answered_with_an_error(self):
        fake = self._fake_input(0)
        result_callback = AsyncMock()
        request = UserImageRequestFrame(
            user_id="p1", video_source="screenVideo", result_callback=result_callback
        )

        await self.transport_cls.request_participant_image(fake, request)

        result_callback.assert_awaited_once()
        self.assertIn("error", result_callback.await_args.args[0])
        self.assertEqual(fake._video_samplers["p1"].keys(), {"camera"})

    async def test_request_for_uncaptured_participant_is_answered_with_an_error(self):
        fake = self._fake_input(0)
        result_callback = AsyncMock()
        request = UserImageRequestFrame(user_id="p2", result_callback=result_callback)

        await self.transport_cls.request_participant_image(fake, request)

        result_callback.assert_awaited_once()
        self.assertIn("error", result_callback.await_args.args[0])


class TestDailyVideoInSourcesCapture(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        try:
            from pipecat.transports.daily.transport import DailyParams, DailyTransport
        except Exception as e:
            self.skipTest(f"Daily transport unavailable: {e}")
        self.transport_cls = DailyTransport
        self.params_cls = DailyParams

    def _fake_transport(self, params):
        # Captures and event handler calls share one parent mock, so their order
        # is recorded in calls.mock_calls.
        calls = MagicMock(capture=AsyncMock(), event=AsyncMock())
        fake = MagicMock()
        fake._params = params
        fake._other_participant_has_joined = True
        fake._input.capture_participant_video = calls.capture
        fake._input.capture_participant_audio = AsyncMock()
        fake._input.push_frame = AsyncMock()
        fake._call_event_handler = calls.event
        return fake, calls

    async def test_captures_configured_sources_on_join(self):
        params = self.params_cls(
            video_in_enabled=True,
            video_in_sources={
                "camera": VideoInSourceParams(on_request_only=True),
                "screenVideo": VideoInSourceParams(framerate=1),
            },
        )
        fake, calls = self._fake_transport(params)

        await self.transport_cls._on_participant_joined(fake, {"id": "p1"})

        self.assertEqual(
            calls.capture.await_args_list,
            [call("p1", 0, "camera"), call("p1", 1, "screenVideo")],
        )

    async def test_captures_before_event_handlers(self):
        params = self.params_cls(
            video_in_enabled=True, video_in_sources={"camera": VideoInSourceParams()}
        )
        fake, calls = self._fake_transport(params)

        await self.transport_cls._on_participant_joined(fake, {"id": "p1"})

        names = [name for name, _args, _kwargs in calls.mock_calls]
        self.assertEqual(names[0], "capture")
        self.assertIn("event", names)

    async def test_no_sources_captures_nothing(self):
        fake, calls = self._fake_transport(self.params_cls(video_in_enabled=True))

        await self.transport_cls._on_participant_joined(fake, {"id": "p1"})

        calls.capture.assert_not_awaited()


class TestSmallWebRTCVideoInSampling(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        try:
            from pipecat.transports.smallwebrtc.transport import SmallWebRTCInputTransport
        except Exception as e:
            self.skipTest(f"SmallWebRTC transport unavailable: {e}")
        self.transport_cls = SmallWebRTCInputTransport

    def _fake_input(self, frames: list[UserImageRawFrame], samplers: dict):
        async def read_video_frame(_source):
            for frame in frames:
                yield frame

        fake = MagicMock()
        fake._client.read_video_frame = read_video_frame
        fake._video_samplers = samplers
        fake._params = TransportParams(video_in_enabled=True)
        fake.push_video_frame = AsyncMock()
        return fake

    def _pushed(self, fake) -> list[UserImageRawFrame]:
        return [awaited.args[0] for awaited in fake.push_video_frame.await_args_list]

    async def test_unthrottled_source_passes_every_frame(self):
        fake = self._fake_input(
            [_image_frame() for _ in range(3)], {"camera": _VideoInSampler(None)}
        )
        await self.transport_cls._receive_video(fake, "camera")
        self.assertEqual(fake.push_video_frame.await_count, 3)

    async def test_request_rides_on_the_pushed_frame(self):
        sampler = _VideoInSampler(0)
        request = UserImageRequestFrame(
            user_id="requester", text="describe", append_to_context=True
        )
        sampler.add_request(request)
        fake = self._fake_input([_image_frame(), _image_frame()], {"camera": sampler})

        await self.transport_cls._receive_video(fake, "camera")

        (frame,) = self._pushed(fake)
        self.assertIs(frame.request, request)
        self.assertEqual(frame.text, "describe")
        self.assertTrue(frame.append_to_context)
        self.assertEqual(frame.user_id, "peer")

    async def test_waiting_requests_answered_by_successive_frames(self):
        sampler = _VideoInSampler(0)
        requests = [UserImageRequestFrame(user_id="a"), UserImageRequestFrame(user_id="b")]
        for request in requests:
            sampler.add_request(request)
        fake = self._fake_input([_image_frame(), _image_frame()], {"camera": sampler})

        await self.transport_cls._receive_video(fake, "camera")

        self.assertEqual([f.request for f in self._pushed(fake)], requests)

    async def test_request_starts_an_uncaptured_source(self):
        fake = self._fake_input([], {})
        fake._receive_video_task = None
        request = UserImageRequestFrame(user_id="requester")

        await self.transport_cls.request_participant_image(fake, request)

        sampler = fake._video_samplers["camera"]
        self.assertIsNone(sampler.framerate)
        self.assertIs(sampler.sample()[1], request)
        fake.create_task.assert_called_once()

    async def test_capture_sets_framerate_without_dropping_requests(self):
        fake = self._fake_input([], {})
        fake._receive_video_task = object()
        request = UserImageRequestFrame(user_id="requester")

        await self.transport_cls.request_participant_image(fake, request)
        await self.transport_cls.capture_participant_media(fake, source="camera", framerate=1)

        sampler = fake._video_samplers["camera"]
        self.assertEqual(sampler.framerate, 1)
        self.assertIs(sampler.sample()[1], request)


class TestSmallWebRTCVideoInSourcesCapture(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        try:
            from pipecat.transports.smallwebrtc import transport
        except Exception as e:
            self.skipTest(f"SmallWebRTC transport unavailable: {e}")
        self.module = transport

    def _fake_transport(self, params):
        # Captures and event handler calls share one parent mock, so their order
        # is recorded in calls.mock_calls.
        calls = MagicMock(capture=AsyncMock(), event=AsyncMock())
        fake = MagicMock()
        fake._params = params
        fake._input.capture_participant_media = calls.capture
        fake._input.push_frame = AsyncMock()
        fake._call_event_handler = calls.event
        return fake, calls

    async def test_captures_configured_sources_on_connect(self):
        params = TransportParams(
            video_in_enabled=True,
            video_in_sources={
                "camera": VideoInSourceParams(on_request_only=True),
                "screenVideo": VideoInSourceParams(framerate=1),
            },
        )
        fake, calls = self._fake_transport(params)

        await self.module.SmallWebRTCTransport._on_client_connected(fake, MagicMock())

        self.assertEqual(
            calls.capture.await_args_list,
            [call(source="camera", framerate=0), call(source="screenVideo", framerate=1)],
        )
        names = [name for name, _args, _kwargs in calls.mock_calls]
        self.assertEqual(names[:2], ["capture", "capture"])
        self.assertIn("event", names)

    async def test_unsupported_source_warned_and_not_captured(self):
        params = TransportParams(
            video_in_enabled=True,
            video_in_sources={"camera": VideoInSourceParams(), "microphone": VideoInSourceParams()},
        )
        with patch.object(self.module, "logger") as logger:
            self.module.SmallWebRTCTransport(webrtc_connection=MagicMock(), params=params)
        logger.warning.assert_called_once()
        self.assertIn("microphone", logger.warning.call_args.args[0])

        fake, calls = self._fake_transport(params)
        await self.module.SmallWebRTCTransport._on_client_connected(fake, MagicMock())
        self.assertEqual(calls.capture.await_args_list, [call(source="camera", framerate=30)])

    async def test_no_sources_captures_nothing(self):
        fake, calls = self._fake_transport(TransportParams(video_in_enabled=True))

        await self.module.SmallWebRTCTransport._on_client_connected(fake, MagicMock())

        calls.capture.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
