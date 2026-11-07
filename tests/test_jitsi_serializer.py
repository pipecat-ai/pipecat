#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for JitsiFrameSerializer.

The gateway interleaves every participant's 20 ms frames on one socket, each tagged
with its source, so the serializer's job beyond base64 and JSON is to demultiplex: mix
the concurrent participants into one stream, or emit one stream per participant.
deserialize() ignores malformed JSON and unknown events rather than raising, since the
transport's receive loop ends the session on any exception.
"""

import base64
import json
import unittest

import numpy as np

from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    UserAudioRawFrame,
)
from pipecat.serializers.jitsi import (
    FRAME_MS,
    JITSI_PCM_SAMPLE_RATE,
    JitsiFrameSerializer,
    _SourceMixer,
)
from tests.frame_processor_helpers import frame_processor_setup

SAMPLES = JITSI_PCM_SAMPLE_RATE * FRAME_MS // 1000


def pcm(value: int, samples: int = SAMPLES) -> bytes:
    """One frame of PCM16 holding a constant sample value."""
    return np.full(samples, value, dtype="<i2").tobytes()


def media(tag: str, value: int, samples: int = SAMPLES) -> str:
    return json.dumps(
        {
            "event": "media",
            "media": {
                "tag": tag,
                "chunk": 0,
                "timestamp": 0,
                "payload": base64.b64encode(pcm(value, samples)).decode(),
            },
        }
    )


def samples_of(frames) -> np.ndarray:
    return np.concatenate([np.frombuffer(f.audio, dtype="<i2") for f in frames])


async def make_serializer(**params) -> JitsiFrameSerializer:
    serializer = JitsiFrameSerializer(
        JitsiFrameSerializer.InputParams(sample_rate=JITSI_PCM_SAMPLE_RATE, **params)
    )
    await serializer.setup(
        frame_processor_setup(
            audio_in_sample_rate=JITSI_PCM_SAMPLE_RATE, audio_out_sample_rate=JITSI_PCM_SAMPLE_RATE
        )
    )
    return serializer


async def feed(serializer, messages):
    frames = [await serializer.deserialize(m) for m in messages]
    return [f for f in frames if f is not None]


class TestJitsiDeserializeRobustness(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.serializer = await make_serializer()

    async def test_malformed_json_is_ignored_not_raised(self):
        self.assertIsNone(await self.serializer.deserialize("not json"))

    async def test_non_object_json_is_ignored_not_raised(self):
        self.assertIsNone(await self.serializer.deserialize("[1, 2]"))

    async def test_media_event_missing_payload_is_ignored_not_raised(self):
        frame = await self.serializer.deserialize(json.dumps({"event": "media", "media": {}}))
        self.assertIsNone(frame)

    async def test_unknown_events_are_ignored(self):
        for message in (
            {"event": "speakers", "speakers": ["aa-a0"], "silence": False},
            {"event": "mark", "mark": {"name": "m1"}},
            {"event": "ping", "id": 1},
            {"event": "something-new", "payload": {}},
            {"foo": "bar"},
        ):
            with self.subTest(event=message.get("event")):
                self.assertIsNone(await self.serializer.deserialize(json.dumps(message)))


class TestJitsiHandshake(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.serializer = await make_serializer()

    async def test_info_is_recorded(self):
        info = {
            "event": "info",
            "protocol": "jitsi-agent-media",
            "version": "1.1",
            "mediaFormat": {"encoding": "audio/l16", "sampleRate": 24000, "channels": 1},
            "sessionId": "s1",
            "agentId": "agent-support",
            "conference": "acme/room1",
            "customParameters": {"tenant": "acme"},
        }
        self.assertIsNone(await self.serializer.deserialize(json.dumps(info)))
        self.assertEqual(self.serializer.info["agentId"], "agent-support")
        self.assertEqual(self.serializer.info["conference"], "acme/room1")
        self.assertEqual(self.serializer.custom_parameters, {"tenant": "acme"})

    async def test_start_sample_rate_is_honored(self):
        start = {
            "event": "start",
            "start": {
                "tag": "aa-a0",
                "mediaFormat": {"encoding": "audio/l16", "sampleRate": 16000, "channels": 1},
                "customParameters": {"lang": "fr"},
            },
        }
        self.assertIsNone(await self.serializer.deserialize(json.dumps(start)))
        self.assertEqual(self.serializer.custom_parameters, {"lang": "fr"})

        # 20 ms frames at the wire's 16 kHz come out at the pipeline's 24 kHz, less the
        # resampler's filter delay.
        frames = await feed(self.serializer, [media("aa-a0", 1000, samples=320)] * 50)
        self.assertTrue(all(f.sample_rate == JITSI_PCM_SAMPLE_RATE for f in frames))
        self.assertGreaterEqual(len(samples_of(frames)), 45 * SAMPLES)
        self.assertLessEqual(len(samples_of(frames)), 50 * SAMPLES)


class TestJitsiMixing(unittest.IsolatedAsyncioTestCase):
    async def test_single_source_passes_through(self):
        serializer = await make_serializer()
        frames = await feed(serializer, [media("aa-a0", 1000)] * 10)
        self.assertEqual(len(frames), 10)
        self.assertTrue(all(isinstance(f, InputAudioRawFrame) for f in frames))
        self.assertTrue(np.all(samples_of(frames) == 1000))

    async def test_two_sources_mix_at_real_time(self):
        serializer = await make_serializer()
        interleaved = [m for _ in range(10) for m in (media("aa-a0", 1000), media("bb-a0", 2000))]
        mixed = samples_of(await feed(serializer, interleaved))
        # Two interleaved participants make one stream of the same duration, not twice as
        # long. The first frame goes out alone, before the second participant is known.
        self.assertEqual(len(mixed), 10 * SAMPLES)
        self.assertTrue(np.all(mixed[:SAMPLES] == 1000))
        self.assertTrue(np.all(mixed[SAMPLES:] == 3000))

    async def test_mix_clips_instead_of_wrapping(self):
        serializer = await make_serializer()
        mixed = samples_of(
            await feed(serializer, [media("aa-a0", 30000), media("bb-a0", 30000)] * 2)
        )
        self.assertTrue(np.all(mixed[SAMPLES:] == 32767))

    async def test_unmixed_mode_tags_each_frame_with_its_source(self):
        serializer = await make_serializer(mix_sources=False)
        frames = await feed(serializer, [media("aa-a0", 1000), media("bb-a0", 2000)])
        self.assertEqual([type(f) for f in frames], [UserAudioRawFrame, UserAudioRawFrame])
        self.assertEqual([f.user_id for f in frames], ["aa-a0", "bb-a0"])
        self.assertTrue(np.all(samples_of(frames[:1]) == 1000))
        self.assertTrue(np.all(samples_of(frames[1:]) == 2000))


class TestSourceMixer(unittest.TestCase):
    def test_stalled_source_does_not_hold_back_the_others(self):
        now = [0.0]
        mixer = _SourceMixer(
            JITSI_PCM_SAMPLE_RATE, active_window_s=0.2, max_lag_frames=3, clock=lambda: now[0]
        )
        # Both participants interleave: aa's first frame goes out alone, bb waits for aa's next.
        mixer.push("aa", pcm(100))
        self.assertEqual(mixer.push("bb", pcm(200)), b"")
        now[0] += 0.02
        self.assertTrue(np.all(np.frombuffer(mixer.push("aa", pcm(100)), dtype="<i2") == 300))

        # bb goes quiet: aa waits until it is three frames ahead, then flows one frame per
        # frame with bb as silence, a bounded 60 ms behind, so a late bb frame still lines up.
        held = b""
        for _ in range(2):
            now[0] += 0.02
            held += mixer.push("aa", pcm(100))
        self.assertEqual(held, b"")
        for _ in range(2):
            now[0] += 0.02
            flowing = np.frombuffer(mixer.push("aa", pcm(100)), dtype="<i2")
            self.assertEqual(len(flowing), SAMPLES)
            self.assertTrue(np.all(flowing == 100))

        # Past the active window bb drops out: aa's backlog flushes and aa passes straight through.
        now[0] += 0.3
        flushed = np.frombuffer(mixer.push("aa", pcm(100)), dtype="<i2")
        self.assertEqual(len(flushed), 3 * SAMPLES)
        self.assertTrue(np.all(flushed == 100))
        self.assertEqual(mixer.sources, ["aa"])


class TestJitsiSerialize(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.serializer = await make_serializer()

    async def test_audio_is_sent_as_media_without_a_tag(self):
        audio = pcm(1234)
        message = json.loads(
            await self.serializer.serialize(
                AudioRawFrame(audio=audio, sample_rate=JITSI_PCM_SAMPLE_RATE, num_channels=1)
            )
        )
        self.assertEqual(message["event"], "media")
        self.assertEqual(base64.b64decode(message["media"]["payload"]), audio)
        self.assertNotIn("tag", message["media"])

    async def test_audio_is_resampled_to_the_wire_rate(self):
        messages = [
            await self.serializer.serialize(
                AudioRawFrame(audio=pcm(1234, samples=960), sample_rate=48000, num_channels=1)
            )
            for _ in range(10)
        ]
        sent = sum(len(base64.b64decode(json.loads(m)["media"]["payload"])) for m in messages if m)
        # 10 frames of 20 ms at 48 kHz come out at 24 kHz, less the resampler's filter delay.
        self.assertGreaterEqual(sent, 5 * SAMPLES * 2)
        self.assertLessEqual(sent, 10 * SAMPLES * 2)

    async def test_interruption_clears_the_queued_audio(self):
        self.assertEqual(
            json.loads(await self.serializer.serialize(InterruptionFrame())), {"event": "clear"}
        )

    async def test_end_is_sent_once(self):
        self.assertEqual(json.loads(await self.serializer.serialize(EndFrame())), {"event": "end"})
        self.assertIsNone(await self.serializer.serialize(EndFrame()))
        self.assertIsNone(await self.serializer.serialize(CancelFrame()))

    async def test_cancel_also_ends(self):
        self.assertEqual(
            json.loads(await self.serializer.serialize(CancelFrame())), {"event": "end"}
        )

    async def test_auto_end_can_be_disabled(self):
        serializer = await make_serializer(auto_end=False)
        self.assertIsNone(await serializer.serialize(EndFrame()))
