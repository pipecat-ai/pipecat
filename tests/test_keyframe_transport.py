#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Keyframe transport."""

import asyncio
import base64
import json
import re
import struct
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

import pipecat.transports.keyframe.transport as keyframe
from pipecat.adapters.tts.base_expression_adapter import BaseExpressionAdapter
from pipecat.expressive.expression import (
    CORE_DELIVERY,
    EMPHASIS_DELIVERY,
    FAST_DELIVERY,
    SLOW_DELIVERY,
    ExpressionCapabilities,
    ExpressionDelivery,
    ExpressionTerm,
)
from pipecat.expressive.policy import ExpressionGuidance, ExpressionPolicy
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TextFrame,
    TextType,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transports.daily.transport import DailyOutputTransportMessageUrgentFrame
from pipecat.transports.keyframe.api import KeyframeApi
from pipecat.transports.keyframe.protocol import SAMPLE_RATE
from pipecat.transports.keyframe.transport import (
    KEYFRAME_DELIVERY_EMOTIONS,
    KeyframeEmotion,
    KeyframeNewSessionRequest,
    KeyframeOutputTransport,
    KeyframeParams,
    KeyframeSession,
    KeyframeTransport,
    KeyframeTransportClient,
    keyframe_policy_warning,
)
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup

# 40 ms of audio at the avatar's rate: one chunk, as the output transport writes it.
CHUNK = SAMPLE_RATE * 2 * 40 // 1000


class FakeClient:
    """A Keyframe transport client that records what the output transport sends the avatar.

    With ``finish_after``, it reports each turn as played that many seconds
    after the turn's last message, as the avatar does.
    """

    def __init__(self, finish_after: float | None = None):
        self.sent: list[dict[str, Any]] = []
        self.on_playback_finished = None
        self._finish_after = finish_after

    async def setup(self, setup):
        pass

    async def cleanup(self):
        pass

    async def join(self):
        pass

    async def stop(self):
        pass

    async def send_message(self, frame):
        pass

    async def send_to_avatar(self, message: dict[str, Any]):
        self.sent.append(message)
        if message.get("done") and self._finish_after is not None:
            asyncio.get_running_loop().call_later(
                self._finish_after,
                lambda: asyncio.ensure_future(self.on_playback_finished(message["turn"])),
            )

    async def send_interrupt_message(self):
        self.sent.append({"kfl": 2, "type": "interrupt"})

    def turns(self) -> list[str]:
        """The turns sent, in order."""
        return list(dict.fromkeys(m["turn"] for m in self.sent if m["type"] == "audio"))

    def records(self, turn: str) -> list[tuple[str, Any]]:
        """The records of a turn's messages, in order: ("audio", pcm) or ("cue", body)."""
        records = []
        for message in self.sent:
            if message["type"] != "audio" or message["turn"] != turn:
                continue
            data = base64.b64decode(message["data"])
            offset = 0
            while offset < len(data):
                kind, length = struct.unpack_from("<BI", data, offset)
                payload = data[offset + 5 : offset + 5 + length]
                offset += 5 + length
                records.append(("audio", payload) if kind == 1 else ("cue", json.loads(payload)))
        return records

    def audio(self, turn: str) -> bytes:
        return b"".join(payload for kind, payload in self.records(turn) if kind == "audio")

    def cues(self, turn: str) -> list[dict[str, Any]]:
        return [payload for kind, payload in self.records(turn) if kind == "cue"]


def _output(finish_after: float | None = 0.0) -> tuple[KeyframeOutputTransport, FakeClient]:
    client = FakeClient(finish_after)
    return KeyframeOutputTransport(client, KeyframeParams()), client  # pyright: ignore[reportArgumentType]


def _speech(seconds: float, *, context_id: str = "c1", rate: int = SAMPLE_RATE) -> TTSAudioRawFrame:
    """TTS audio whose samples count up, so a gap or a reordering shows."""
    samples = int(seconds * rate)
    audio = b"".join(struct.pack("<h", 1 + i % 1000) for i in range(samples))
    return TTSAudioRawFrame(audio, rate, 1, context_id=context_id)


def _word(text: str, delivery: list[str] | None, *, pts: int | None = None, context_id="c1"):
    frame = TTSTextFrame(text, text_type=TextType.WORD, context_id=context_id, delivery=delivery)
    frame.pts = pts
    return frame


def _types(frames, *types) -> list[type]:
    return [type(f) for f in frames if isinstance(f, types)]


#
# Audio sent to the avatar
#


@pytest.mark.asyncio
async def test_a_turn_reaches_the_avatar_as_numbered_messages_ended_by_done():
    transport, client = _output()
    speech = _speech(0.2)

    await run_test(
        transport,
        frames_to_send=[
            TTSStartedFrame(context_id="c1"),
            speech,
            TTSStoppedFrame(context_id="c1"),
        ],
    )

    messages = [m for m in client.sent if m["type"] == "audio"]
    assert [m["turn"] for m in messages] == ["t1"] * 6
    assert [m["seq"] for m in messages] == [0, 1, 2, 3, 4, 5]
    assert [m["done"] for m in messages] == [False] * 5 + [True]
    assert client.audio("t1") == speech.audio
    # One 40 ms chunk a message, which keeps each under Daily's app message limit.
    assert all(len(json.dumps(m)) < 4096 for m in messages)


@pytest.mark.asyncio
async def test_audio_at_another_rate_is_converted_to_the_avatars():
    transport, client = _output()

    await run_test(
        transport,
        frames_to_send=[
            TTSStartedFrame(context_id="c1"),
            _speech(0.5, rate=16000),
            TTSStoppedFrame(context_id="c1"),
        ],
    )

    samples = len(client.audio("t1")) // 2
    assert samples == pytest.approx(0.5 * SAMPLE_RATE, abs=CHUNK // 2)


def test_the_avatars_sample_rate_is_the_default():
    params = KeyframeParams()
    assert (params.audio_out_sample_rate, params.audio_out_channels) == (SAMPLE_RATE, 1)
    assert params.microphone_out_enabled is False


@pytest.mark.asyncio
async def test_another_output_sample_rate_is_refused():
    client = FakeClient()
    transport = KeyframeOutputTransport(client, KeyframeParams(audio_out_sample_rate=16000))  # pyright: ignore[reportArgumentType]

    with pytest.raises(ValueError, match="expects 24000 Hz mono audio"):
        await transport.setup(frame_processor_setup(TaskManager()))


#
# The bot speaks until the avatar has played its audio
#


@pytest.mark.asyncio
async def test_the_bot_stops_speaking_when_the_avatar_reports_it_played_the_turn():
    """The audio is sent well before it plays, so the stop waits for the avatar."""
    transport, client = _output(finish_after=0.3)

    down, up = await run_test(
        transport,
        frames_to_send=[
            TTSStartedFrame(context_id="c1"),
            _speech(0.2),
            TTSStoppedFrame(context_id="c1"),
            TextFrame("sent after the speech"),
        ],
    )

    assert _types(down, BotStartedSpeakingFrame, TextFrame, BotStoppedSpeakingFrame) == [
        BotStartedSpeakingFrame,
        TextFrame,
        BotStoppedSpeakingFrame,
    ]
    assert _types(up, BotStartedSpeakingFrame, BotStoppedSpeakingFrame) == [
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
    ]


@pytest.mark.asyncio
async def test_the_bot_stops_speaking_anyway_if_the_avatar_never_reports(monkeypatch):
    monkeypatch.setattr(keyframe, "PLAYBACK_GRACE_SECS", 0.1)
    transport, client = _output(finish_after=None)

    down, up = await run_test(
        transport,
        frames_to_send=[
            TTSStartedFrame(context_id="c1"),
            _speech(0.2),
            TTSStoppedFrame(context_id="c1"),
        ],
    )

    assert _types(down, BotStoppedSpeakingFrame) == [BotStoppedSpeakingFrame]
    assert _types(up, BotStoppedSpeakingFrame) == [BotStoppedSpeakingFrame]


class _StopTimes(FrameProcessor):
    """Records when each BotStoppedSpeakingFrame passes downstream."""

    def __init__(self):
        super().__init__()
        self.times: list[float] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStoppedSpeakingFrame) and direction == FrameDirection.DOWNSTREAM:
            self.times.append(time.monotonic())
        await self.push_frame(frame, direction)


@pytest.mark.asyncio
async def test_a_stop_the_avatar_never_reports_waits_for_the_turns_sent_before_it(monkeypatch):
    """The avatar plays a turn after the ones before it, however fast they were sent."""
    monkeypatch.setattr(keyframe, "PLAYBACK_GRACE_SECS", 0.3)
    transport, client = _output(finish_after=None)
    stops = _StopTimes()
    started = time.monotonic()

    await run_test(
        Pipeline([transport, stops]),
        frames_to_send=[
            TTSStartedFrame(context_id="c1"),
            _speech(1.5, context_id="c1"),
            TTSStoppedFrame(context_id="c1"),
            SleepFrame(0.05),
            TTSStartedFrame(context_id="c2"),
            _speech(0.2, context_id="c2"),
            TTSStoppedFrame(context_id="c2"),
        ],
    )

    assert client.turns() == ["t1", "t2"]
    # Not before both turns could have played: 1.5 s and 0.2 s.
    assert [stopped - started >= 1.7 for stopped in stops.times] == [True]


@pytest.mark.asyncio
async def test_speech_that_resumes_while_the_avatar_plays_never_stops():
    """A second turn sent before the first has played is one stretch of speech to the pipeline."""
    transport, client = _output(finish_after=0.3)

    down, up = await run_test(
        transport,
        frames_to_send=[
            TTSStartedFrame(context_id="c1"),
            _speech(0.2, context_id="c1"),
            TTSStoppedFrame(context_id="c1"),
            SleepFrame(0.1),
            TTSStartedFrame(context_id="c2"),
            _speech(0.2, context_id="c2"),
            TTSStoppedFrame(context_id="c2"),
        ],
    )

    assert client.turns() == ["t1", "t2"]
    for frames in (down, up):
        assert _types(frames, BotStartedSpeakingFrame, BotStoppedSpeakingFrame) == [
            BotStartedSpeakingFrame,
            BotStoppedSpeakingFrame,
        ]


#
# Interruptions
#


@pytest.mark.asyncio
async def test_an_interruption_stops_the_avatar_and_ends_the_turn_without_done():
    transport, client = _output()

    down, _ = await run_test(
        transport,
        frames_to_send=[
            TTSStartedFrame(context_id="c1"),
            _speech(1.0, context_id="c1"),
            SleepFrame(0.1),
            InterruptionFrame(),
            SleepFrame(0.1),
            TTSStartedFrame(context_id="c2"),
            _speech(0.2, context_id="c2"),
            TTSStoppedFrame(context_id="c2"),
        ],
    )

    kinds = [(m["type"], m.get("turn"), m.get("done")) for m in client.sent]
    interrupt = kinds.index(("interrupt", None, None))
    assert ("audio", "t1", True) not in kinds
    assert all(turn == "t1" for _, turn, _ in kinds[:interrupt])
    assert all(turn == "t2" for _, turn, _ in kinds[interrupt + 1 :])
    assert kinds[-1] == ("audio", "t2", True)
    # The interrupted turn stops at once; the next one starts and stops on its own.
    assert _types(down, BotStartedSpeakingFrame, BotStoppedSpeakingFrame) == [
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
    ]


@pytest.mark.asyncio
async def test_an_interruption_while_the_avatar_plays_stops_the_bot_speaking():
    """The turn was sent in full and its stop is held; the interruption releases it."""
    transport, client = _output(finish_after=None)

    down, up = await run_test(
        transport,
        frames_to_send=[
            TTSStartedFrame(context_id="c1"),
            _speech(0.2),
            TTSStoppedFrame(context_id="c1"),
            SleepFrame(0.1),
            InterruptionFrame(),
            TextFrame("after the interruption"),
        ],
    )

    assert [m["type"] for m in client.sent][-1] == "interrupt"
    assert _types(down, BotStoppedSpeakingFrame, TextFrame) == [BotStoppedSpeakingFrame, TextFrame]
    assert _types(up, BotStoppedSpeakingFrame) == [BotStoppedSpeakingFrame]


#
# Emotion
#


def test_every_core_delivery_but_pace_has_an_emotion():
    """The LLM is offered the core deliveries by default, and the tagger writes only them."""
    assert set(CORE_DELIVERY) - {SLOW_DELIVERY, FAST_DELIVERY} <= set(KEYFRAME_DELIVERY_EMOTIONS)


def test_every_keyframe_emotion_converts_from_its_own_name():
    for emotion in KeyframeEmotion:
        assert keyframe._delivery_emotion([emotion.value]) is emotion


@pytest.mark.parametrize(
    "delivery, emotion",
    [
        ([ExpressionDelivery.WARM], KeyframeEmotion.HAPPY),
        ([ExpressionDelivery.SYMPATHETIC, EMPHASIS_DELIVERY], KeyframeEmotion.CONCERNED),
        ([ExpressionDelivery.SLOW, ExpressionDelivery.EXCITED], KeyframeEmotion.EXCITED),
        ([ExpressionDelivery.NOSTALGIC], KeyframeEmotion.NEUTRAL),  # no emotion for it
        ([EMPHASIS_DELIVERY], KeyframeEmotion.NEUTRAL),  # how baseline words are spoken
        ([], KeyframeEmotion.NEUTRAL),  # the voice's baseline
        (None, None),  # expressive conversation off: keep the current emotion
    ],
)
def test_a_delivery_converts_to_the_emotion_of_its_mood(delivery, emotion):
    assert keyframe._delivery_emotion(delivery) is emotion


def test_the_default_deliveries_all_have_an_emotion():
    assert keyframe_policy_warning(None) is None
    assert (
        keyframe_policy_warning(ExpressionPolicy(guidance=[ExpressionGuidance.MATCH_STYLE])) is None
    )


def test_a_policy_delivery_with_no_emotion_is_warned_about():
    warning = keyframe_policy_warning(
        ExpressionPolicy(delivery=["calm", "slow", "nostalgic", "skeptical", "proud"])
    )

    assert warning is not None
    assert "no emotion for nostalgic, proud from the expression policy" in warning


@pytest.mark.asyncio
async def test_a_change_of_delivery_is_cued_on_the_sample_its_word_starts_at():
    """Word timestamps count from the start of their context's audio.

    The first word is timed a second into the clock, later than the context's
    audio arrives, so that is where the context's words count from: the second
    word, 250 ms later, starts 6000 samples into the turn.
    """
    transport, client = _output()
    second = 1_000_000_000

    await run_test(
        transport,
        frames_to_send=[
            _word("Before", None, pts=second, context_id="c0"),
            TTSStartedFrame(context_id="c1"),
            _speech(0.4),
            _word("Great", [ExpressionDelivery.EXCITED], pts=second + second // 4),
            _word("news", [ExpressionDelivery.EXCITED], pts=second + second // 2),
            _speech(0.2),
            TTSStoppedFrame(context_id="c1"),
        ],
    )

    assert client.cues("t1") == [{"emotion": "excited", "at": 6000}]
    # The cue was read when its word arrived, with the audio after it still to send.
    kinds = [kind for kind, _ in client.records("t1")]
    assert kinds.index("cue") == 10 and kinds.count("audio") == 15


@pytest.mark.asyncio
async def test_a_second_contexts_words_count_from_where_its_audio_starts_in_the_turn():
    transport, client = _output()
    second = 1_000_000_000

    await run_test(
        transport,
        frames_to_send=[
            _word("Before", None, pts=second, context_id="c0"),
            TTSStartedFrame(context_id="c1"),
            _speech(0.4, context_id="c1"),
            # The next context starts where the last word left the clock.
            _word("last", None, pts=2 * second, context_id="c1"),
            _speech(0.4, context_id="c2"),
            _word(
                "Oh", [ExpressionDelivery.SURPRISED], pts=2 * second + second // 10, context_id="c2"
            ),
            TTSStoppedFrame(context_id="c2"),
        ],
    )

    # 0.4 s of the first context, then 100 ms into the second.
    assert client.cues("t1") == [{"emotion": "surprised", "at": 9600 + 2400}]


@pytest.mark.asyncio
async def test_a_word_without_a_timestamp_is_cued_at_the_audio_sent_so_far():
    transport, client = _output()

    await run_test(
        transport,
        frames_to_send=[
            TTSStartedFrame(context_id="c1"),
            _speech(0.4),
            _word("I'm sorry to hear that.", [ExpressionDelivery.SYMPATHETIC]),
            _speech(0.2),
            TTSStoppedFrame(context_id="c1"),
        ],
    )

    assert client.cues("t1") == [{"emotion": "concerned", "at": 9600}]


@pytest.mark.asyncio
async def test_an_emotion_is_cued_only_when_it_changes_and_again_after_an_interruption():
    transport, client = _output()

    await run_test(
        transport,
        frames_to_send=[
            TTSStartedFrame(context_id="c1"),
            _speech(0.2, context_id="c1"),
            _word("so", [ExpressionDelivery.WARM], context_id="c1"),
            _word("glad", [ExpressionDelivery.HAPPY], context_id="c1"),  # happy already
            _word("really", [ExpressionDelivery.HAPPY, EMPHASIS_DELIVERY], context_id="c1"),
            _word("plain", [], context_id="c1"),
            SleepFrame(0.1),
            InterruptionFrame(),
            SleepFrame(0.1),
            TTSStartedFrame(context_id="c2"),
            _word("plain", [], context_id="c2"),
            _speech(0.2, context_id="c2"),
            TTSStoppedFrame(context_id="c2"),
        ],
    )

    assert [cue["emotion"] for cue in client.cues("t1")] == ["happy", "neutral"]
    # The interruption may have dropped a cue the avatar had not reached, so the
    # emotion is sent again, with the next turn's first audio.
    assert client.cues("t2") == [{"emotion": "neutral", "at": 0}]
    assert client.records("t2")[0][0] == "cue"


@pytest.mark.asyncio
async def test_a_cue_reached_while_an_interruption_is_handled_is_not_kept_for_the_next_turn():
    """It is for audio the interruption discards."""
    transport, client = _output()
    transport._interrupting = True
    await transport.write_transport_frame(keyframe._EmotionCueFrame(emotion="sad"))
    transport._interrupting = False

    await run_test(
        transport,
        frames_to_send=[
            TTSStartedFrame(context_id="c1"),
            _speech(0.2),
            TTSStoppedFrame(context_id="c1"),
        ],
    )

    assert client.cues("t1") == []


@pytest.mark.asyncio
async def test_a_cue_is_sent_when_its_word_arrives_not_with_the_audio_after_it():
    """The avatar has the audio ahead of its playback, so a prompt cue lands on its word."""
    transport, client = _output(finish_after=None)

    await run_test(
        transport,
        frames_to_send=[
            TTSStartedFrame(context_id="c1"),
            _speech(0.2),
            _word("Oh no.", [ExpressionDelivery.WORRIED]),
            SleepFrame(0.1),
            InterruptionFrame(),
        ],
    )

    assert client.cues("t1") == [{"emotion": "concerned", "at": 4800}]


class _TagAdapter(BaseExpressionAdapter):
    """Formats a change of delivery as <delivery> before the words it starts at."""

    def capabilities(self):
        return ExpressionCapabilities(
            delivery=[ExpressionTerm("excited"), ExpressionTerm("sad"), ExpressionTerm("calm")]
        )

    def format_text(self, text, delivery, previous_delivery):
        changed = delivery != (previous_delivery or [])
        return f"<{','.join(delivery)}> {text}" if changed and delivery else text


class _FourWordsASecondTTSService(TTSService):
    """Speaks four words a second, reporting each word's start after sending its audio.

    Each word's audio is sent before its timestamp, as a service that streams
    alignment with its audio does.
    """

    expression_adapter_class = _TagAdapter

    def __init__(self, **kwargs):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            push_text_frames=False,
            sample_rate=SAMPLE_RATE,
            **kwargs,
        )
        self._spoken_secs: dict[str, float] = {}

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        # Sentences of one response share a context, and its words are timed
        # from the start of its audio.
        for word in re.sub(r"<[^>]*>", " ", text).split():
            start = self._spoken_secs.get(context_id, 0.0)
            self._spoken_secs[context_id] = start + 0.25
            yield _speech(0.25, context_id=context_id)
            await self.add_word_timestamps([(word, start)], context_id=context_id)


@pytest.mark.asyncio
async def test_the_emotion_changes_on_the_word_a_tts_service_reports_the_new_delivery_at():
    """The LLM's tags, read by a TTS service that times its words, reach the avatar as cues.

    Two sentences, each a request of its own: the emotion changes at the first
    word, at the third word of the first sentence, and at the second sentence.
    """
    transport, client = _output()
    word = SAMPLE_RATE // 4

    await run_test(
        Pipeline([_FourWordsASecondTTSService(), transport]),
        frames_to_send=[
            LLMFullResponseStartFrame(),
            LLMTextFrame("Well that <<excited>> worked really nicely. <<sad>> Then it broke."),
            LLMFullResponseEndFrame(),
        ],
        pipeline_params=PipelineParams(enable_expressive_conversation=True),
    )

    assert client.turns() == ["t1"]
    assert len(client.audio("t1")) == 8 * word * 2
    cues = client.cues("t1")
    assert [cue["emotion"] for cue in cues] == ["neutral", "excited", "sad"]
    # Within a video frame (40 ms) of the words "Well", "worked" and "Then".
    assert [cue["at"] for cue in cues] == pytest.approx([0, 2 * word, 5 * word], abs=960)


@pytest.mark.asyncio
async def test_the_emotion_still_changes_on_its_word_after_an_interruption():
    """The words after an interruption are timed from now, not after the words it cut off.

    The interrupted response's twelve words were timed three seconds ahead,
    since their audio was sent faster than it plays.
    """
    transport, client = _output()
    word = SAMPLE_RATE // 4

    await run_test(
        Pipeline([_FourWordsASecondTTSService(), transport]),
        frames_to_send=[
            LLMFullResponseStartFrame(),
            LLMTextFrame("One two three four five six seven eight nine ten eleven twelve."),
            LLMFullResponseEndFrame(),
            SleepFrame(0.3),
            InterruptionFrame(),
            SleepFrame(0.2),
            LLMFullResponseStartFrame(),
            LLMTextFrame("Now we <<sad>> start again today."),
            LLMFullResponseEndFrame(),
        ],
        pipeline_params=PipelineParams(enable_expressive_conversation=True),
    )

    assert client.turns() == ["t1", "t2"]
    cues = client.cues("t2")
    assert [cue["emotion"] for cue in cues] == ["neutral", "sad"]
    assert [cue["at"] for cue in cues] == pytest.approx([0, 2 * word], abs=960)


#
# The client: the avatar's app messages
#


def _client() -> KeyframeTransportClient:
    client = KeyframeTransportClient(
        bot_name="Pipecat",
        callbacks=MagicMock(),
        api_key="test-key",
        session_request=KeyframeNewSessionRequest(persona_slug="public:cosmo_persona-2.0-live"),
        session=MagicMock(),
    )
    client._session = KeyframeSession(
        reservation_id="r1",
        room_url="https://example.daily.co/room",
        avatar_participant_name="keyframe-avatar",
    )
    client._daily_transport_client = MagicMock()
    client._daily_transport_client.send_message = AsyncMock()
    return client


def _sent(client: KeyframeTransportClient) -> list[tuple[dict, str | None]]:
    frames = [call.args[0] for call in client._daily_transport_client.send_message.call_args_list]
    assert all(isinstance(f, DailyOutputTransportMessageUrgentFrame) for f in frames)
    return [(f.message, f.participant_id) for f in frames]


HELLO = {"kfl": 2, "type": "hello", "protocol": "2"}


@pytest.mark.asyncio
async def test_messages_are_held_until_the_avatar_says_hello_then_sent_to_it_alone():
    client = _client()

    await client.send_to_avatar({"n": 1})
    await client.avatar_joined("avatar-id")
    await client.send_to_avatar({"n": 2})
    assert _sent(client) == []

    await client._on_app_message(HELLO, "avatar-id")
    await client.send_to_avatar({"n": 3})

    assert _sent(client) == [({"n": n}, "avatar-id") for n in (1, 2, 3)]


@pytest.mark.asyncio
async def test_a_message_sent_while_held_ones_go_out_follows_them():
    """Sending takes time, and the bot goes on speaking while the held messages are sent."""
    client = _client()

    async def slow_send(frame):
        await asyncio.sleep(0.01)

    client._daily_transport_client.send_message = AsyncMock(side_effect=slow_send)
    await client.avatar_joined("avatar-id")
    for n in (1, 2, 3):
        await client.send_to_avatar({"n": n})

    hello = asyncio.create_task(client._on_app_message(HELLO, "avatar-id"))
    await asyncio.sleep(0.015)  # the second held message is on its way
    await client.send_to_avatar({"n": 4})
    await hello
    await client.send_to_avatar({"n": 5})

    assert _sent(client) == [({"n": n}, "avatar-id") for n in (1, 2, 3, 4, 5)]


@pytest.mark.asyncio
async def test_held_messages_are_not_sent_to_the_room_if_the_avatar_leaves_meanwhile():
    client = _client()

    async def leave_after_first(frame):
        client.avatar_left()

    client._daily_transport_client.send_message = AsyncMock(side_effect=leave_after_first)
    await client.avatar_joined("avatar-id")
    for n in (1, 2):
        await client.send_to_avatar({"n": n})

    await client._on_app_message(HELLO, "avatar-id")

    assert _sent(client) == [({"n": 1}, "avatar-id")]


@pytest.mark.asyncio
async def test_a_hello_that_arrives_before_the_avatars_participant_is_known_counts():
    client = _client()
    await client.send_to_avatar({"n": 1})

    await client._on_app_message(HELLO, "avatar-id")
    assert _sent(client) == []
    await client.avatar_joined("avatar-id")

    assert _sent(client) == [({"n": 1}, "avatar-id")]


@pytest.mark.asyncio
async def test_another_participants_hello_does_not_release_the_avatars_messages():
    client = _client()
    await client.avatar_joined("avatar-id")
    await client.send_to_avatar({"n": 1})

    await client._on_app_message(HELLO, "someone-else")

    assert _sent(client) == []


@pytest.mark.asyncio
async def test_an_interrupt_discards_held_messages_and_is_sent_only_to_a_ready_avatar():
    client = _client()
    await client.avatar_joined("avatar-id")
    await client.send_to_avatar({"n": 1})

    await client.send_interrupt_message()
    await client._on_app_message(HELLO, "avatar-id")
    assert _sent(client) == []

    await client.send_interrupt_message()
    assert _sent(client) == [({"kfl": 2, "type": "interrupt"}, "avatar-id")]


@pytest.mark.asyncio
async def test_only_the_avatars_report_of_a_turn_played_to_its_end_is_passed_on():
    client = _client()
    client.on_playback_finished = AsyncMock()
    await client.avatar_joined("avatar-id")
    finished = {"kfl": 2, "type": "playback_finished", "turn": "t1", "interrupted": False}

    await client._on_app_message(finished, "someone-else")
    await client._on_app_message({**finished, "interrupted": True}, "avatar-id")
    await client._on_app_message({**finished, "turn": None}, "avatar-id")
    await client._on_app_message({**finished, "kfl": 1}, "avatar-id")
    client.on_playback_finished.assert_not_awaited()

    await client._on_app_message(finished, "avatar-id")
    client.on_playback_finished.assert_awaited_once_with("t1")


@pytest.mark.asyncio
async def test_the_oldest_held_messages_are_dropped_past_the_limit(monkeypatch):
    monkeypatch.setattr(keyframe, "MAX_PENDING_MESSAGES", 2)
    client = _client()
    await client.avatar_joined("avatar-id")

    for n in (1, 2, 3):
        await client.send_to_avatar({"n": n})
    await client._on_app_message(HELLO, "avatar-id")

    assert _sent(client) == [({"n": 2}, "avatar-id"), ({"n": 3}, "avatar-id")]


#
# The session and the room
#


class _FakeHttp:
    """An aiohttp session that answers every POST with ``response``."""

    def __init__(self, response: dict[str, Any]):
        self.response = response
        self.posts: list[tuple[str, dict, dict]] = []

    @asynccontextmanager
    async def post(self, url, headers=None, json=None):
        self.posts.append((url, headers, json))
        reply = MagicMock()
        reply.json = AsyncMock(return_value=self.response)
        yield reply


@pytest.mark.asyncio
async def test_a_session_in_the_callers_room_sends_the_room_and_the_avatars_token():
    http = _FakeHttp({"reservation_id": "r1"})

    await KeyframeApi("test-key", http).create_session(  # pyright: ignore[reportArgumentType]
        source_participant_name="Pipecat",
        persona_slug="public:cosmo_persona-2.0-live",
        daily_room_url="https://example.daily.co/room",
        avatar_daily_token="avatar-token",
    )

    assert http.posts == [
        (
            "https://api.keyframelabs.com/v1/sessions/plugins/daily",
            {"Authorization": "Bearer test-key", "Content-Type": "application/json"},
            {
                "source_participant_name": "Pipecat",
                "persona_slug": "public:cosmo_persona-2.0-live",
                "room_url": "https://example.daily.co/room",
                "avatar_token": "avatar-token",
            },
        )
    ]


@pytest.mark.asyncio
async def test_the_bot_joins_a_room_keyframe_created_with_the_token_it_was_given():
    http = _FakeHttp(
        {
            "reservation_id": "r1",
            "model_id": "persona-2.0-live",
            "protocol": "2",
            "avatar_participant_name": "keyframe-avatar",
            "avatar_participant_id": "avatar-user",
            "room_url": "https://keyframe.daily.co/made",
            "bot_token": "bot-token",
            "user_token": "user-token",
        }
    )
    client = KeyframeTransportClient(
        bot_name="Pipecat",
        callbacks=MagicMock(),
        api_key="test-key",
        session_request=KeyframeNewSessionRequest(persona_id="p1", api_url="http://localhost:1"),
        session=http,  # pyright: ignore[reportArgumentType]
    )

    session = await client._initialize()

    assert http.posts[0][0] == "http://localhost:1/v1/sessions/plugins/daily"
    assert http.posts[0][2] == {"source_participant_name": "Pipecat", "persona_id": "p1"}
    assert session == KeyframeSession(
        reservation_id="r1",
        model="persona-2.0-live",
        room_url="https://keyframe.daily.co/made",
        avatar_participant_name="keyframe-avatar",
        avatar_participant_id="avatar-user",
        bot_token="bot-token",
        user_token="user-token",
    )


@pytest.mark.asyncio
async def test_the_bot_joins_the_callers_room_with_the_callers_token():
    http = _FakeHttp(
        {
            "reservation_id": "r1",
            "avatar_participant_name": "keyframe-avatar",
            "room_url": "https://example.daily.co/room",
        }
    )
    client = KeyframeTransportClient(
        bot_name="Pipecat",
        callbacks=MagicMock(),
        api_key="test-key",
        session_request=KeyframeNewSessionRequest(
            persona_id="p1",
            daily_room_url="https://example.daily.co/room",
            daily_token="mine",
            daily_user_id="bot-7",
        ),
        session=http,  # pyright: ignore[reportArgumentType]
    )

    session = await client._initialize()

    assert (session.bot_token, session.user_token) == ("mine", None)
    # The avatar is told which user ID the bot's token carries, to know it by.
    assert http.posts[0][2]["source_participant_id"] == "bot-7"
    assert session.avatar_participant_id is None


@pytest.mark.parametrize(
    "request_fields, error",
    [
        ({}, "exactly one of persona_id or persona_slug"),
        (
            {"persona_id": "p", "persona_slug": "public:cosmo_persona-2.0-live"},
            "exactly one of persona_id",
        ),
        ({"persona_id": "p", "daily_token": "t"}, "require daily_room_url"),
        ({"persona_id": "p", "avatar_daily_token": "t"}, "require daily_room_url"),
        ({"persona_id": "p", "daily_user_id": "u"}, "require daily_room_url"),
        (
            {
                "persona_id": "p",
                "daily_room_url": "https://example.daily.co/r",
                "daily_user_id": "u",
            },
            "provide both",
        ),
    ],
)
def test_a_session_request_names_one_persona_and_tokens_only_with_a_room(request_fields, error):
    with pytest.raises(ValidationError, match=error):
        KeyframeNewSessionRequest(**request_fields)


def _transport() -> KeyframeTransport:
    transport = KeyframeTransport(
        bot_name="Pipecat",
        session=MagicMock(),
        api_key="test-key",
        session_request=KeyframeNewSessionRequest(persona_slug="public:cosmo_persona-2.0-live"),
    )
    transport._client._session = KeyframeSession(
        reservation_id="r1",
        room_url="https://example.daily.co/room",
        avatar_participant_name="keyframe-avatar",
    )
    daily = MagicMock()
    daily.send_message = AsyncMock()
    daily.update_subscriptions = AsyncMock()
    transport._client._daily_transport_client = daily
    transport._call_event_handler = AsyncMock()
    return transport


@pytest.mark.asyncio
async def test_the_avatar_is_the_participant_with_its_user_name_and_its_microphone_is_ignored():
    transport = _transport()
    avatar = {"id": "avatar-id", "info": {"userName": "keyframe-avatar"}}

    await transport._on_participant_joined(avatar)

    transport._client._daily_transport_client.update_subscriptions.assert_awaited_once_with(
        participant_settings={"avatar-id": {"media": {"microphone": "unsubscribed"}}},
        profile_settings=None,
    )
    transport._call_event_handler.assert_awaited_once_with("on_avatar_connected", avatar)

    await transport._on_participant_left(avatar, "leftCall")
    transport._call_event_handler.assert_awaited_with("on_avatar_disconnected", avatar, "leftCall")


@pytest.mark.asyncio
async def test_in_a_room_keyframe_created_the_avatar_is_known_by_its_user_id_not_its_name():
    """A participant can change its user name in the room, but not its token's user ID."""
    transport = _transport()
    transport._client._session = KeyframeSession(
        reservation_id="r1",
        room_url="https://keyframe.daily.co/made",
        avatar_participant_name="keyframe-avatar",
        avatar_participant_id="avatar-user",
    )
    transport._input = MagicMock()
    transport._input.start_capturing_audio = AsyncMock()
    impostor = {"id": "user-id", "info": {"userName": "keyframe-avatar", "userId": "someone"}}
    avatar = {"id": "avatar-id", "info": {"userName": "Renamed", "userId": "avatar-user"}}

    await transport._on_participant_joined(impostor)
    await transport._on_participant_joined(avatar)

    assert [call.args for call in transport._call_event_handler.await_args_list] == [
        ("on_client_connected", impostor),
        ("on_avatar_connected", avatar),
    ]
    assert transport._client._avatar_id == "avatar-id"


@pytest.mark.asyncio
async def test_any_other_participant_is_a_client_whose_audio_is_captured():
    transport = _transport()
    transport._input = MagicMock()
    transport._input.start_capturing_audio = AsyncMock()
    user = {"id": "user-id", "info": {"userName": "Ada"}}

    await transport._on_participant_joined(user)

    transport._call_event_handler.assert_awaited_once_with("on_client_connected", user)
    transport._input.start_capturing_audio.assert_awaited_once_with(user)
    transport._client._daily_transport_client.update_subscriptions.assert_not_awaited()


@pytest.mark.asyncio
async def test_set_emotion_changes_the_avatars_emotion_right_away():
    transport = _transport()
    await transport._client.avatar_joined("avatar-id")
    await transport._client._on_app_message(HELLO, "avatar-id")

    await transport.set_emotion("skeptical")

    assert _sent(transport._client) == [
        ({"kfl": 2, "type": "emotion_now", "emotion": "skeptical"}, "avatar-id")
    ]
    with pytest.raises(ValueError):
        await transport.set_emotion("smug")


def test_frames_the_transport_queues_for_itself_are_not_pushed_on():
    """They order emotion cues with the audio, and mean nothing past the transport."""
    assert issubclass(keyframe._ContextStartFrame, Frame)
    assert issubclass(keyframe._EmotionCueFrame, Frame)
