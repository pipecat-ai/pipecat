#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Tavus transport."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from pipecat.frames.frames import (
    OutputAudioRawFrame,
    OutputTransportMessageUrgentFrame,
    TextType,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.tavus.transport import (
    TavusOutputTransport,
    TavusParams,
    TavusTransport,
    TavusTransportClient,
)
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup


def _make_output_transport(**params_kwargs) -> tuple[TavusOutputTransport, MagicMock]:
    client = MagicMock()
    client.queue_tts_frame = AsyncMock(return_value=True)
    client.send_realtime_audio_frame = AsyncMock(return_value=True)
    client.out_sample_rate = 24000
    params = TavusParams(**params_kwargs)
    return TavusOutputTransport(client, params), client


def test_audio_is_sent_faster_than_realtime_by_default():
    assert TavusParams().audio_out_faster_than_realtime is True


@pytest.mark.asyncio
async def test_default_params_send_audio_through_the_queue():
    """By default audio is queued for the send task rather than paced to playback time."""
    transport, client = _make_output_transport()

    frame = OutputAudioRawFrame(audio=b"\x00" * 960, sample_rate=24000, num_channels=1)
    assert await transport.write_audio_frame(frame) is True

    client.queue_tts_frame.assert_awaited_once_with(frame)
    client.send_realtime_audio_frame.assert_not_awaited()


@pytest.mark.asyncio
async def test_opting_out_paces_audio_to_playback_time():
    """With the flag off, each frame is sent immediately and the caller paces itself."""
    transport, client = _make_output_transport(audio_out_faster_than_realtime=False)

    frame = OutputAudioRawFrame(audio=b"\x00" * 960, sample_rate=24000, num_channels=1)
    assert await transport.write_audio_frame(frame) is True

    client.send_realtime_audio_frame.assert_awaited_once_with(frame)
    client.queue_tts_frame.assert_not_awaited()


@pytest.mark.asyncio
async def test_concurrent_setup_builds_a_single_daily_client(monkeypatch):
    """The input and output transports share one client, and both set it up.

    They are set up concurrently, so a client built per caller would leave the
    losing one orphaned, with its callback tasks running and nobody to clean
    them up.
    """
    import pipecat.transports.tavus.transport as tavus

    built = []

    def fake_daily_client(*args, **kwargs):
        client = MagicMock()
        client.setup = AsyncMock()
        built.append(client)
        return client

    monkeypatch.setattr(tavus, "DailyTransportClient", fake_daily_client)

    client = TavusTransportClient(
        bot_name="Pipecat",
        callbacks=MagicMock(),
        api_key="test-key",
        replica_id="replica",
        session=MagicMock(),
    )

    conversations = []

    async def fake_initialize():
        await asyncio.sleep(0.01)  # the real one calls the Tavus API
        conversations.append("conversation")
        client._conversation_id = f"conversation-{len(conversations)}"
        return "https://example.daily.co/room"

    monkeypatch.setattr(client, "_initialize", fake_initialize)

    setup = frame_processor_setup(TaskManager())
    await asyncio.gather(client.setup(setup), client.setup(setup))

    assert len(conversations) == 1, "a Tavus conversation was created per caller"
    assert len(built) == 1, f"{len(built)} Daily clients built, so one is orphaned"


@pytest.mark.asyncio
async def test_the_output_transport_joins_the_room():
    """Both transports join, so a pipeline using only the output one still joins."""
    transport, client = _make_output_transport()
    client.setup = AsyncMock()
    client.join = AsyncMock()

    await transport.setup(frame_processor_setup(TaskManager()))

    client.join.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_conversation_outlives_the_first_transport_to_stop(monkeypatch):
    """The input and output transports share one client, and both join its room.

    The input transport stops first, while the output still has audio to flush,
    so leaving the room and ending the conversation wait for the output too.
    """
    import pipecat.transports.tavus.transport as tavus

    daily = MagicMock()
    daily.setup = AsyncMock()
    daily.join = AsyncMock()
    daily.leave = AsyncMock()
    monkeypatch.setattr(tavus, "DailyTransportClient", lambda *args, **kwargs: daily)

    client = TavusTransportClient(
        bot_name="Pipecat",
        callbacks=MagicMock(),
        api_key="test-key",
        replica_id="replica",
        session=MagicMock(),
    )

    async def fake_initialize():
        client._conversation_id = "conversation-1"
        return "https://example.daily.co/room"

    monkeypatch.setattr(client, "_initialize", fake_initialize)
    client._api = MagicMock()
    client._api.end_conversation = AsyncMock()

    setup = frame_processor_setup(TaskManager())
    await asyncio.gather(client.setup(setup), client.setup(setup))
    await asyncio.gather(client.join(), client.join())

    await client.stop()
    daily.leave.assert_not_awaited()
    client._api.end_conversation.assert_not_awaited()

    await client.stop()
    daily.leave.assert_awaited_once()
    client._api.end_conversation.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_bot_name_reaches_the_daily_client(monkeypatch):
    """The name the caller picks is the bot's display name in the room."""
    import pipecat.transports.tavus.transport as tavus

    captured = {}

    def fake_daily_client(room_url, token, bot_name, params, callbacks, transport_name):
        captured["bot_name"] = bot_name
        daily = MagicMock()
        daily.setup = AsyncMock()
        return daily

    monkeypatch.setattr(tavus, "DailyTransportClient", fake_daily_client)

    transport = TavusTransport(
        bot_name="Ada",
        session=MagicMock(),
        api_key="test-key",
        replica_id="replica",
    )

    async def fake_initialize():
        return "https://example.daily.co/room"

    monkeypatch.setattr(transport._client, "_initialize", fake_initialize)

    await transport._client.setup(frame_processor_setup(TaskManager()))

    assert captured["bot_name"] == "Ada"


def _make_emotion_client() -> TavusTransportClient:
    client = TavusTransportClient(
        bot_name="Pipecat",
        callbacks=MagicMock(),
        api_key="test-key",
        replica_id="replica",
        session=MagicMock(),
    )
    client._conversation_id = "conversation"
    client._client = MagicMock()
    client._client.send_message = AsyncMock()
    return client


def _sent_emotions(client: TavusTransportClient) -> list[str]:
    return [
        call.args[0].message["properties"]["emotion"]
        for call in client._client.send_message.call_args_list
    ]


@pytest.mark.asyncio
async def test_a_delivery_sends_a_set_emotion_app_message():
    """The replica reads the emotion from a conversation.set_emotion app message."""
    client = _make_emotion_client()

    await client.update_emotion(["excited", "emphasis"])

    client._client.send_message.assert_awaited_once()
    frame = client._client.send_message.call_args.args[0]
    assert isinstance(frame, OutputTransportMessageUrgentFrame)
    assert frame.message == {
        "message_type": "conversation",
        "event_type": "conversation.set_emotion",
        "conversation_id": "conversation",
        "properties": {"emotion": "happiness"},
    }


@pytest.mark.asyncio
async def test_the_emotion_is_sent_only_when_it_changes():
    """Words in the same emotion, and names with no emotion, send nothing new."""
    client = _make_emotion_client()

    for delivery in (["sad"], ["sad"], ["apologetic"], ["slow"], ["surprised"], ["surprised"]):
        await client.update_emotion(delivery)

    assert _sent_emotions(client) == ["sadness", "surprise"]


@pytest.mark.asyncio
async def test_words_without_a_delivery_tag_return_the_replica_to_neutral():
    """Words the LLM wrote no tag for are in the voice's baseline, an empty delivery."""
    client = _make_emotion_client()

    await client.update_emotion(["excited"])
    await client.update_emotion([])

    assert _sent_emotions(client) == ["happiness", "neutral"]


@pytest.mark.asyncio
async def test_words_without_any_delivery_keep_the_emotion():
    """With expressive conversation off, words carry no delivery at all."""
    client = _make_emotion_client()

    await client.update_emotion(["excited"])
    await client.update_emotion(None)

    assert _sent_emotions(client) == ["happiness"]


@pytest.mark.asyncio
async def test_a_tts_text_frame_sets_the_emotion_when_the_clock_releases_it(monkeypatch):
    """The output transport pushes a timed frame on when its words are heard."""
    monkeypatch.setattr(BaseOutputTransport, "push_frame", AsyncMock())
    transport, client = _make_output_transport()
    client.update_emotion = AsyncMock()

    await transport.push_frame(
        TTSTextFrame("Great", text_type=TextType.WORD, delivery=["excited"]),
        FrameDirection.DOWNSTREAM,
    )

    client.update_emotion.assert_awaited_once_with(["excited"])
