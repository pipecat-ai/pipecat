#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("riva.client")

from pipecat.frames.frames import InterimTranscriptionFrame, TranscriptionFrame
from pipecat.services.nvidia.stt import AudioChunkIterator, NvidiaSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.string import TextPartForConcatenation, concatenate_aggregated_text


def _make_service(**kwargs) -> NvidiaSTTService:
    return NvidiaSTTService(api_key="test-key", **kwargs)


def _word(text: str, speaker_tag: int | None = None) -> SimpleNamespace:
    word = SimpleNamespace(word=text)
    if speaker_tag is not None:
        word.speaker_tag = speaker_tag
    return word


def _streaming_response(
    transcript: str,
    *,
    words: list[SimpleNamespace] | None = None,
    is_final: bool = True,
) -> SimpleNamespace:
    alternative = SimpleNamespace(transcript=transcript)
    if words is not None:
        alternative.words = words
    result = SimpleNamespace(alternatives=[alternative], is_final=is_final)
    return SimpleNamespace(results=[result])


async def _capture_response_frames(
    monkeypatch,
    service: NvidiaSTTService,
    response: SimpleNamespace,
    *,
    speaker_diarization: bool | None = None,
) -> list:
    frames = []
    monkeypatch.setattr(service, "push_frame", AsyncMock(side_effect=frames.append))
    monkeypatch.setattr(service, "emit_stt_usage_metrics", AsyncMock())
    monkeypatch.setattr(service, "_handle_transcription", AsyncMock())

    if speaker_diarization is None:
        speaker_diarization = bool(service._settings.speaker_diarization)
    await service._handle_response(response, speaker_diarization=speaker_diarization)

    return frames


def _aggregate(frames) -> str:
    return concatenate_aggregated_text(
        [
            TextPartForConcatenation(
                frame.text,
                includes_inter_part_spaces=frame.includes_inter_frame_spaces,
            )
            for frame in frames
        ]
    )


@pytest.mark.asyncio
async def test_keepalive_enabled():
    """NVIDIA STT enables silence keepalive (the base default is off)."""
    service = _make_service()
    assert service._keepalive_timeout == 30.0
    assert service._keepalive_interval == 5.0


@pytest.mark.asyncio
async def test_keepalive_not_ready_without_iterator():
    """No active stream means keepalive should not fire."""
    service = _make_service()
    assert service._audio_iterator is None
    assert service._is_keepalive_ready() is False


@pytest.mark.asyncio
async def test_keepalive_ready_with_open_iterator():
    """An open iterator is a valid keepalive target."""
    service = _make_service()
    service._audio_iterator = AudioChunkIterator(asyncio.get_running_loop())
    assert service._is_keepalive_ready() is True


@pytest.mark.asyncio
async def test_keepalive_not_ready_with_closed_iterator():
    """A closed iterator must not be fed silence."""
    service = _make_service()
    iterator = AudioChunkIterator(asyncio.get_running_loop())
    await iterator.close()
    service._audio_iterator = iterator
    assert service._is_keepalive_ready() is False


@pytest.mark.asyncio
async def test_send_keepalive_enqueues_silence():
    """Silence is pushed into the active stream iterator."""
    service = _make_service()
    iterator = AudioChunkIterator(asyncio.get_running_loop())
    service._audio_iterator = iterator

    silence = b"\x00\x00\x00\x00"
    await service._send_keepalive(silence)

    assert iterator._queue.get_nowait() == silence


@pytest.mark.asyncio
async def test_send_keepalive_noop_when_closed():
    """Sending keepalive to a closed iterator is a no-op."""
    service = _make_service()
    iterator = AudioChunkIterator(asyncio.get_running_loop())
    await iterator.close()
    # close() enqueues a sentinel; drain it so the queue reflects keepalive only.
    iterator._queue.get_nowait()
    service._audio_iterator = iterator

    await service._send_keepalive(b"\x00\x00")

    assert iterator._queue.empty()


@pytest.mark.asyncio
async def test_send_keepalive_noop_without_iterator():
    """Sending keepalive with no active stream does not raise."""
    service = _make_service()
    await service._send_keepalive(b"\x00\x00")


@pytest.mark.asyncio
async def test_update_settings_reconnects_so_the_stream_uses_them(monkeypatch):
    """A settings change must reach the gRPC stream, not just the local config.

    streaming_response_generator() is handed streaming_config once, when the
    stream is opened, so rebuilding the config without reconnecting leaves the
    live stream transcribing with the previous settings and nothing logs.
    """
    service = _make_service()
    service._config = service._create_recognition_config()
    reconnect = AsyncMock()
    monkeypatch.setattr(service, "_request_reconnect", reconnect)

    changed = await service._update_settings(NvidiaSTTService.Settings(language=Language.ES))

    assert changed
    assert service._settings.language == Language.ES
    reconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_settings_rebuilds_the_recognition_config(monkeypatch):
    """The rebuilt config carries the new language into the next stream."""
    service = _make_service()
    service._config = service._create_recognition_config()
    monkeypatch.setattr(service, "_request_reconnect", AsyncMock())

    assert service._config.config.language_code == Language.EN_US

    await service._update_settings(NvidiaSTTService.Settings(language=Language.ES))

    assert service._config.config.language_code == Language.ES


@pytest.mark.asyncio
async def test_update_settings_without_changes_does_not_reconnect(monkeypatch):
    """A no-op delta must not tear down a healthy stream."""
    service = _make_service()
    service._config = service._create_recognition_config()
    reconnect = AsyncMock()
    monkeypatch.setattr(service, "_request_reconnect", reconnect)

    changed = await service._update_settings(NvidiaSTTService.Settings())

    assert not changed
    reconnect.assert_not_awaited()


@pytest.mark.asyncio
async def test_speaker_format_update_does_not_reconnect(monkeypatch):
    """Local speaker text formatting does not restart the NVIDIA stream."""
    service = _make_service()
    reconnect = AsyncMock()
    monkeypatch.setattr(service, "_request_reconnect", reconnect)

    changed = await service._update_settings(
        NvidiaSTTService.Settings(speaker_format="Speaker {speaker}: {text}")
    )

    assert changed
    reconnect.assert_not_awaited()


def test_diarization_does_not_change_word_time_offsets():
    """Diarization leaves the word time offset setting as configured."""
    service = _make_service(
        settings=NvidiaSTTService.Settings(
            speaker_diarization=True,
            word_time_offsets=False,
        )
    )

    config = service._create_recognition_config()

    assert config.config.enable_word_time_offsets is False
    assert config.config.diarization_config.enable_speaker_diarization


@pytest.mark.asyncio
async def test_diarization_off_preserves_one_frame_and_session_user(monkeypatch):
    """Speaker metadata has no effect unless diarization is enabled."""
    service = _make_service()
    service._user_id = "session-user"
    response = _streaming_response(
        "Hi there",
        words=[_word("Hi", 0), _word("there", 1)],
    )

    frames = await _capture_response_frames(monkeypatch, service, response)

    assert len(frames) == 1
    assert frames[0].text == "Hi there"
    assert frames[0].user_id == "session-user"


@pytest.mark.asyncio
async def test_zero_speaker_tag_is_a_valid_speaker(monkeypatch):
    """NVIDIA speaker tag zero maps to user ID zero instead of the session user."""
    service = _make_service(settings=NvidiaSTTService.Settings(speaker_diarization=True))
    response = _streaming_response(
        " Hello there",
        words=[_word("Hello", 0), _word("there", 0)],
    )

    frames = await _capture_response_frames(monkeypatch, service, response)

    assert len(frames) == 1
    assert isinstance(frames[0], TranscriptionFrame)
    assert frames[0].text == " Hello there"
    assert frames[0].user_id == "0"
    assert frames[0].result is response.results[0]
    assert frames[0].finalized


@pytest.mark.asyncio
async def test_tagged_interim_result_maps_speaker_user_id(monkeypatch):
    """Tagged interim word metadata is exposed on the interim frame."""
    service = _make_service(settings=NvidiaSTTService.Settings(speaker_diarization=True))
    response = _streaming_response(
        "Hello there",
        words=[_word("Hello", 2), _word("there", 2)],
        is_final=False,
    )

    frames = await _capture_response_frames(monkeypatch, service, response)

    assert len(frames) == 1
    assert isinstance(frames[0], InterimTranscriptionFrame)
    assert frames[0].text == "Hello there"
    assert frames[0].user_id == "2"


@pytest.mark.asyncio
async def test_mixed_speakers_emit_contiguous_run_frames(monkeypatch):
    """A mixed-speaker NVIDIA result becomes one frame per contiguous run."""
    service = _make_service(settings=NvidiaSTTService.Settings(speaker_diarization=True))
    response = _streaming_response(
        "Hi there hello again",
        words=[
            _word("Hi", 0),
            _word("there", 0),
            _word("hello", 1),
            _word("again", 0),
        ],
    )

    frames = await _capture_response_frames(monkeypatch, service, response)

    assert [frame.text for frame in frames] == ["Hi there ", "hello ", "again"]
    assert [frame.user_id for frame in frames] == ["0", "1", "0"]
    assert len({frame.timestamp for frame in frames}) == 1
    assert all(isinstance(frame, TranscriptionFrame) for frame in frames)
    assert [frame.finalized for frame in frames] == [False, False, True]
    assert all(frame.includes_inter_frame_spaces for frame in frames)
    assert _aggregate(frames) == "Hi there hello again"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transcript", "words", "expected"),
    [
        (
            "Hi, hello!",
            [_word("Hi", 0), _word("hello", 1)],
            ["Hi, ", "hello!"],
        ),
        (
            "你好世界",
            [_word("你好", 0), _word("世界", 1)],
            ["你好", "世界"],
        ),
    ],
)
async def test_speaker_runs_preserve_provider_text(monkeypatch, transcript, words, expected):
    """Splitting speaker runs preserves punctuation, whitespace, and CJK text."""
    service = _make_service(settings=NvidiaSTTService.Settings(speaker_diarization=True))
    response = _streaming_response(transcript, words=words)

    frames = await _capture_response_frames(monkeypatch, service, response)

    assert [frame.text for frame in frames] == expected
    assert _aggregate(frames) == transcript


@pytest.mark.asyncio
async def test_speaker_format_labels_zero_speaker_tag(monkeypatch):
    """Speaker formatting includes tag zero and normalizes provider whitespace."""
    service = _make_service(
        settings=NvidiaSTTService.Settings(
            speaker_diarization=True,
            speaker_format="Speaker {speaker}: {text}",
        )
    )
    response = _streaming_response(
        " Hello",
        words=[_word("Hello", 0)],
    )

    frames = await _capture_response_frames(monkeypatch, service, response)

    assert frames[0].text == " Speaker 0: Hello"
    assert frames[0].user_id == "0"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "words",
    [
        None,
        [_word("No"), _word("speaker"), _word("metadata")],
    ],
)
async def test_missing_speaker_metadata_falls_back_to_full_transcript(monkeypatch, words):
    """Missing word or speaker metadata preserves the provider transcript."""
    service = _make_service(settings=NvidiaSTTService.Settings(speaker_diarization=True))
    service._user_id = "session-user"
    response = _streaming_response("No speaker metadata", words=words)

    frames = await _capture_response_frames(monkeypatch, service, response)

    assert len(frames) == 1
    assert frames[0].text == "No speaker metadata"
    assert frames[0].user_id == "session-user"


@pytest.mark.asyncio
async def test_deferred_diarization_update_uses_active_stream_config(monkeypatch):
    """A deferred reconnect does not reinterpret results from the old stream."""
    service = _make_service()
    service._user_id = "session-user"
    service._config = service._create_recognition_config()
    old_config = service._config
    service._can_reconnect = False

    await service._update_settings(NvidiaSTTService.Settings(speaker_diarization=True))

    assert service._need_reconnect
    assert not old_config.config.diarization_config.enable_speaker_diarization
    assert service._config.config.diarization_config.enable_speaker_diarization

    response = _streaming_response("Hello", words=[_word("Hello", 0)])
    frames = await _capture_response_frames(
        monkeypatch,
        service,
        response,
        speaker_diarization=False,
    )

    assert frames[0].user_id == "session-user"

    frames = await _capture_response_frames(
        monkeypatch,
        service,
        response,
        speaker_diarization=True,
    )

    assert frames[0].user_id == "0"
