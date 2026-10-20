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


def _word(text: str, speaker_tag: int = 0) -> SimpleNamespace:
    # Riva's ``speaker_tag`` is a proto3 int: speakers are numbered from 1, and an
    # untagged word reads as 0.
    return SimpleNamespace(word=text, speaker_tag=speaker_tag)


def _streaming_response(
    transcript: str,
    *,
    words: list[SimpleNamespace] | None = None,
    is_final: bool = True,
) -> SimpleNamespace:
    alternative = SimpleNamespace(transcript=transcript, words=words or [])
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


@pytest.mark.asyncio
async def test_speaker_format_update_applies_to_next_result(monkeypatch):
    """A speaker format update keeps the stream config and labels later results."""
    service = _make_service(settings=NvidiaSTTService.Settings(speaker_diarization=True))
    service._config = service._create_recognition_config()
    config = service._config
    monkeypatch.setattr(service, "_request_reconnect", AsyncMock())
    response = _streaming_response("Hello", words=[_word("Hello", 1)])

    await service._update_settings(NvidiaSTTService.Settings(speaker_format="[{speaker}] {text}"))
    frames = await _capture_response_frames(monkeypatch, service, response)

    assert service._config is config
    assert frames[0].text == "[1] Hello"


@pytest.mark.asyncio
async def test_speaker_format_with_server_setting_still_reconnects(monkeypatch):
    """A delta mixing local and server settings still reconnects."""
    service = _make_service()
    service._config = service._create_recognition_config()
    reconnect = AsyncMock()
    monkeypatch.setattr(service, "_request_reconnect", reconnect)

    await service._update_settings(
        NvidiaSTTService.Settings(speaker_format="{speaker}: {text}", language=Language.ES)
    )

    reconnect.assert_awaited_once()
    assert service._config.config.language_code == Language.ES


def test_diarization_max_speakers_reaches_recognition_config():
    """The configured speaker limit is sent with the diarization config."""
    service = _make_service(
        settings=NvidiaSTTService.Settings(speaker_diarization=True, diarization_max_speakers=3)
    )

    config = service._create_recognition_config()

    assert config.config.diarization_config.max_speaker_count == 3


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
        words=[_word("Hi", 1), _word("there", 2)],
    )

    frames = await _capture_response_frames(monkeypatch, service, response)

    assert len(frames) == 1
    assert frames[0].text == "Hi there"
    assert frames[0].user_id == "session-user"


@pytest.mark.asyncio
async def test_tagged_final_result_maps_speaker_user_id(monkeypatch):
    """A single-speaker final result carries the speaker tag as its user ID."""
    service = _make_service(settings=NvidiaSTTService.Settings(speaker_diarization=True))
    response = _streaming_response(
        " Hello there",
        words=[_word("Hello", 1), _word("there", 1)],
    )

    frames = await _capture_response_frames(monkeypatch, service, response)

    assert len(frames) == 1
    assert isinstance(frames[0], TranscriptionFrame)
    assert frames[0].text == "Hello there"
    assert frames[0].user_id == "1"
    assert frames[0].result is response.results[0]
    assert frames[0].finalized


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "words",
    [
        [],
        [_word("Hi", 1), _word("there", 1), _word("hello", 2)],
    ],
)
async def test_interim_results_are_not_split_by_speaker(monkeypatch, words):
    """Interim results are emitted whole, unlabelled, with the session user."""
    service = _make_service(
        settings=NvidiaSTTService.Settings(
            speaker_diarization=True,
            speaker_format="Speaker {speaker}: {text}",
        )
    )
    service._user_id = "session-user"
    response = _streaming_response(" Hi there hello", words=words, is_final=False)

    frames = await _capture_response_frames(monkeypatch, service, response)

    assert len(frames) == 1
    assert isinstance(frames[0], InterimTranscriptionFrame)
    assert frames[0].text == " Hi there hello"
    assert frames[0].user_id == "session-user"


@pytest.mark.asyncio
async def test_mixed_speakers_emit_contiguous_run_frames(monkeypatch):
    """A mixed-speaker NVIDIA result becomes one frame per contiguous run."""
    service = _make_service(settings=NvidiaSTTService.Settings(speaker_diarization=True))
    response = _streaming_response(
        "Hi there hello again",
        words=[
            _word("Hi", 1),
            _word("there", 1),
            _word("hello", 2),
            _word("again", 1),
        ],
    )

    frames = await _capture_response_frames(monkeypatch, service, response)

    assert [frame.text for frame in frames] == ["Hi there", "hello", "again"]
    assert [frame.user_id for frame in frames] == ["1", "2", "1"]
    assert len({frame.timestamp for frame in frames}) == 1
    assert all(isinstance(frame, TranscriptionFrame) for frame in frames)
    assert [frame.finalized for frame in frames] == [False, False, True]
    assert not any(frame.includes_inter_frame_spaces for frame in frames)
    assert _aggregate(frames) == "Hi there hello again"


@pytest.mark.asyncio
async def test_split_result_reports_usage_and_tracing_once(monkeypatch):
    """Usage metrics and tracing see one full transcript per result, not one per run."""
    service = _make_service(settings=NvidiaSTTService.Settings(speaker_diarization=True))
    calls = []
    monkeypatch.setattr(
        service, "push_frame", AsyncMock(side_effect=lambda f: calls.append(("frame", f.text)))
    )
    monkeypatch.setattr(
        service, "emit_stt_usage_metrics", AsyncMock(side_effect=lambda: calls.append(("usage",)))
    )
    handle_transcription = AsyncMock()
    monkeypatch.setattr(service, "_handle_transcription", handle_transcription)
    response = _streaming_response(
        "Hi there hello",
        words=[_word("Hi", 1), _word("there", 1), _word("hello", 2)],
    )

    await service._handle_response(response, speaker_diarization=True)

    assert calls == [("usage",), ("frame", "Hi there"), ("frame", "hello")]
    handle_transcription.assert_awaited_once()
    assert handle_transcription.await_args.kwargs["transcript"] == "Hi there hello"


@pytest.mark.asyncio
async def test_each_result_in_a_response_is_split_and_finalized(monkeypatch):
    """Every final result in one response finalizes its own last run."""
    service = _make_service(settings=NvidiaSTTService.Settings(speaker_diarization=True))
    first = _streaming_response("Hi there", words=[_word("Hi", 1), _word("there", 2)])
    second = _streaming_response("Bye", words=[_word("Bye", 2)])
    response = SimpleNamespace(results=first.results + second.results)

    frames = await _capture_response_frames(monkeypatch, service, response)

    assert [frame.text for frame in frames] == ["Hi", "there", "Bye"]
    assert [frame.finalized for frame in frames] == [False, True, True]
    assert frames[0].result is first.results[0]
    assert frames[2].result is second.results[0]


@pytest.mark.asyncio
async def test_separate_results_keep_joining_spaces(monkeypatch):
    """Split runs and separate results are both joined by the aggregator."""
    service = _make_service(settings=NvidiaSTTService.Settings(speaker_diarization=True))
    split = _streaming_response(
        "Hi there hello",
        words=[_word("Hi", 1), _word("there", 1), _word("hello", 2)],
    )
    single = _streaming_response("Again", words=[_word("Again", 2)])

    frames = await _capture_response_frames(monkeypatch, service, split)
    frames += await _capture_response_frames(monkeypatch, service, single)

    assert not any(frame.includes_inter_frame_spaces for frame in frames)
    assert _aggregate(frames) == "Hi there hello Again"


@pytest.mark.asyncio
async def test_consecutive_mixed_speaker_results_keep_result_spacing(monkeypatch):
    """Consecutive split results join cleanly with or without a leading space."""
    service = _make_service(settings=NvidiaSTTService.Settings(speaker_diarization=True))
    first = _streaming_response(
        "Hi there hello",
        words=[_word("Hi", 1), _word("there", 1), _word("hello", 2)],
    )
    second = _streaming_response(
        "Hello again friend.",
        words=[_word("Hello", 2), _word("again", 2), _word("friend", 1)],
    )
    third = _streaming_response(
        " Goodnight moon.",
        words=[_word("Goodnight", 1), _word("moon", 2)],
    )

    frames = await _capture_response_frames(monkeypatch, service, first)
    frames += await _capture_response_frames(monkeypatch, service, second)
    frames += await _capture_response_frames(monkeypatch, service, third)

    assert [frame.text for frame in frames] == [
        "Hi there",
        "hello",
        "Hello again",
        "friend.",
        "Goodnight",
        "moon.",
    ]
    assert _aggregate(frames) == "Hi there hello Hello again friend. Goodnight moon."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transcript", "words", "expected", "aggregated"),
    [
        (
            "Hi, hello!",
            [_word("Hi", 1), _word("hello", 2)],
            ["Hi,", "hello!"],
            "Hi, hello!",
        ),
        (
            "你好世界",
            [_word("你好", 1), _word("世界", 2)],
            ["你好", "世界"],
            "你好 世界",
        ),
    ],
)
async def test_speaker_runs_preserve_provider_text(
    monkeypatch, transcript, words, expected, aggregated
):
    """Splitting speaker runs preserves punctuation and CJK text within each run."""
    service = _make_service(settings=NvidiaSTTService.Settings(speaker_diarization=True))
    response = _streaming_response(transcript, words=words)

    frames = await _capture_response_frames(monkeypatch, service, response)

    assert [frame.text for frame in frames] == expected
    assert _aggregate(frames) == aggregated


@pytest.mark.parametrize(
    ("words", "expected"),
    [
        ([("Hi", 0), ("there", 0), ("hello", 1)], [(0, "Hi there"), (1, "hello")]),
        ([("Hi", 0), ("there", None)], []),
        ([("Hi", 0), ("missing", 1)], []),
        ([("Hi", 0), ("", 1)], []),
        ([], []),
    ],
)
def test_speaker_runs_from_word_pairs(words, expected):
    """Speaker runs are built from plain (text, speaker) pairs."""
    assert NvidiaSTTService._speaker_runs(" Hi there hello ", words) == expected


def test_speaker_runs_locate_repeated_words_in_order():
    """A repeated word is matched after the previous one, not at its first occurrence."""
    runs = NvidiaSTTService._speaker_runs(
        "no, no, no.",
        [("no", 0), ("no", 1), ("no", 0)],
    )

    assert runs == [(0, "no,"), (1, "no,"), (0, "no.")]


@pytest.mark.asyncio
async def test_speaker_format_labels_speaker_and_strips_whitespace(monkeypatch):
    """Speaker formatting labels the run and strips provider whitespace."""
    service = _make_service(
        settings=NvidiaSTTService.Settings(
            speaker_diarization=True,
            speaker_format="Speaker {speaker}: {text}",
        )
    )
    response = _streaming_response(
        " Hello",
        words=[_word("Hello", 1)],
    )

    frames = await _capture_response_frames(monkeypatch, service, response)

    assert frames[0].text == "Speaker 1: Hello"
    assert frames[0].user_id == "1"


@pytest.mark.asyncio
async def test_speaker_format_labels_each_run(monkeypatch):
    """Each speaker run gets its own label."""
    service = _make_service(
        settings=NvidiaSTTService.Settings(
            speaker_diarization=True,
            speaker_format="Speaker {speaker}: {text}",
        )
    )
    response = _streaming_response(
        "Hi there hello",
        words=[_word("Hi", 1), _word("there", 1), _word("hello", 2)],
    )

    frames = await _capture_response_frames(monkeypatch, service, response)

    assert [frame.text for frame in frames] == ["Speaker 1: Hi there", "Speaker 2: hello"]


@pytest.mark.asyncio
async def test_speaker_format_is_not_applied_to_unsplit_fallback(monkeypatch):
    """A result that can't be split is emitted unlabelled."""
    service = _make_service(
        settings=NvidiaSTTService.Settings(
            speaker_diarization=True,
            speaker_format="Speaker {speaker}: {text}",
        )
    )
    service._user_id = "session-user"
    response = _streaming_response("No speaker", words=[_word("No", 1), _word("missing", 2)])

    frames = await _capture_response_frames(monkeypatch, service, response)

    assert [frame.text for frame in frames] == ["No speaker"]
    assert frames[0].user_id == "session-user"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "words",
    [
        [],
        [_word("No", 1), _word("missing", 2), _word("metadata", 1)],
    ],
)
async def test_unsplittable_result_falls_back_to_full_transcript(monkeypatch, words):
    """No word information, or a word not in the transcript, keeps the transcript whole."""
    service = _make_service(settings=NvidiaSTTService.Settings(speaker_diarization=True))
    service._user_id = "session-user"
    response = _streaming_response("No speaker metadata", words=words)

    frames = await _capture_response_frames(monkeypatch, service, response)

    assert len(frames) == 1
    assert frames[0].text == "No speaker metadata"
    assert frames[0].user_id == "session-user"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "words",
    [
        [_word("No"), _word("speaker"), _word("tags")],
        [_word("No", 1), _word("speaker"), _word("tags", 2)],
    ],
)
async def test_untagged_words_fall_back_to_session_user(monkeypatch, words):
    """A result with a word the server did not tag (speaker 0) is not split."""
    service = _make_service(
        settings=NvidiaSTTService.Settings(
            speaker_diarization=True,
            speaker_format="Speaker {speaker}: {text}",
        )
    )
    service._user_id = "session-user"
    response = _streaming_response("No speaker tags", words=words)

    frames = await _capture_response_frames(monkeypatch, service, response)

    assert [frame.text for frame in frames] == ["No speaker tags"]
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

    response = _streaming_response("Hello", words=[_word("Hello", 1)])
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

    assert frames[0].user_id == "1"


@pytest.mark.asyncio
@pytest.mark.parametrize("speaker_diarization", [False, True])
async def test_response_handler_reads_diarization_from_stream_config(
    monkeypatch, speaker_diarization
):
    """Whether results are split follows the config the stream was opened with."""
    service = _make_service(
        settings=NvidiaSTTService.Settings(speaker_diarization=speaker_diarization)
    )
    service._user_id = "session-user"
    config = service._create_recognition_config()
    response = _streaming_response("Hi there", words=[_word("Hi", 1), _word("there", 2)])
    service._asr_service = SimpleNamespace(
        streaming_response_generator=lambda audio_chunks, streaming_config: iter([response])
    )
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(service, "get_event_loop", lambda: loop)
    monkeypatch.setattr(service, "_handle_stream_drop", AsyncMock())
    handled = asyncio.Event()
    received = []

    async def handle_response(response, *, speaker_diarization=False):
        received.append(speaker_diarization)
        handled.set()

    monkeypatch.setattr(service, "_handle_response", handle_response)

    iterator = AudioChunkIterator(loop)
    await asyncio.to_thread(service._response_handler, iterator, config)
    await asyncio.wait_for(handled.wait(), timeout=1)

    assert received == [speaker_diarization]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
