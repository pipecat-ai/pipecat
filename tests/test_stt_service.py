#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the audio handling shared by every STTService."""

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

import pytest

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    TranscriptionSpeaker,
)
from pipecat.services.ai_service import AIService
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService, STTService
from pipecat.tests.utils import run_test
from pipecat.utils.types import NOT_GIVEN, NotGiven

SAMPLE_RATE = 16000
# Distinct payloads so the audio handed to run_stt() shows which frame it came from.
INPUT_PCM = b"\x01\x00" * 160
OUTPUT_PCM = b"\x02\x00" * 160


@dataclass
class _DiarizingSettings(STTSettings):
    speaker_format: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


def _diarizing_settings(speaker_format: str | None = None) -> _DiarizingSettings:
    return _DiarizingSettings(model=None, language=None, speaker_format=speaker_format)


def _make_capturing_service(**kwargs) -> STTService:
    """Build an STTService that records the audio handed to run_stt().

    Defined as a factory (not a module-level class) so this concrete subclass
    isn't picked up by the service-discovery scan in test_service_init.py.
    """

    class _CapturingSTTService(STTService):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.captured: list[bytes] = []

        async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
            self.captured.append(audio)
            return
            yield  # make this an async generator

    return _CapturingSTTService(**kwargs)


def _input_audio() -> InputAudioRawFrame:
    return InputAudioRawFrame(audio=INPUT_PCM, sample_rate=SAMPLE_RATE, num_channels=1)


def _output_audio() -> OutputAudioRawFrame:
    return OutputAudioRawFrame(audio=OUTPUT_PCM, sample_rate=SAMPLE_RATE, num_channels=1)


@pytest.mark.asyncio
async def test_output_audio_is_forwarded_without_being_transcribed():
    service = _make_capturing_service()

    await run_test(
        service,
        frames_to_send=[_input_audio(), _output_audio()],
        expected_down_frames=[STTMetadataFrame, InputAudioRawFrame, OutputAudioRawFrame],
    )

    assert service.captured == [INPUT_PCM]


@pytest.mark.asyncio
async def test_output_audio_is_forwarded_even_without_audio_passthrough():
    service = _make_capturing_service(audio_passthrough=False)

    await run_test(
        service,
        frames_to_send=[_input_audio(), _output_audio()],
        expected_down_frames=[STTMetadataFrame, OutputAudioRawFrame],
    )

    assert service.captured == [INPUT_PCM]


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
    assert STTService._speaker_runs(" Hi there hello ", words) == expected


def test_speaker_runs_locate_repeated_words_in_order():
    """A repeated word is matched after the previous one, not at its first occurrence."""
    runs = STTService._speaker_runs(
        "no, no, no.",
        [("no", 0), ("no", 1), ("no", 0)],
    )

    assert runs == [(0, "no,"), (1, "no,"), (0, "no.")]


def test_speaker_runs_keep_provider_label_type():
    """String labels pass through unchanged."""
    runs = STTService._speaker_runs("Hi. Hello.", [("Hi.", "A"), ("Hello.", "B")])

    assert runs == [("A", "Hi."), ("B", "Hello.")]


SPEAKER_A = TranscriptionSpeaker("A", 2)


@pytest.mark.parametrize(
    ("speaker_format", "speaker", "expected"),
    [
        ("{speaker}: {text}", SPEAKER_A, "A: hello"),
        ("Speaker {speaker_number}: {text}", SPEAKER_A, "Speaker 2: hello"),
        ("{speaker}: {text}", None, "hello"),
        (None, SPEAKER_A, "hello"),
    ],
)
def test_apply_speaker_format(speaker_format, speaker, expected):
    """The format is applied only when both a format and a speaker are present."""
    service = _make_capturing_service(settings=_diarizing_settings(speaker_format))

    assert service._apply_speaker_format("hello", speaker) == expected


def test_apply_speaker_format_without_the_setting():
    """A service whose settings have no ``speaker_format`` leaves text unchanged."""
    service = _make_capturing_service()

    assert service._apply_speaker_format("hello", SPEAKER_A) == "hello"


def test_speaker_numbers_follow_order_of_first_appearance():
    """Labels are numbered in the order speakers first appear, whatever the provider calls them."""
    service = _make_capturing_service()

    speakers = [service._speaker(label) for label in ["S7", "A", "S7", 0, "A"]]

    assert speakers == [
        TranscriptionSpeaker("S7", 1),
        TranscriptionSpeaker("A", 2),
        TranscriptionSpeaker("S7", 1),
        TranscriptionSpeaker("0", 3),
        TranscriptionSpeaker("A", 2),
    ]
    assert service._speaker(None) is None


def test_diarized_runs_give_each_run_its_speaker_and_format():
    service = _make_capturing_service(settings=_diarizing_settings("{speaker}: {text}"))

    runs = service._diarized_runs("Hi. Hello.", [("Hi.", 0), ("Hello.", 1)])

    assert runs == [
        (TranscriptionSpeaker("0", 1), "0: Hi."),
        (TranscriptionSpeaker("1", 2), "1: Hello."),
    ]


@pytest.mark.parametrize("words", [None, [], [("Hi.", 0), ("missing", 1)]])
def test_diarized_runs_fall_back_to_the_whole_transcript(words):
    service = _make_capturing_service(settings=_diarizing_settings("{speaker}: {text}"))

    assert service._diarized_runs("Hi. Hello.", words) == [(None, "Hi. Hello.")]


@pytest.fixture
def pushed_frames(monkeypatch) -> list[Frame]:
    """Capture frames below STTService.push_frame, so its finalize handling still runs."""
    pushed: list[Frame] = []

    async def fake_push_frame(self, frame, direction=None):
        pushed.append(frame)

    monkeypatch.setattr(AIService, "push_frame", fake_push_frame)
    return pushed


@pytest.mark.asyncio
@pytest.mark.parametrize("confirmed", [False, True])
async def test_push_transcription_runs_finalizes_only_the_last_run(pushed_frames, confirmed):
    """A finalized transcript ends the turn, so earlier runs must not be finalized."""
    service = _make_capturing_service()
    service._user_id = "session-user"
    if confirmed:
        service.request_finalize()
        service.confirm_finalize()

    runs = [(TranscriptionSpeaker("0", 1), "Hi."), (TranscriptionSpeaker("1", 2), "Hello.")]
    await service._push_transcription_runs(runs, finalized=not confirmed)

    assert [(f.speaker, f.text, f.finalized) for f in pushed_frames] == [
        (TranscriptionSpeaker("0", 1), "Hi.", False),
        (TranscriptionSpeaker("1", 2), "Hello.", True),
    ]
    assert all(f.user_id == "session-user" for f in pushed_frames)


def _make_segmented_service(frames: list[Frame]) -> SegmentedSTTService:
    class _SegmentedSTTService(SegmentedSTTService):
        async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
            for frame in frames:
                yield frame

    return _SegmentedSTTService()


@pytest.mark.asyncio
async def test_segment_finalizes_only_the_last_of_consecutive_transcripts(pushed_frames):
    first = TranscriptionFrame("Hi.", "", "", speaker=TranscriptionSpeaker("0", 1))
    second = TranscriptionFrame("Hello.", "", "", speaker=TranscriptionSpeaker("1", 2))
    service = _make_segmented_service([first, second])

    await service._push_segment_frames(b"audio")

    assert pushed_frames == [first, second]
    assert [f.finalized for f in pushed_frames] == [False, True]


@pytest.mark.asyncio
async def test_segment_keeps_frame_order_around_other_frames(pushed_frames, monkeypatch):
    first = TranscriptionFrame("Hi.", "", "")
    error = ErrorFrame(error="boom")
    second = TranscriptionFrame("Hello.", "", "")
    service = _make_segmented_service([first, error, second])
    errors = []

    async def fake_push_error_frame(frame):
        errors.append((len(pushed_frames), frame))

    monkeypatch.setattr(service, "push_error_frame", fake_push_error_frame)

    await service._push_segment_frames(b"audio")

    assert pushed_frames == [first, second]
    assert errors == [(1, error)]
    assert first.finalized and second.finalized


@pytest.mark.asyncio
async def test_segment_pushes_unlabelled_transcript_without_waiting(pushed_frames):
    """A transcript with no speaker is pushed before run_stt resumes."""
    transcript = TranscriptionFrame("Hello.", "", "")
    pushed_before_resume = []

    class _SegmentedSTTService(SegmentedSTTService):
        async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
            yield transcript
            pushed_before_resume.append(list(pushed_frames))

    await _SegmentedSTTService()._push_segment_frames(b"audio")

    assert pushed_before_resume == [[transcript]]
    assert transcript.finalized
