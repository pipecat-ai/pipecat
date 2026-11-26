#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the audio handling shared by every STTService."""

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

import numpy as np
import pytest

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    EagerTranscriptionFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    OutputAudioRawFrame,
    STTMetadataFrame,
    STTMuteFrame,
    TranscriptionFinalizedFrame,
    TranscriptionFrame,
    TranscriptionSpeaker,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_service import AIService
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService, STTService, TranscriptionType
from pipecat.tests.utils import run_test
from pipecat.turns.eager_end_of_turn_mixin import EagerEndOfTurnSTTServiceMixin
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


def _make_capturing_service(send_sample_rate: int | None = None, **kwargs) -> STTService:
    """Build an STTService that records the audio handed to run_stt().

    Defined as a factory (not a module-level class) so this concrete subclass
    isn't picked up by the service-discovery scan in test_service_init.py.
    """

    class _CapturingSTTService(STTService):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.captured: list[bytes] = []

        @property
        def send_sample_rate(self) -> int:
            return send_sample_rate or self.sample_rate

        async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
            self.captured.append(audio)
            return
            yield  # make this an async generator

    return _CapturingSTTService(**kwargs)


def _input_audio() -> InputAudioRawFrame:
    return InputAudioRawFrame(audio=INPUT_PCM, sample_rate=SAMPLE_RATE, num_channels=1)


def _tone(rate: int, seconds: float) -> InputAudioRawFrame:
    t = np.arange(int(rate * seconds)) / rate
    audio = (np.sin(2 * np.pi * 220 * t) * 8000).astype(np.int16).tobytes()
    return InputAudioRawFrame(audio=audio, sample_rate=rate, num_channels=1)


async def _resampled(frame: InputAudioRawFrame, out_rate: int) -> bytes:
    return await create_stream_resampler().resample(frame.audio, frame.sample_rate, out_rate)


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


@pytest.mark.asyncio
async def test_input_audio_at_another_rate_is_resampled_to_the_sample_rate():
    service = _make_capturing_service()
    frame = _tone(24000, 0.1)

    await run_test(
        service,
        frames_to_send=[frame],
        expected_down_frames=[STTMetadataFrame, InputAudioRawFrame],
    )

    assert service.captured == [await _resampled(frame, SAMPLE_RATE)]


@pytest.mark.asyncio
async def test_input_audio_is_resampled_to_the_send_sample_rate():
    service = _make_capturing_service(send_sample_rate=8000)
    frame = _tone(SAMPLE_RATE, 0.1)

    await run_test(
        service,
        frames_to_send=[frame],
        expected_down_frames=[STTMetadataFrame, InputAudioRawFrame],
    )

    assert service.captured == [await _resampled(frame, 8000)]


@pytest.mark.asyncio
async def test_usage_is_counted_at_the_send_sample_rate():
    service = _make_capturing_service(send_sample_rate=8000)
    service._sample_rate = 48000

    await service.process_audio_frame(_tone(48000, 1.0), FrameDirection.DOWNSTREAM)

    assert len(service.captured[0]) == 8000 * 2
    assert service._stt_usage_pending_seconds == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_muting_starts_the_resampled_audio_afresh():
    service = _make_capturing_service()
    before, after = _tone(24000, 0.1), _tone(24000, 0.1)

    await run_test(
        service,
        frames_to_send=[before, STTMuteFrame(mute=True), STTMuteFrame(mute=False), after],
        expected_down_frames=[STTMetadataFrame, InputAudioRawFrame, InputAudioRawFrame],
    )

    assert service.captured[-1] == await _resampled(after, SAMPLE_RATE)


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


@pytest.mark.asyncio
async def test_a_confirmation_without_text_is_pushed_as_a_frame_of_its_own(pushed_frames):
    """The transcript arrived before the service confirmed it was the last."""
    service = _make_capturing_service()
    service._user_id = "session-user"
    stopped: list[bool] = []

    async def stop_ttfb_metrics(**kwargs):
        stopped.append(True)

    service.stop_ttfb_metrics = stop_ttfb_metrics

    await service._push_transcription_runs([(None, "Hi.")])
    service.request_finalize()
    service.confirm_finalize()
    await service._push_transcription_finalized()

    transcript, confirmation = pushed_frames
    assert (transcript.text, transcript.finalized) == ("Hi.", False)
    assert isinstance(confirmation, TranscriptionFinalizedFrame)
    assert confirmation.user_id == "session-user"
    # The wait for the final transcript ends at the confirmation.
    assert stopped == [True]

    # One confirmation finalizes once.
    await service._push_transcription_finalized()
    assert len(pushed_frames) == 2


@pytest.mark.asyncio
async def test_no_finalized_frame_without_a_confirmed_finalization(pushed_frames):
    """Nothing is finalized unless the service asked and the server confirmed."""
    service = _make_capturing_service()

    await service._push_transcription_finalized()
    service.request_finalize()
    await service._push_transcription_finalized()

    assert pushed_frames == []


@pytest.mark.asyncio
async def test_a_confirmation_with_text_finalizes_its_transcript_only(pushed_frames):
    """An utterance is finalized by its transcript or by the frame, never both."""
    service = _make_capturing_service()

    service.request_finalize()
    service.confirm_finalize()
    await service._push_transcription_runs([(None, "Hi.")])
    await service._push_transcription_finalized()

    assert [(type(f), f.finalized) for f in pushed_frames] == [(TranscriptionFrame, True)]


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


async def _mask_card(text: str, text_type: str) -> str:
    return text.replace("4111 1111 1111 1111", "[card number]")


def _transcription(text: str, frame_class: type = TranscriptionFrame) -> Frame:
    return frame_class(text, "user", "now", result={"transcript": text})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("frame_class", "text_type"),
    [
        (TranscriptionFrame, TranscriptionType.FINAL),
        (InterimTranscriptionFrame, TranscriptionType.INTERIM),
    ],
)
async def test_text_transforms_rewrite_a_transcript_before_it_is_pushed(
    pushed_frames, frame_class, text_type
):
    types: list[str] = []

    async def transform(text: str, text_type: str) -> str:
        types.append(text_type)
        return await _mask_card(text, text_type)

    service = _make_capturing_service(text_transforms=[("*", transform)])

    await service.push_frame(_transcription("It's 4111 1111 1111 1111", frame_class))

    assert [f.text for f in pushed_frames] == ["It's [card number]"]
    assert types == [text_type]
    # The provider's result has the transcript as heard, so it goes.
    assert pushed_frames[0].result is None


@pytest.mark.asyncio
async def test_a_transcript_a_transform_leaves_alone_keeps_its_result(pushed_frames):
    service = _make_capturing_service(text_transforms=[("*", _mask_card)])

    await service.push_frame(_transcription("Hello"))

    assert [(f.text, f.result) for f in pushed_frames] == [("Hello", {"transcript": "Hello"})]


@pytest.mark.asyncio
async def test_text_transforms_apply_to_their_text_type_in_order(pushed_frames):
    async def upper(text: str, text_type: str) -> str:
        return text.upper()

    async def exclaim(text: str, text_type: str) -> str:
        return f"{text}!"

    service = _make_capturing_service(
        text_transforms=[(TranscriptionType.FINAL, upper), ("*", exclaim)]
    )

    await service.push_frame(_transcription("hi"))
    await service.push_frame(_transcription("hi", InterimTranscriptionFrame))

    assert [f.text for f in pushed_frames] == ["HI!", "hi!"]


@pytest.mark.asyncio
async def test_an_eager_transcript_is_rewritten_like_a_final_one(pushed_frames):
    class _EagerSTTService(EagerEndOfTurnSTTServiceMixin, STTService):
        async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
            yield None

    service = _EagerSTTService(
        enable_eager_end_of_turn=True, text_transforms=[(TranscriptionType.FINAL, _mask_card)]
    )

    await service._push_eager_end_of_turn(
        "It's 4111 1111 1111 1111", user_id="user", result="as heard"
    )

    assert [(type(f), f.text, f.result) for f in pushed_frames] == [
        (EagerTranscriptionFrame, "It's [card number]", None)
    ]


@pytest.mark.asyncio
async def test_text_transformers_can_be_added_and_removed(pushed_frames):
    service = _make_capturing_service()

    service.add_text_transformer(_mask_card)
    await service.push_frame(_transcription("4111 1111 1111 1111"))
    service.remove_text_transformer(_mask_card)
    await service.push_frame(_transcription("4111 1111 1111 1111"))

    assert [f.text for f in pushed_frames] == ["[card number]", "4111 1111 1111 1111"]


@pytest.mark.asyncio
async def test_a_transcription_going_upstream_isnt_transformed(pushed_frames):
    service = _make_capturing_service(text_transforms=[("*", _mask_card)])

    await service.push_frame(_transcription("4111 1111 1111 1111"), FrameDirection.UPSTREAM)

    assert [f.text for f in pushed_frames] == ["4111 1111 1111 1111"]


@pytest.mark.asyncio
async def test_observers_see_only_the_transformed_transcript():
    """An observer, such as the RTVI client's transcript, sees a frame as it's pushed."""

    class TranscriptObserver(BaseObserver):
        def __init__(self):
            super().__init__()
            self.texts: list[str] = []

        async def on_push_frame(self, data: FramePushed):
            if isinstance(data.frame, TranscriptionFrame):
                self.texts.append(data.frame.text)

    class _TranscribingSTTService(STTService):
        async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
            yield _transcription("It's 4111 1111 1111 1111")

    observer = TranscriptObserver()
    service = _TranscribingSTTService(
        settings=STTSettings(model=None, language=None),
        text_transforms=[("*", _mask_card)],
    )

    await run_test(
        service,
        frames_to_send=[_input_audio()],
        expected_down_frames=[STTMetadataFrame, TranscriptionFrame, InputAudioRawFrame],
        observers=[observer],
    )

    assert observer.texts and set(observer.texts) == {"It's [card number]"}
