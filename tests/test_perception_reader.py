#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the processor that reads each user turn by asking a classifier."""

import asyncio

import pytest

from pipecat.classifiers.base_classifier import (
    BaseClassifier,
    ChoiceQuestion,
    ChoiceResult,
    ClassifierAudio,
    ClassifierError,
    ScoreLevel,
    ScoreQuestion,
    ScoreResult,
    YesNoQuestion,
    YesNoResult,
)
from pipecat.expressive.perception import PerceptionSignal
from pipecat.expressive.processors import perception_reader as reader_module
from pipecat.expressive.processors.perception_reader import (
    PerceptionQuestion,
    PerceptionReader,
    PerceptionSource,
)
from pipecat.frames.frames import (
    ErrorFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMContextAssistantTurnFrame,
    MetricsFrame,
    PerceptionFrame,
    PerceptionMetadataFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    UserMuteStartedFrame,
    UserMuteStoppedFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.worker import PipelineParams
from pipecat.tests.utils import SleepFrame, run_test

QUESTIONS = {
    "tone": PerceptionQuestion(
        question=ChoiceQuestion(
            instructions="How does the user sound?",
            options={"calm": None, "frustrated": None, "confused": None},
        ),
        description="How the user sounded.",
    ),
    "hesitation": PerceptionQuestion(
        question=YesNoQuestion(instructions="Does the user sound unsure?"),
        description="The chance the user sounded unsure, from 0 to 1.",
    ),
    "energy": PerceptionQuestion(
        question=ScoreQuestion(instructions="How energetic is the user?", levels=["low", "high"]),
        description="From 0, low, to 1, high.",
    ),
}

RESULTS = {
    "tone": ChoiceResult(
        choice="frustrated",
        probabilities={"calm": 0.1, "frustrated": 0.7, "confused": 0.2},
        confidence=0.6,
    ),
    "hesitation": YesNoResult(probability=0.6432),
    "energy": ScoreResult(
        score=0.678,
        levels=[
            ScoreLevel(level="low", probability=0.32),
            ScoreLevel(level="high", probability=0.68),
        ],
        confidence=0.4,
    ),
}


class FakeClassifier(BaseClassifier):
    def __init__(
        self,
        *,
        error: Exception | None = None,
        setup_error: Exception | None = None,
        delay: float = 0.0,
    ):
        super().__init__()
        self.asked: list[tuple] = []
        self._error = error
        self._setup_error = setup_error
        self._delay = delay

    async def setup(self, task_manager):
        await super().setup(task_manager)
        if self._setup_error:
            raise self._setup_error

    async def _ask(self, state, questions, inputs=()):
        self.asked.append((state, dict(questions)))
        await asyncio.sleep(self._delay)
        if self._error:
            raise self._error
        return {name: RESULTS[name] for name in questions}, None


def _transcription(text: str, *, finalized: bool = False) -> TranscriptionFrame:
    return TranscriptionFrame(text=text, user_id="", timestamp="now", finalized=finalized)


def _finalized(frames) -> list[bool]:
    return [f.finalized for f in frames if isinstance(f, TranscriptionFrame)]


def _reader(
    *,
    classifier: BaseClassifier,
    questions: dict[str, PerceptionQuestion],
    timeout: float | None = None,
) -> PerceptionReader:
    """A reader with one source, which asks ``classifier`` these questions."""
    source = PerceptionSource(classifier=classifier, questions=questions, timeout=timeout)
    return PerceptionReader(sources=[source])


@pytest.mark.asyncio
async def test_the_signals_are_declared_at_start():
    reader = _reader(classifier=FakeClassifier(), questions=QUESTIONS)

    down, _ = await run_test(reader, frames_to_send=[])

    metadata = [f for f in down if isinstance(f, PerceptionMetadataFrame)]
    assert [(t.name, t.description, t.labels) for t in metadata[0].perception.signals] == [
        ("tone", "How the user sounded.", ["calm", "frustrated", "confused"]),
        ("hesitation", "The chance the user sounded unsure, from 0 to 1.", []),
        ("energy", "From 0, low, to 1, high.", []),
    ]


@pytest.mark.asyncio
async def test_each_answer_is_set_as_a_signal_on_its_transcription():
    down, _ = await run_test(
        _reader(classifier=FakeClassifier(), questions=QUESTIONS),
        frames_to_send=[_transcription("I already gave you my order number.")],
        expected_down_frames=[PerceptionMetadataFrame, PerceptionFrame, TranscriptionFrame],
    )

    assert down[2].perception == [
        PerceptionSignal("tone", labels=[("frustrated", 0.7), ("confused", 0.2)]),
        PerceptionSignal("hesitation", value=0.64),
        PerceptionSignal("energy", value=0.68),
    ]


@pytest.mark.asyncio
async def test_a_choice_reports_at_most_its_max_labels():
    question = QUESTIONS["tone"].model_copy(update={"max_labels": 1})

    down, _ = await run_test(
        _reader(classifier=FakeClassifier(), questions={"tone": question}),
        frames_to_send=[_transcription("Hello.")],
        expected_down_frames=[PerceptionMetadataFrame, PerceptionFrame, TranscriptionFrame],
    )

    assert down[2].perception == [PerceptionSignal("tone", labels=[("frustrated", 0.7)])]


@pytest.mark.asyncio
async def test_a_turn_is_asked_about_with_the_bots_words_before_it():
    classifier = FakeClassifier()

    await run_test(
        _reader(classifier=classifier, questions=QUESTIONS),
        frames_to_send=[
            _transcription("Hi."),
            LLMContextAssistantTurnFrame(text="What's your order number?", timestamp="now"),
            _transcription("I already gave it to you."),
        ],
    )

    assert [state for state, _ in classifier.asked] == [
        {"user": "Hi."},
        {"assistant": "What's your order number?", "user": "I already gave it to you."},
    ]


@pytest.mark.asyncio
async def test_a_late_reading_lets_the_transcription_pass_unread():
    down, up = await run_test(
        _reader(classifier=FakeClassifier(delay=1.0), questions=QUESTIONS, timeout=0.05),
        frames_to_send=[_transcription("Hello.")],
        expected_down_frames=[PerceptionMetadataFrame, TranscriptionFrame],
    )

    assert down[1].perception is None
    assert not [f for f in up if isinstance(f, ErrorFrame)]


@pytest.mark.asyncio
async def test_a_failed_reading_is_reported_and_the_transcription_still_passes():
    down, up = await run_test(
        _reader(classifier=FakeClassifier(error=ClassifierError("down")), questions=QUESTIONS),
        frames_to_send=[_transcription("Hello.")],
        expected_down_frames=[PerceptionMetadataFrame, TranscriptionFrame],
    )

    assert down[1].perception is None
    assert [f for f in up if isinstance(f, ErrorFrame)]


@pytest.mark.asyncio
async def test_a_classifier_that_cant_be_set_up_leaves_the_reader_unusable():
    classifier = FakeClassifier(setup_error=ClassifierError("no model"))
    reader = _reader(classifier=classifier, questions=QUESTIONS)

    down, up = await run_test(reader, frames_to_send=[_transcription("Hello.")])

    assert not reader.is_usable
    assert [f for f in up if isinstance(f, ErrorFrame)]
    assert not [f for f in down if isinstance(f, PerceptionMetadataFrame)]
    assert [f.perception for f in down if isinstance(f, TranscriptionFrame)] == [None]
    assert classifier.asked == []


@pytest.mark.asyncio
async def test_the_classifiers_metrics_are_pushed_when_metrics_are_on():
    frames = [_transcription("Hello."), SleepFrame(sleep=0.2)]

    with_metrics, _ = await run_test(
        _reader(classifier=FakeClassifier(), questions=QUESTIONS),
        frames_to_send=frames,
        pipeline_params=PipelineParams(enable_metrics=True),
    )
    without_metrics, _ = await run_test(
        _reader(classifier=FakeClassifier(), questions=QUESTIONS),
        frames_to_send=frames,
    )

    assert [f for f in with_metrics if isinstance(f, MetricsFrame)]
    assert not [f for f in without_metrics if isinstance(f, MetricsFrame)]


def _stt_latencies(frames) -> list:
    return [f.ttfs_p99_latency for f in frames if isinstance(f, STTMetadataFrame)]


def _released(frames) -> list:
    """The order the transcriptions and the bot's replies went on in."""
    kinds = (TranscriptionFrame, LLMContextAssistantTurnFrame)
    return [type(f) for f in frames if isinstance(f, kinds)]


@pytest.mark.asyncio
async def test_a_text_reading_goes_on_as_soon_as_its_read():
    down, _ = await run_test(
        _reader(classifier=FakeClassifier(), questions=QUESTIONS),
        frames_to_send=[
            STTMetadataFrame(service_name="stt", ttfs_p99_latency=0.3),
            VADUserStartedSpeakingFrame(),
            _transcription("I already gave you my order number.", finalized=True),
            SleepFrame(sleep=0.2),
            LLMContextAssistantTurnFrame(text="Let me check.", timestamp="now"),
            VADUserStoppedSpeakingFrame(),
        ],
    )

    # It doesn't wait for the user to stop speaking, and leaves the STT's
    # finalized and latency as they are.
    assert _released(down) == [TranscriptionFrame, LLMContextAssistantTurnFrame]
    assert _finalized(down) == [True]
    assert _stt_latencies(down) == [pytest.approx(0.3)]


#: The sample rate of the pipeline's input audio in these tests.
SAMPLE_RATE = 16000

AUDIO_QUESTIONS = {"hesitation": QUESTIONS["hesitation"]}


class FakeAudioClassifier(BaseClassifier):
    """Answers every clip with RESULTS, or the nth clip's hesitation with ``probabilities[n]``."""

    def __init__(
        self,
        *,
        min_secs: float = 1.0,
        max_secs: float | None = None,
        delays: list[float] | None = None,
        probabilities: list[float] | None = None,
        setup_error: Exception | None = None,
    ):
        super().__init__()
        self.clips: list[ClassifierAudio] = []
        self.states: list = []
        self._min_secs = min_secs
        self._max_secs = max_secs
        self._delays = delays or []
        self._probabilities = probabilities
        self._setup_error = setup_error

    async def setup(self, task_manager):
        await super().setup(task_manager)
        if self._setup_error:
            raise self._setup_error

    @property
    def supports_audio(self) -> bool:
        return True

    @property
    def min_audio_secs(self) -> float:
        return self._min_secs

    @property
    def max_audio_secs(self) -> float | None:
        return self._max_secs

    async def _ask(self, state, questions, inputs=()):
        call = len(self.clips)
        self.clips.append(next(iter(inputs), None))
        self.states.append(state)
        await asyncio.sleep(self._delays[call] if call < len(self._delays) else 0.0)
        results = dict(RESULTS)
        if self._probabilities:
            results["hesitation"] = YesNoResult(probability=self._probabilities[call])
        return {name: results[name] for name in questions}, None


def _audio(value: int, secs: float) -> InputAudioRawFrame:
    """``secs`` seconds of input audio whose every byte is ``value``."""
    audio = bytes([value]) * (int(SAMPLE_RATE * secs) * 2)
    return InputAudioRawFrame(audio=audio, sample_rate=SAMPLE_RATE, num_channels=1)


def _pcm(value: int, secs: float) -> bytes:
    return _audio(value, secs).audio


def _silence(secs: float) -> bytes:
    return bytes(int(SAMPLE_RATE * secs) * 2)


def _perception(frames) -> list:
    return [f.perception for f in frames if isinstance(f, TranscriptionFrame)]


@pytest.mark.asyncio
async def test_an_audio_classifier_hears_the_users_speech():
    classifier = FakeAudioClassifier()

    down, _ = await run_test(
        _reader(classifier=classifier, questions=AUDIO_QUESTIONS),
        frames_to_send=[
            LLMContextAssistantTurnFrame(text="What's your order number?", timestamp="now"),
            SleepFrame(sleep=0.05),
            _audio(1, 2),
            VADUserStartedSpeakingFrame(),
            _audio(2, 2),
            VADUserStoppedSpeakingFrame(),
            _transcription("I already gave it to you."),
        ],
    )

    # Two seconds of speech is more than the classifier takes, so it's all.
    assert classifier.clips == [ClassifierAudio(audio=_pcm(2, 2), sample_rate=SAMPLE_RATE)]
    assert classifier.states == [{"assistant": "What's your order number?"}]
    assert _perception(down) == [[PerceptionSignal("hesitation", value=0.64)]]


@pytest.mark.asyncio
async def test_each_stop_is_heard_with_the_audio_just_before_it():
    classifier = FakeAudioClassifier(min_secs=1.5)

    down, _ = await run_test(
        _reader(classifier=classifier, questions=AUDIO_QUESTIONS),
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            _audio(1, 0.5),
            VADUserStoppedSpeakingFrame(),
            VADUserStartedSpeakingFrame(),
            _audio(2, 1),
            VADUserStoppedSpeakingFrame(),
            _transcription("I want to... cancel my order."),
        ],
    )

    assert classifier.clips == [
        ClassifierAudio(audio=_silence(1) + _pcm(1, 0.5), sample_rate=SAMPLE_RATE),
        ClassifierAudio(audio=_pcm(1, 0.5) + _pcm(2, 1), sample_rate=SAMPLE_RATE),
    ]
    assert _perception(down) == [[PerceptionSignal("hesitation", value=0.64)]]


@pytest.mark.asyncio
async def test_a_short_turn_at_the_start_of_a_call_is_heard_after_silence():
    classifier = FakeAudioClassifier(min_secs=5.0)

    down, _ = await run_test(
        _reader(classifier=classifier, questions=AUDIO_QUESTIONS),
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            _audio(1, 1),
            VADUserStoppedSpeakingFrame(),
            _transcription("Fine."),
        ],
    )

    assert classifier.clips == [
        ClassifierAudio(audio=_silence(4) + _pcm(1, 1), sample_rate=SAMPLE_RATE)
    ]
    assert _perception(down) == [[PerceptionSignal("hesitation", value=0.64)]]


def _two_turns(first: InputAudioRawFrame, second: InputAudioRawFrame) -> list:
    """Two user turns, the first read before the second, with the second's transcription."""
    return [
        VADUserStartedSpeakingFrame(),
        first,
        VADUserStoppedSpeakingFrame(),
        SleepFrame(sleep=0.1),
        UserStoppedSpeakingFrame(),
        VADUserStartedSpeakingFrame(),
        second,
        VADUserStoppedSpeakingFrame(),
        _transcription("No!"),
    ]


@pytest.mark.asyncio
async def test_a_short_turn_is_heard_with_the_users_speech_before_it():
    classifier = FakeAudioClassifier(probabilities=[0.2, 0.9])

    down, _ = await run_test(
        _reader(classifier=classifier, questions=AUDIO_QUESTIONS),
        frames_to_send=_two_turns(_audio(1, 2), _audio(2, 0.2)),
    )

    assert classifier.clips[1] == ClassifierAudio(
        audio=_pcm(1, 0.8) + _pcm(2, 0.2), sample_rate=SAMPLE_RATE
    )
    assert _perception(down) == [[PerceptionSignal("hesitation", value=0.9)]]


@pytest.mark.asyncio
async def test_a_new_turn_doesnt_get_the_last_turns_reading():
    down, _ = await run_test(
        _reader(
            classifier=FakeAudioClassifier(delays=[0.0, 1.0], probabilities=[0.2, 0.9]),
            questions=AUDIO_QUESTIONS,
            timeout=0.05,
        ),
        frames_to_send=_two_turns(_audio(1, 2), _audio(2, 0.2)),
    )

    assert _perception(down) == [None]


@pytest.mark.asyncio
async def test_the_audio_between_the_users_speech_is_kept():
    classifier = FakeAudioClassifier(min_secs=2.5)

    await run_test(
        _reader(classifier=classifier, questions=AUDIO_QUESTIONS),
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            _audio(1, 1),
            VADUserStoppedSpeakingFrame(),
            _audio(9, 3),
            VADUserStartedSpeakingFrame(),
            _audio(2, 1),
            VADUserStoppedSpeakingFrame(),
            _transcription("Hello?"),
        ],
    )

    # The reply is heard after the quiet before it, as it was heard.
    assert classifier.clips[1] == ClassifierAudio(
        audio=_pcm(9, 1.5) + _pcm(2, 1), sample_rate=SAMPLE_RATE
    )


@pytest.mark.asyncio
async def test_audio_heard_while_the_user_is_muted_is_heard_as_silence():
    classifier = FakeAudioClassifier(min_secs=2.0)

    await run_test(
        _reader(classifier=classifier, questions=AUDIO_QUESTIONS),
        frames_to_send=[
            _audio(1, 1),
            UserMuteStartedFrame(),
            _audio(2, 0.5),
            UserMuteStoppedFrame(),
            VADUserStartedSpeakingFrame(),
            _audio(3, 0.5),
            VADUserStoppedSpeakingFrame(),
            _transcription("Hello?"),
        ],
    )

    assert classifier.clips == [
        ClassifierAudio(audio=_pcm(1, 1) + _silence(0.5) + _pcm(3, 0.5), sample_rate=SAMPLE_RATE)
    ]


@pytest.mark.asyncio
async def test_speech_while_the_user_is_muted_isnt_read():
    classifier = FakeAudioClassifier()

    down, _ = await run_test(
        _reader(classifier=classifier, questions=AUDIO_QUESTIONS),
        frames_to_send=[
            UserMuteStartedFrame(),
            VADUserStartedSpeakingFrame(),
            _audio(1, 2),
            VADUserStoppedSpeakingFrame(),
            _transcription("Hello?"),
        ],
    )

    assert classifier.clips == []
    assert _perception(down) == [None]


@pytest.mark.asyncio
async def test_a_late_audio_reading_lets_the_transcription_pass_unread():
    down, up = await run_test(
        _reader(
            classifier=FakeAudioClassifier(delays=[1.0]), questions=AUDIO_QUESTIONS, timeout=0.05
        ),
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            _audio(1, 2),
            VADUserStoppedSpeakingFrame(),
            _transcription("Hello."),
        ],
    )

    assert _perception(down) == [None]
    assert not [f for f in up if isinstance(f, ErrorFrame)]


def _two_pauses(first: InputAudioRawFrame, second: InputAudioRawFrame) -> list:
    """A turn with a pause after each stretch of speech, the first read before the second."""
    return [
        VADUserStartedSpeakingFrame(),
        first,
        VADUserStoppedSpeakingFrame(),
        SleepFrame(sleep=0.1),
        VADUserStartedSpeakingFrame(),
        second,
        VADUserStoppedSpeakingFrame(),
        _transcription("I already told you. No!"),
    ]


@pytest.mark.asyncio
async def test_each_pause_sends_the_audio_since_the_turns_last_clip():
    classifier = FakeAudioClassifier()

    await run_test(
        _reader(classifier=classifier, questions=AUDIO_QUESTIONS),
        frames_to_send=_two_pauses(_audio(1, 2), _audio(2, 3)),
    )

    assert classifier.clips == [
        ClassifierAudio(audio=_pcm(1, 2), sample_rate=SAMPLE_RATE),
        ClassifierAudio(audio=_pcm(2, 3), sample_rate=SAMPLE_RATE),
    ]


@pytest.mark.asyncio
async def test_a_short_last_stretch_reaches_back_into_the_turn():
    classifier = FakeAudioClassifier(min_secs=2.0)

    await run_test(
        _reader(classifier=classifier, questions=AUDIO_QUESTIONS),
        frames_to_send=_two_pauses(_audio(1, 3), _audio(2, 0.5)),
    )

    assert classifier.clips[1] == ClassifierAudio(
        audio=_pcm(1, 1.5) + _pcm(2, 0.5), sample_rate=SAMPLE_RATE
    )


@pytest.mark.asyncio
async def test_a_transcription_gets_the_answers_about_the_turns_latest_clip():
    down, _ = await run_test(
        _reader(
            classifier=FakeAudioClassifier(probabilities=[0.2, 0.9]), questions=AUDIO_QUESTIONS
        ),
        frames_to_send=_two_pauses(_audio(1, 2), _audio(2, 2)),
    )

    assert _perception(down) == [[PerceptionSignal("hesitation", value=0.9)]]


@pytest.mark.asyncio
async def test_a_late_clip_leaves_its_speech_unread():
    down, _ = await run_test(
        _reader(
            classifier=FakeAudioClassifier(delays=[0.0, 1.0], probabilities=[0.2, 0.9]),
            questions=AUDIO_QUESTIONS,
            timeout=0.05,
        ),
        frames_to_send=_two_pauses(_audio(1, 2), _audio(2, 2)),
    )

    # The earlier clip's reading is the earlier speech's, not this one's.
    assert _perception(down) == [None]


@pytest.mark.asyncio
async def test_each_stop_is_read_even_while_an_earlier_clip_is_being_read():
    classifier = FakeAudioClassifier(delays=[0.3])

    await run_test(
        _reader(classifier=classifier, questions=AUDIO_QUESTIONS),
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            _audio(1, 2),
            VADUserStoppedSpeakingFrame(),
            VADUserStartedSpeakingFrame(),
            _audio(2, 2),
            VADUserStoppedSpeakingFrame(),
            _transcription("Hello?"),
        ],
    )

    assert classifier.clips == [
        ClassifierAudio(audio=_pcm(1, 2), sample_rate=SAMPLE_RATE),
        ClassifierAudio(audio=_pcm(2, 2), sample_rate=SAMPLE_RATE),
    ]


@pytest.mark.asyncio
async def test_a_stop_during_an_earlier_read_gets_its_own_clips_reading():
    down, _ = await run_test(
        _reader(
            classifier=FakeAudioClassifier(delays=[0.5, 0.0], probabilities=[0.2, 0.9]),
            questions=AUDIO_QUESTIONS,
        ),
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            _audio(1, 2),
            VADUserStoppedSpeakingFrame(),
            VADUserStartedSpeakingFrame(),
            _audio(2, 2),
            VADUserStoppedSpeakingFrame(),
            _transcription("I already told you. No!"),
            SleepFrame(sleep=0.6),
        ],
    )

    assert _perception(down) == [[PerceptionSignal("hesitation", value=0.9)]]


@pytest.mark.asyncio
async def test_a_transcription_of_no_new_speech_gets_no_reading():
    down, _ = await run_test(
        _reader(classifier=FakeAudioClassifier(), questions=AUDIO_QUESTIONS),
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            _audio(1, 2),
            VADUserStoppedSpeakingFrame(),
            _transcription("Can you check my order?"),
            SleepFrame(sleep=0.1),
            # A soft reply the VAD missed: the earlier speech's reading isn't its.
            _transcription("Yes."),
        ],
    )

    assert _perception(down) == [[PerceptionSignal("hesitation", value=0.64)], None]


def _readings(frames) -> list:
    return [f.perception for f in frames if isinstance(f, PerceptionFrame)]


@pytest.mark.asyncio
async def test_a_text_reading_is_broadcast_both_ways():
    down, up = await run_test(
        _reader(
            classifier=FakeClassifier(), questions={"hesitation": AUDIO_QUESTIONS["hesitation"]}
        ),
        frames_to_send=[_transcription("Hello.")],
    )

    reading = [PerceptionSignal("hesitation", value=0.64)]
    assert _readings(down) == [reading]
    assert _readings(up) == [reading]


@pytest.mark.asyncio
async def test_every_clip_reading_is_broadcast():
    down, up = await run_test(
        _reader(
            classifier=FakeAudioClassifier(probabilities=[0.2, 0.9]), questions=AUDIO_QUESTIONS
        ),
        frames_to_send=_two_pauses(_audio(1, 2), _audio(2, 2)),
    )

    readings = [
        [PerceptionSignal("hesitation", value=0.2)],
        [PerceptionSignal("hesitation", value=0.9)],
    ]
    assert _readings(down) == readings
    assert _readings(up) == readings


@pytest.mark.asyncio
async def test_a_transcription_waits_for_the_user_to_stop_speaking():
    down, _ = await run_test(
        _reader(classifier=FakeAudioClassifier(), questions=AUDIO_QUESTIONS),
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            _audio(1, 2),
            _transcription("I already told you."),
            SleepFrame(sleep=0.1),
            VADUserStoppedSpeakingFrame(),
        ],
    )

    assert _perception(down) == [[PerceptionSignal("hesitation", value=0.64)]]


@pytest.mark.asyncio
async def test_the_timeout_counts_from_when_the_user_stops_speaking():
    down, _ = await run_test(
        _reader(classifier=FakeAudioClassifier(), questions=AUDIO_QUESTIONS, timeout=0.2),
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            _audio(1, 2),
            _transcription("I already told you."),
            SleepFrame(sleep=0.5),
            VADUserStoppedSpeakingFrame(),
        ],
    )

    assert _perception(down) == [[PerceptionSignal("hesitation", value=0.64)]]


@pytest.mark.parametrize("stop_secs, read", [(0.0, True), (0.3, False)])
@pytest.mark.asyncio
async def test_the_timeout_counts_from_stop_secs_before_the_vad_stop(stop_secs, read):
    down, _ = await run_test(
        _reader(
            classifier=FakeAudioClassifier(delays=[0.3]), questions=AUDIO_QUESTIONS, timeout=0.45
        ),
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            _audio(1, 2),
            VADUserStoppedSpeakingFrame(stop_secs=stop_secs),
            _transcription("I already told you."),
            SleepFrame(sleep=0.6),
        ],
    )

    reading = [PerceptionSignal("hesitation", value=0.64)] if read else None
    assert _perception(down) == [reading]


@pytest.mark.asyncio
async def test_transcriptions_of_the_same_speech_share_one_deadline():
    down, _ = await run_test(
        _reader(
            classifier=FakeAudioClassifier(delays=[0.3]), questions=AUDIO_QUESTIONS, timeout=0.2
        ),
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            _audio(1, 2),
            _transcription("Just perfect."),
            _transcription("Exactly what I needed today."),
            SleepFrame(sleep=0.05),
            VADUserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.6),
        ],
    )

    # The second waits no longer than the first, so neither gets the late clip.
    assert _perception(down) == [None, None]


@pytest.mark.asyncio
async def test_a_transcription_stops_waiting_for_a_user_who_keeps_speaking(monkeypatch):
    monkeypatch.setattr(reader_module, "_SPEECH_END_WAIT_SECS", 0.1)

    down, _ = await run_test(
        _reader(classifier=FakeAudioClassifier(), questions=AUDIO_QUESTIONS),
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            _audio(1, 2),
            _transcription("I already told you."),
            SleepFrame(sleep=0.5),
            VADUserStoppedSpeakingFrame(),
        ],
    )

    assert _perception(down) == [None]


@pytest.mark.asyncio
async def test_a_user_who_keeps_speaking_doesnt_get_their_earlier_speechs_reading(monkeypatch):
    monkeypatch.setattr(reader_module, "_SPEECH_END_WAIT_SECS", 0.1)

    down, _ = await run_test(
        _reader(classifier=FakeAudioClassifier(), questions=AUDIO_QUESTIONS),
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            _audio(1, 2),
            VADUserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.1),
            VADUserStartedSpeakingFrame(),
            _audio(2, 2),
            _transcription("And another thing."),
            SleepFrame(sleep=0.5),
            VADUserStoppedSpeakingFrame(),
        ],
    )

    assert _perception(down) == [None]


@pytest.mark.asyncio
async def test_a_clip_read_after_its_turn_is_still_broadcast():
    down, _ = await run_test(
        _reader(
            classifier=FakeAudioClassifier(delays=[0.3]), questions=AUDIO_QUESTIONS, timeout=0.05
        ),
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            _audio(1, 2),
            VADUserStoppedSpeakingFrame(),
            _transcription("Hello."),
            UserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.5),
        ],
    )

    assert _perception(down) == [None]
    assert _readings(down) == [[PerceptionSignal("hesitation", value=0.64)]]


@pytest.mark.asyncio
async def test_an_interruption_doesnt_drop_a_transcription_waiting_for_its_reading():
    down, _ = await run_test(
        _reader(classifier=FakeAudioClassifier(delays=[0.2]), questions=AUDIO_QUESTIONS),
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            _audio(1, 2),
            VADUserStoppedSpeakingFrame(),
            _transcription("I've been on hold for an hour."),
            SleepFrame(sleep=0.05),
            InterruptionFrame(),
            SleepFrame(sleep=0.4),
        ],
    )

    assert _perception(down) == [[PerceptionSignal("hesitation", value=0.64)]]


@pytest.mark.asyncio
async def test_only_the_last_transcription_of_the_users_speech_is_finalized():
    down, _ = await run_test(
        _reader(classifier=FakeAudioClassifier(), questions=AUDIO_QUESTIONS),
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            _audio(1, 2),
            _transcription("Just perfect.", finalized=True),
            _transcription("Exactly what I needed today."),
            SleepFrame(sleep=0.05),
            VADUserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.1),
        ],
    )

    assert _finalized(down) == [False, True]


@pytest.mark.asyncio
async def test_a_transcription_sent_within_the_stts_latency_goes_with_the_rest():
    down, _ = await run_test(
        _reader(classifier=FakeAudioClassifier(), questions=AUDIO_QUESTIONS),
        frames_to_send=[
            STTMetadataFrame(service_name="stt", ttfs_p99_latency=0.3),
            VADUserStartedSpeakingFrame(),
            _audio(1, 2),
            _transcription("This is ridiculous."),
            VADUserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.15),
            _transcription("Nobody can tell me anything."),
            SleepFrame(sleep=0.4),
        ],
    )

    assert _finalized(down) == [False, True]


@pytest.mark.asyncio
async def test_no_transcription_is_finalized_when_the_user_speaks_again_before_the_reading():
    down, _ = await run_test(
        _reader(classifier=FakeAudioClassifier(delays=[0.2]), questions=AUDIO_QUESTIONS),
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            _audio(1, 2),
            VADUserStoppedSpeakingFrame(),
            _transcription("This is ridiculous."),
            SleepFrame(sleep=0.05),
            VADUserStartedSpeakingFrame(),
            SleepFrame(sleep=0.3),
            VADUserStoppedSpeakingFrame(),
        ],
    )

    assert _finalized(down) == [False]


@pytest.mark.asyncio
async def test_an_audio_reader_adds_its_wait_to_the_stt_latency_it_passes_on():
    down, _ = await run_test(
        _reader(classifier=FakeAudioClassifier(), questions=AUDIO_QUESTIONS, timeout=0.5),
        frames_to_send=[STTMetadataFrame(service_name="stt", ttfs_p99_latency=0.3)],
    )

    assert _stt_latencies(down) == [pytest.approx(0.8)]


@pytest.mark.asyncio
async def test_an_unusable_audio_reader_passes_the_stt_latency_on_as_it_is():
    down, _ = await run_test(
        _reader(
            classifier=FakeAudioClassifier(setup_error=ClassifierError("no key")),
            questions=AUDIO_QUESTIONS,
        ),
        frames_to_send=[STTMetadataFrame(service_name="stt", ttfs_p99_latency=0.3)],
    )

    assert _stt_latencies(down) == [pytest.approx(0.3)]


TONE = PerceptionSignal("tone", labels=[("frustrated", 0.7), ("confused", 0.2)])
HESITATION = PerceptionSignal("hesitation", value=0.64)
ENERGY = PerceptionSignal("energy", value=0.68)


def _source(
    classifier: BaseClassifier, *names: str, timeout: float | None = None
) -> PerceptionSource:
    """A source that asks ``classifier`` the questions with these names."""
    return PerceptionSource(
        classifier=classifier, questions={name: QUESTIONS[name] for name in names}, timeout=timeout
    )


def _speech(text: str) -> list:
    """Two seconds of the user's speech, then its transcription."""
    return [
        VADUserStartedSpeakingFrame(),
        _audio(1, 2),
        VADUserStoppedSpeakingFrame(),
        _transcription(text),
    ]


class MeetingClassifier(FakeClassifier):
    """Answers once another classifier has been asked too, so only when both are asked at once."""

    def __init__(self):
        super().__init__()
        self.asked_now = asyncio.Event()
        self.other: MeetingClassifier | None = None

    async def _ask(self, state, questions, inputs=()):
        self.asked_now.set()
        assert self.other
        await self.other.asked_now.wait()
        return await super()._ask(state, questions, inputs)


@pytest.mark.asyncio
async def test_the_sources_signals_are_declared_together():
    reader = PerceptionReader(
        sources=[
            _source(FakeClassifier(), "tone"),
            _source(FakeAudioClassifier(), "hesitation", "energy"),
        ]
    )

    down, _ = await run_test(reader, frames_to_send=[])

    metadata = [f for f in down if isinstance(f, PerceptionMetadataFrame)]
    assert [[t.name for t in f.perception.signals] for f in metadata] == [
        ["tone", "hesitation", "energy"]
    ]
    assert list(reader.questions) == ["tone", "hesitation", "energy"]


@pytest.mark.asyncio
async def test_text_sources_write_one_reading_in_their_order():
    down, _ = await run_test(
        PerceptionReader(
            sources=[
                _source(FakeClassifier(), "hesitation"),
                _source(FakeClassifier(), "tone", "energy"),
            ]
        ),
        frames_to_send=[_transcription("I already gave you my order number.")],
    )

    assert _perception(down) == [[HESITATION, TONE, ENERGY]]


@pytest.mark.asyncio
async def test_text_sources_are_asked_at_the_same_time():
    first, second = MeetingClassifier(), MeetingClassifier()
    first.other, second.other = second, first

    down, _ = await run_test(
        PerceptionReader(sources=[_source(first, "tone"), _source(second, "hesitation")]),
        frames_to_send=[_transcription("Hello.")],
    )

    assert _perception(down) == [[TONE, HESITATION]]


@pytest.mark.asyncio
async def test_a_source_reads_the_words_while_another_hears_them():
    words, voice = FakeClassifier(), FakeAudioClassifier()

    down, _ = await run_test(
        PerceptionReader(sources=[_source(words, "tone"), _source(voice, "hesitation")]),
        frames_to_send=[
            LLMContextAssistantTurnFrame(text="What's your order number?", timestamp="now"),
            SleepFrame(sleep=0.05),
            *_speech("I already gave it to you."),
        ],
    )

    assert [state for state, _ in words.asked] == [
        {"assistant": "What's your order number?", "user": "I already gave it to you."}
    ]
    assert voice.clips == [ClassifierAudio(audio=_pcm(1, 2), sample_rate=SAMPLE_RATE)]
    assert _perception(down) == [[TONE, HESITATION]]


@pytest.mark.asyncio
async def test_a_reading_follows_the_sources_order_not_the_order_they_read_in():
    down, _ = await run_test(
        PerceptionReader(
            sources=[
                _source(FakeAudioClassifier(delays=[0.2]), "hesitation"),
                _source(FakeClassifier(), "tone"),
            ]
        ),
        frames_to_send=_speech("I already told you."),
    )

    assert _perception(down) == [[HESITATION, TONE]]


@pytest.mark.asyncio
async def test_a_held_transcription_waits_for_its_words_to_be_read():
    down, _ = await run_test(
        PerceptionReader(
            sources=[
                _source(FakeClassifier(delay=0.2), "tone"),
                _source(FakeAudioClassifier(), "hesitation"),
            ]
        ),
        frames_to_send=_speech("I already told you."),
    )

    assert _perception(down) == [[TONE, HESITATION]]


@pytest.mark.asyncio
async def test_each_transcription_of_a_speech_gets_its_own_words_reading_and_the_clips():
    words = FakeClassifier()

    down, _ = await run_test(
        PerceptionReader(
            sources=[_source(words, "tone"), _source(FakeAudioClassifier(), "hesitation")]
        ),
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            _audio(1, 2),
            _transcription("I already told you."),
            _transcription("Twice."),
            VADUserStoppedSpeakingFrame(),
        ],
    )

    assert [state for state, _ in words.asked] == [
        {"user": "I already told you."},
        {"user": "Twice."},
    ]
    assert _perception(down) == [[TONE, HESITATION], [TONE, HESITATION]]


@pytest.mark.asyncio
async def test_a_late_source_leaves_only_its_signals_out():
    down, up = await run_test(
        PerceptionReader(
            sources=[
                _source(FakeClassifier(delay=1.0), "tone", timeout=0.05),
                _source(FakeClassifier(), "hesitation"),
            ]
        ),
        frames_to_send=[_transcription("Hello.")],
    )

    assert _perception(down) == [[HESITATION]]
    assert not [f for f in up if isinstance(f, ErrorFrame)]


@pytest.mark.asyncio
async def test_a_failed_source_is_reported_and_leaves_only_its_signals_out():
    down, up = await run_test(
        PerceptionReader(
            sources=[
                _source(FakeClassifier(error=ClassifierError("down")), "tone"),
                _source(FakeClassifier(), "hesitation"),
            ]
        ),
        frames_to_send=[_transcription("Hello.")],
    )

    assert _perception(down) == [[HESITATION]]
    assert [f for f in up if isinstance(f, ErrorFrame)]


@pytest.mark.asyncio
async def test_a_late_clip_reading_leaves_the_words_reading():
    down, _ = await run_test(
        PerceptionReader(
            sources=[
                _source(FakeClassifier(), "tone"),
                _source(FakeAudioClassifier(delays=[1.0]), "hesitation", timeout=0.05),
            ]
        ),
        frames_to_send=_speech("Hello."),
    )

    assert _perception(down) == [[TONE]]


@pytest.mark.asyncio
async def test_a_late_words_reading_leaves_the_clips_reading():
    down, _ = await run_test(
        PerceptionReader(
            sources=[
                _source(FakeClassifier(delay=1.0), "tone", timeout=0.05),
                _source(FakeAudioClassifier(), "hesitation"),
            ]
        ),
        frames_to_send=_speech("Hello."),
    )

    assert _perception(down) == [[HESITATION]]


@pytest.mark.asyncio
async def test_a_transcription_no_source_read_passes_unread():
    down, _ = await run_test(
        PerceptionReader(
            sources=[
                _source(FakeClassifier(delay=1.0), "tone", timeout=0.05),
                _source(FakeAudioClassifier(delays=[1.0]), "hesitation", timeout=0.05),
            ]
        ),
        frames_to_send=_speech("Hello."),
    )

    assert _perception(down) == [None]


@pytest.mark.asyncio
async def test_each_audio_source_hears_a_clip_of_the_length_its_classifier_takes():
    whole, short, long = (
        FakeAudioClassifier(min_secs=0.5),
        FakeAudioClassifier(min_secs=0.5, max_secs=1.0),
        FakeAudioClassifier(min_secs=3.0),
    )

    await run_test(
        PerceptionReader(
            sources=[
                _source(whole, "tone"),
                _source(short, "hesitation"),
                _source(long, "energy"),
            ]
        ),
        frames_to_send=[
            _audio(1, 2),
            VADUserStartedSpeakingFrame(),
            _audio(2, 1),
            _audio(3, 1),
            VADUserStoppedSpeakingFrame(),
            _transcription("I already told you."),
        ],
    )

    # The speech, its last second, and the speech with the second before it.
    assert whole.clips == [ClassifierAudio(audio=_pcm(2, 1) + _pcm(3, 1), sample_rate=SAMPLE_RATE)]
    assert short.clips == [ClassifierAudio(audio=_pcm(3, 1), sample_rate=SAMPLE_RATE)]
    assert long.clips == [
        ClassifierAudio(audio=_pcm(1, 1) + _pcm(2, 1) + _pcm(3, 1), sample_rate=SAMPLE_RATE)
    ]


@pytest.mark.asyncio
async def test_each_sources_reading_is_broadcast_as_its_own():
    down, up = await run_test(
        PerceptionReader(
            sources=[
                _source(FakeClassifier(), "tone"),
                _source(FakeAudioClassifier(), "hesitation"),
            ]
        ),
        frames_to_send=_speech("Hello."),
    )

    for readings in (_readings(down), _readings(up)):
        assert sorted(readings, key=lambda reading: reading[0].name) == [[HESITATION], [TONE]]


@pytest.mark.asyncio
async def test_every_classifiers_metrics_are_pushed():
    words, voice = FakeClassifier(), FakeAudioClassifier()

    down, _ = await run_test(
        PerceptionReader(sources=[_source(words, "tone"), _source(voice, "hesitation")]),
        frames_to_send=[*_speech("Hello."), SleepFrame(sleep=0.2)],
        pipeline_params=PipelineParams(enable_metrics=True),
    )

    measured = {d.processor for f in down if isinstance(f, MetricsFrame) for d in f.data}
    assert {words.name, voice.name} <= measured


@pytest.mark.parametrize("words_timeout, expected", [(0.1, 1.0), (0.9, 1.2)])
@pytest.mark.asyncio
async def test_a_reader_that_hears_adds_its_longest_wait_to_the_stt_latency(
    words_timeout, expected
):
    down, _ = await run_test(
        PerceptionReader(
            sources=[
                _source(FakeClassifier(), "tone", timeout=words_timeout),
                _source(FakeAudioClassifier(), "hesitation", timeout=0.2),
                _source(FakeAudioClassifier(), "energy", timeout=0.7),
            ]
        ),
        frames_to_send=[STTMetadataFrame(service_name="stt", ttfs_p99_latency=0.3)],
    )

    assert _stt_latencies(down) == [pytest.approx(expected)]


@pytest.mark.asyncio
async def test_a_source_that_cant_be_set_up_leaves_the_reader_unusable():
    words = FakeClassifier()
    reader = PerceptionReader(
        sources=[
            _source(words, "tone"),
            _source(FakeAudioClassifier(setup_error=ClassifierError("no key")), "hesitation"),
        ]
    )

    down, up = await run_test(reader, frames_to_send=_speech("Hello."))

    assert not reader.is_usable
    assert [f for f in up if isinstance(f, ErrorFrame)]
    assert not [f for f in down if isinstance(f, PerceptionMetadataFrame)]
    assert _perception(down) == [None]
    assert words.asked == []


def test_two_sources_cant_report_a_signal_of_the_same_name():
    with pytest.raises(ValueError, match="hesitation"):
        PerceptionReader(
            sources=[
                _source(FakeClassifier(), "tone", "hesitation"),
                _source(FakeAudioClassifier(), "hesitation"),
            ]
        )


def test_a_classifier_goes_in_one_source():
    classifier = FakeClassifier()

    with pytest.raises(ValueError):
        PerceptionReader(sources=[_source(classifier, "tone"), _source(classifier, "energy")])


def test_a_reader_needs_a_source():
    with pytest.raises(ValueError):
        PerceptionReader(sources=[])


def test_a_source_takes_its_classifiers_default_timeout():
    assert _source(FakeClassifier(), "tone").timeout == 0.5
    assert _source(FakeAudioClassifier(), "tone").timeout == 2.0
    assert _source(FakeAudioClassifier(), "tone", timeout=0.3).timeout == 0.3


def test_a_sources_name_that_cant_be_written_in_a_reading_is_refused():
    with pytest.raises(ValueError):
        PerceptionSource(
            classifier=FakeClassifier(), questions={"speech rate": QUESTIONS["energy"]}
        )
