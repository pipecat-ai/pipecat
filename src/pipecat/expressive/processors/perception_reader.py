#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The processor that reads how each user turn was said by asking a classifier."""

import asyncio
import re
import time
from collections.abc import Mapping
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from pipecat.audio.speech_segment_buffer import SpeechSegmentBuffer
from pipecat.classifiers.base_classifier import (
    BaseClassifier,
    ChoiceQuestion,
    ChoiceResult,
    ClassifierAudio,
    ClassifierError,
    ClassifierQuestion,
    ClassifierResult,
    ScoreResult,
)
from pipecat.expressive.perception import PerceptionCapabilities, PerceptionSignal, PerceptionTerm
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InputAudioRawFrame,
    LLMContextAssistantTurnFrame,
    MetricsFrame,
    PerceptionFrame,
    PerceptionMetadataFrame,
    StartFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    UserMuteStartedFrame,
    UserMuteStoppedFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import MetricsData
from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessor,
    FrameProcessorSetup,
)

#: A signal's name, as written in a reading: no whitespace, "=", ",", "[", "]", "(" or ")".
_SIGNAL_NAME = re.compile(r"[^\s=,\[\]()]+")

#: The longest a transcription waits for the user to stop speaking, in seconds:
#: as long as the user aggregator keeps a turn open without user activity by
#: default (``user_turn_stop_timeout``).
_SPEECH_END_WAIT_SECS = 5.0


class PerceptionQuestion(BaseModel):
    """One signal a :class:`PerceptionReader` reports, and the question it asks for it.

    Parameters:
        question: What the classifier is asked. A choice is reported as its
            likeliest options, a yes/no as its probability and a score as its
            position on the scale.
        description: A short line telling the LLM how to read the signal,
            such as its range, or empty.
        max_labels: The most options a choice reports, likeliest first.
    """

    question: ClassifierQuestion
    description: str = ""
    max_labels: int = Field(default=2, ge=1)


class PerceptionReader(FrameProcessor):
    """Reads how each user turn was said by asking a classifier.

    Goes between the STT service and the user aggregator. Each final
    transcription gets the classifier's answers as its ``perception``, one
    signal per question, named after it. Every reading is also broadcast in a
    :class:`~pipecat.frames.frames.PerceptionFrame` as soon as it's made, and
    the signals are declared at start in a
    :class:`~pipecat.frames.frames.PerceptionMetadataFrame`.

    A text classifier reads each final transcription, with the bot's reply
    before it. A classifier that can hear, as its ``supports_audio`` says,
    hears a clip of the user's recent speech each time they pause, and the
    transcriptions of that speech wait for its answers, up to ``timeout``
    seconds after the user stopped. A transcription passes unread when its
    reading is late or the classifier fails, and a classifier that can't be set
    up leaves the reader unusable.

    Part of expressive conversation (alpha): turn it on with
    ``PipelineParams(enable_expressive_conversation=True)`` for the LLM to read
    the turns.

    Example::

        reader = PerceptionReader(
            classifier=JevClassifier(api_key=os.getenv("TYPESAFE_API_KEY")),
            questions={
                "frustration": PerceptionQuestion(
                    question=YesNoQuestion(instructions="Is the user frustrated?"),
                    description="The chance the user is frustrated, from 0 to 1.",
                ),
            },
        )

        pipeline = Pipeline([transport.input(), stt, reader, user_aggregator, llm, ...])
    """

    def __init__(
        self,
        *,
        classifier: BaseClassifier,
        questions: Mapping[str, PerceptionQuestion],
        timeout: float | None = None,
        **kwargs,
    ):
        """Initialize the processor.

        Args:
            classifier: What answers the questions about each turn.
            questions: The questions, by the name of the signal each reports.
            timeout: Seconds a transcription waits for its reading. By default
                0.5 for a text classifier, counted from when the transcription
                arrives, and 2 for an audio classifier, counted from when the
                user stops speaking.
            **kwargs: Additional arguments passed to the parent.

        Raises:
            ValueError: If a name can't be written in a reading.
        """
        super().__init__(**kwargs)
        for name in questions:
            if not _SIGNAL_NAME.fullmatch(name):
                raise ValueError(f"{name!r} can't name a signal: it has a space or one of =,[]()")
        # What to ask, and how long a transcription waits for the answers.
        self._classifier = classifier
        self._questions = dict(questions)
        if timeout is None:
            timeout = 2.0 if classifier.supports_audio else 0.5
        self._timeout = timeout
        # The bot's last reply, which the user's turn is read against.
        self._previous: str | None = None
        # For a classifier that hears: the user's recent speech, read a clip at
        # each pause.
        self._speech = _RecentSpeech(classifier)
        self._audio_read: asyncio.Task | None = None
        # Transcriptions held until their speech is read, and the task that
        # pushes them on once the STT's latency has passed.
        self._held: list[TranscriptionFrame] = []
        self._release_task: asyncio.Task | None = None
        self._stt_latency = 0.0
        self._classifier.add_event_handler("on_metrics", self._on_classifier_metrics)

    @property
    def classifier(self) -> BaseClassifier:
        """The classifier that answers the questions."""
        return self._classifier

    @property
    def questions(self) -> dict[str, PerceptionQuestion]:
        """The questions asked about each turn, by the name of the signal each reports."""
        return self._questions

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the processor and its classifier.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        self._speech.set_sample_rate(setup.audio_in_sample_rate)
        try:
            await self._classifier.setup(self.task_manager)
        except ClassifierError as e:
            await self.push_error(f"{self}: {e}", e, force_treat_as_permanent=True)

    async def cleanup(self):
        """Clean up the processor and its classifier."""
        await super().cleanup()
        await self._cancel_read()
        if self._release_task:
            await self.cancel_task(self._release_task)
        await self._classifier.cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Read the user's turns, and keep the bot's replies to read them against.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            await self._handle_audio(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, StartFrame):
            # The declaration follows the StartFrame, which must reach every
            # processor before anything else.
            await self.push_frame(frame, direction)
            await self._declare_signals()
        elif isinstance(frame, EndFrame):
            # The transcriptions held go on before the pipeline ends.
            if self._release_task:
                await self._release_task
            await self.push_frame(frame, direction)
        elif isinstance(frame, TranscriptionFrame) and self._holds(frame):
            # Pushed on with the reading once the user's speech is over.
            await self._hold_transcription(frame)
        elif isinstance(frame, TranscriptionFrame):
            await self._handle_transcription(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, STTMetadataFrame):
            await self._handle_stt_metadata(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            await self._handle_vad_user_started_speaking(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._handle_vad_user_stopped_speaking(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, (UserMuteStartedFrame, UserMuteStoppedFrame)):
            await self._handle_mute(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMContextAssistantTurnFrame):
            await self._handle_assistant_turn(frame)
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _handle_audio(self, frame: InputAudioRawFrame):
        if self._classifier.supports_audio:
            self._speech.append(frame.audio)

    def _holds(self, frame: TranscriptionFrame) -> bool:
        """Whether the transcription waits for an audio reading."""
        return bool(self._classifier.supports_audio and frame.text.strip() and self.is_usable)

    async def _hold_transcription(self, frame: TranscriptionFrame):
        self._held.append(frame)
        if not self._release_task:
            self._release_task = self.create_task(self._release_transcriptions())

    async def _handle_transcription(self, frame: TranscriptionFrame):
        if frame.text.strip() and self.is_usable:
            frame.perception = await self._read(frame.text)

    async def _handle_stt_metadata(self, frame: STTMetadataFrame):
        self._stt_latency = frame.ttfs_p99_latency
        # Transcriptions held for an audio reading reach the user aggregator up
        # to the timeout later than the STT sends them, so the stop strategies
        # wait that much longer.
        if self._classifier.supports_audio and self.is_usable:
            frame.ttfs_p99_latency += self._timeout

    async def _handle_vad_user_started_speaking(self, frame: VADUserStartedSpeakingFrame):
        self._speech.start()

    async def _handle_vad_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        self._speech.stop(frame.stop_secs)
        if self._classifier.supports_audio and self.is_usable:
            self._read_speech()

    async def _handle_mute(self, frame: UserMuteStartedFrame | UserMuteStoppedFrame):
        muted = isinstance(frame, UserMuteStartedFrame)
        self._speech.mute(muted)
        if muted:
            await self._cancel_read()

    async def _handle_assistant_turn(self, frame: LLMContextAssistantTurnFrame):
        self._previous = frame.text

    async def _declare_signals(self):
        """Push the signals the reader reports, if it can report them."""
        if self.is_usable:
            await self.push_frame(
                PerceptionMetadataFrame(service_name=self.name, perception=self._capabilities())
            )

    def _capabilities(self) -> PerceptionCapabilities:
        """The signals the reader reports, from its questions."""
        return PerceptionCapabilities(
            signals=[
                PerceptionTerm(name, question.description, labels=self._labels(question.question))
                for name, question in self.questions.items()
            ]
        )

    def _labels(self, question: ClassifierQuestion) -> list[str]:
        """The labels a signal can report: a choice's options, or none."""
        return list(question.options) if isinstance(question, ChoiceQuestion) else []

    async def _read(self, text: str) -> list[PerceptionSignal] | None:
        """A transcription's reading, or None when it's late or the classifier fails."""
        questions = self.questions
        state = {"assistant": self._previous, "user": text} if self._previous else {"user": text}
        signals = None
        try:
            results = await asyncio.wait_for(
                self._classifier.ask(state, self._classifier_questions(questions)),
                self._timeout,
            )
            signals = self._signals(results, questions)
            await self.broadcast_frame(PerceptionFrame, perception=signals)
        except TimeoutError:
            logger.warning(f"{self}: no reading after {self._timeout}s, the turn passes unread")
        except ClassifierError as e:
            await self.push_error(f"{self}: {e}", e)
        return signals

    def _read_speech(self):
        """Start reading the speech since the last clip, unless a read is still waiting."""
        if self._audio_read and not self._audio_read.done():
            logger.debug(f"{self}: a clip is still being read, the new speech waits for the next")
            return
        clip = self._speech.clip()
        if clip:
            logger.debug(f"{self}: reading {clip.duration_secs:.1f}s of the user's speech")
            state = {"assistant": self._previous} if self._previous else {}
            self._audio_read = self.create_task(self._read_audio(state, clip))
        else:
            logger.debug(f"{self}: too little of the user's speech yet for a clip")
            # This speech has no clip of its own, so it gets no reading.
            self._audio_read = None

    async def _read_audio(
        self, state: dict[str, Any], clip: ClassifierAudio
    ) -> list[PerceptionSignal] | None:
        """A clip's reading, broadcast as soon as it's made, or None when the classifier fails."""
        questions = self.questions
        signals = None
        try:
            results = await self._classifier.ask(
                state, self._classifier_questions(questions), audio=clip
            )
            signals = self._signals(results, questions)
            await self.broadcast_frame(PerceptionFrame, perception=signals)
        except ClassifierError as e:
            await self.push_error(f"{self}: {e}", e)
        return signals

    async def _release_transcriptions(self):
        """Push the held transcriptions on with the reading, in order."""
        try:
            while self._held:
                await self._release_speech()
        finally:
            self._release_task = None

    async def _release_speech(self):
        """Push the transcriptions of the user's speech on together, once it's over and read."""
        speech_end = await self._wait_for_speech_end()
        held, self._held = self._held, []
        reading = await self._clip_reading(speech_end)
        # The stop strategies end the turn at once on a finalized transcription,
        # so only the speech's last one is, and none when the user has spoken
        # again since.
        over = speech_end is not None and self._speech.ended_at == speech_end
        for frame in held:
            frame.perception = reading
            frame.finalized = over and frame is held[-1]
            await self.push_frame(frame)

    async def _wait_for_speech_end(self) -> float | None:
        """When the user stopped speaking, once the STT's latency has passed; None if not yet."""
        if self._speech.speaking:
            logger.debug(f"{self}: transcriptions are waiting for the user to stop speaking")
        speech_end = await self._speech.wait_for_end(_SPEECH_END_WAIT_SECS)
        if speech_end is not None:
            await asyncio.sleep(max(0.0, speech_end + self._stt_latency - time.monotonic()))
        return speech_end

    async def _clip_reading(self, speech_end: float | None) -> list[PerceptionSignal] | None:
        """The reading of the speech's clip, waited for up to the timeout after the user stopped."""
        read = self._audio_read
        if read and not read.done():
            logger.debug(f"{self}: transcriptions are waiting for their clip's reading")
            deadline = (speech_end if speech_end is not None else time.monotonic()) + self._timeout
            # A wait that doesn't cancel the read when it's late, nor fails
            # when a mute cancels it.
            await asyncio.wait({read}, timeout=max(0.0, deadline - time.monotonic()))
            if not read.done():
                logger.warning(
                    f"{self}: the clip wasn't read after {self._timeout}s, the speech passes unread"
                )
        reading = None
        if read and read.done() and not read.cancelled() and read.exception() is None:
            reading = read.result()
        return reading

    async def _cancel_read(self):
        """Stop the read still running, if any."""
        if self._audio_read and not self._audio_read.done():
            logger.debug(f"{self}: dropping a clip still being read")
            await self.cancel_task(self._audio_read)
        self._audio_read = None

    def _classifier_questions(
        self, questions: Mapping[str, PerceptionQuestion]
    ) -> dict[str, ClassifierQuestion]:
        """The questions as the classifier is asked them."""
        return {name: question.question for name, question in questions.items()}

    def _signals(
        self, results: Mapping[str, ClassifierResult], questions: Mapping[str, PerceptionQuestion]
    ) -> list[PerceptionSignal]:
        """The answers as the signals they report, in the questions' order."""
        return [self._signal(name, results[name], questions[name]) for name in questions]

    def _signal(
        self, name: str, result: ClassifierResult, question: PerceptionQuestion
    ) -> PerceptionSignal:
        """One answer as the signal it reports."""
        if isinstance(result, ChoiceResult):
            likeliest = sorted(result.probabilities.items(), key=lambda p: -p[1])
            signal = PerceptionSignal(name, labels=likeliest[: question.max_labels])
        elif isinstance(result, ScoreResult):
            signal = PerceptionSignal(name, value=round(result.score, 2))
        else:
            signal = PerceptionSignal(name, value=round(result.probability, 2))
        return signal

    async def _on_classifier_metrics(self, classifier: BaseClassifier, data: list[MetricsData]):
        """Push the classifier's metrics into the pipeline, when metrics are on."""
        if self.metrics_enabled:
            await self.push_frame(MetricsFrame(data=data))


class _RecentSpeech:
    """The user's recent speech, without the silence between, cut into clips for a classifier.

    Each stretch the VAD reports is kept from a second before it. A clip is the
    speech since the last one, topped up with the speech before it when that's
    shorter than the classifier's ``min_audio_secs``, and only that much of its
    end is kept to top up the next. No speech is kept while the user is muted.
    """

    def __init__(self, classifier: BaseClassifier):
        # How long a clip can be, and the audio's sample rate.
        self._min_secs = classifier.min_audio_secs
        self._max_secs = classifier.max_audio_secs
        self._sample_rate = 0
        # The stretch the VAD is reporting, the speech kept before it, and how
        # much of that no clip has had yet. Nothing is kept while muted.
        self._buffer = SpeechSegmentBuffer(max_secs=self._max_secs)
        self._speech = bytearray()
        self._new_bytes = 0
        self._muted = False
        # Set while the user is silent, and when they stopped speaking.
        self._ended = asyncio.Event()
        self._ended.set()
        self._ended_at: float | None = None

    @property
    def speaking(self) -> bool:
        """Whether the user is speaking."""
        return not self._ended.is_set()

    @property
    def ended_at(self) -> float | None:
        """When the user stopped speaking, or None while they speak and before they ever have."""
        return self._ended_at

    def set_sample_rate(self, sample_rate: int):
        """Set the sample rate of the user's audio."""
        self._sample_rate = sample_rate
        self._buffer.set_sample_rate(sample_rate)

    def append(self, audio: bytes):
        """Add the user's audio, unless they're muted."""
        if not self._muted:
            self._buffer.append(audio)

    def start(self):
        """Start keeping a stretch of the user's speech."""
        self._buffer.start()
        self._ended.clear()
        self._ended_at = None

    def stop(self, stop_secs: float):
        """Keep the stretch the user just finished, stop_secs before the VAD said so."""
        segment = self._buffer.stop()
        self._speech += segment
        self._new_bytes += len(segment)
        # The stop strategies count from when the user stopped, too.
        self._ended_at = time.monotonic() - stop_secs
        self._ended.set()

    def clip(self) -> ClassifierAudio | None:
        """The speech since the last clip, topped up to the minimum, or None when there's too little."""
        min_bytes = self._bytes(self._min_secs)
        size = max(self._new_bytes, min_bytes)
        if self._max_secs is not None:
            size = min(size, self._bytes(self._max_secs))
        clip = None
        if self._new_bytes and len(self._speech) >= size:
            clip = ClassifierAudio(
                audio=bytes(self._speech[len(self._speech) - size :]), sample_rate=self._sample_rate
            )
            self._new_bytes = 0
            # A later clip only ever tops up with the end of this one.
            del self._speech[: len(self._speech) - min_bytes]
        return clip

    def mute(self, muted: bool):
        """Keep no speech while the user is muted, and drop what was kept when they're muted."""
        self._muted = muted
        if muted:
            self._buffer.stop()
            self._speech.clear()
            self._new_bytes = 0
            self._ended_at = time.monotonic()
            self._ended.set()

    async def wait_for_end(self, timeout: float) -> float | None:
        """When the user stopped speaking, waiting up to timeout for it; None if they still speak."""
        if self.speaking:
            try:
                await asyncio.wait_for(self._ended.wait(), timeout)
            except TimeoutError:
                pass
        return self._ended_at

    def _bytes(self, secs: float) -> int:
        """How many bytes of 16-bit mono audio make secs seconds."""
        return int(secs * self._sample_rate) * 2
