#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The processor that reads how each user turn was said by asking classifiers."""

import asyncio
import re
import time
from collections.abc import Coroutine, Mapping, Sequence
from typing import Any, NamedTuple

from loguru import logger
from pydantic import BaseModel, Field

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


class PerceptionSource:
    """One classifier a :class:`PerceptionReader` asks, and the questions it's asked.

    A reader with several sources asks each one about the same turn and writes
    their answers in one reading, so a signal's name belongs to one source. The
    classifier says how its source reads: a text classifier reads each final
    transcription, and one that can hear, as its ``supports_audio`` says, hears
    the user's audio.

    A classifier whose questions aren't known until it's set up gets a subclass
    that overrides :attr:`questions`, and :attr:`names` to name its signals
    before then.

    Example::

        hume = PerceptionSource(
            classifier=HumeAudioClassifier(api_key=os.getenv("HUME_API_KEY")),
            questions={
                "hesitation": PerceptionQuestion(
                    question=YesNoQuestion(instructions="Does the speaker sound unsure?"),
                    description="The chance the user sounded unsure, from 0 to 1.",
                ),
            },
        )
    """

    def __init__(
        self,
        *,
        classifier: BaseClassifier,
        questions: Mapping[str, PerceptionQuestion],
        timeout: float | None = None,
    ):
        """Initialize the source.

        Args:
            classifier: What answers the questions about each turn.
            questions: The questions, by the name of the signal each reports.
            timeout: Seconds a transcription waits for the classifier's
                answers. By default 0.5 for a text classifier, counted from
                when the transcription arrives, and 2 for an audio classifier,
                counted from when the user stops speaking.

        Raises:
            ValueError: If a name can't be written in a reading.
        """
        for name in questions:
            if not _SIGNAL_NAME.fullmatch(name):
                raise ValueError(f"{name!r} can't name a signal: it has a space or one of =,[]()")
        self._classifier = classifier
        self._questions = dict(questions)
        if timeout is None:
            timeout = 2.0 if classifier.supports_audio else 0.5
        self._timeout = timeout

    @property
    def classifier(self) -> BaseClassifier:
        """The classifier that answers the questions."""
        return self._classifier

    @property
    def questions(self) -> dict[str, PerceptionQuestion]:
        """The questions asked about each turn, by the name of the signal each reports."""
        return self._questions

    @property
    def names(self) -> list[str]:
        """The names of the signals the source reports, known before it's set up."""
        return list(self.questions)

    @property
    def timeout(self) -> float:
        """Seconds a transcription waits for the classifier's answers."""
        return self._timeout


class PerceptionReader(FrameProcessor):
    """Reads how each user turn was said by asking classifiers.

    Goes between the STT service and the user aggregator. Each final
    transcription gets the classifiers' answers as its ``perception``, one
    signal per question, named after it. Every classifier's reading is also
    broadcast in a :class:`~pipecat.frames.frames.PerceptionFrame` as soon as
    it's made, and the signals are declared at start in a
    :class:`~pipecat.frames.frames.PerceptionMetadataFrame`.

    A text classifier reads each final transcription, with the bot's reply
    before it. A classifier that can hear, as its ``supports_audio`` says,
    hears the user's audio each time they stop speaking: their speech, and the
    audio heard just before it when that's shorter than the classifier takes.
    The transcriptions of that speech wait for its answers, up to its source's
    ``timeout`` after the user stopped.

    A reader asks each of its ``sources``, a classifier with its own questions,
    about the same turn, all at once, and their signals go in the turn's one
    reading, in the sources' order. A transcription goes on once every source
    has read it or run out of time. A source that's late or whose classifier
    fails leaves its signals out, and a transcription none of them read passes
    unread. A classifier that can't be set up leaves the reader unusable.

    Part of expressive conversation (alpha): turn it on with
    ``PipelineParams(enable_expressive_conversation=True)`` for the LLM to read
    the turns.

    Example::

        reader = PerceptionReader(
            sources=[
                PerceptionSource(
                    classifier=JevClassifier(api_key=os.getenv("TYPESAFE_API_KEY")),
                    questions={
                        "frustration": PerceptionQuestion(
                            question=YesNoQuestion(instructions="Is the user frustrated?"),
                            description="The chance the user is frustrated, from 0 to 1.",
                        ),
                    },
                ),
            ],
        )

        pipeline = Pipeline([transport.input(), stt, reader, user_aggregator, llm, ...])

    A reader that reads each turn's words and hears its audio::

        reader = PerceptionReader(
            sources=[
                PipecatToneSource(),
                PerceptionSource(
                    classifier=HumeAudioClassifier(api_key=os.getenv("HUME_API_KEY")),
                    questions={
                        "hesitation": PerceptionQuestion(
                            question=YesNoQuestion(instructions="Does the speaker sound unsure?"),
                            description="The chance the user sounded unsure, from 0 to 1.",
                        ),
                    },
                ),
            ],
        )
    """

    def __init__(self, *, sources: Sequence[PerceptionSource], **kwargs):
        """Initialize the processor.

        Args:
            sources: The classifiers to ask, each with its questions and its
                timeout.
            **kwargs: Additional arguments passed to the parent.

        Raises:
            ValueError: If there are no sources, or if two report the same
                signal or ask the same classifier.
        """
        super().__init__(**kwargs)
        if not sources:
            raise ValueError(f"{self} needs at least one source")
        if len({id(source.classifier) for source in sources}) < len(sources):
            raise ValueError(f"{self}: a classifier goes in one source, with all its questions")
        # Who to ask: the sources that read transcriptions, and the ones that hear.
        self._sources = list(sources)
        self._text_sources = [s for s in self._sources if not s.classifier.supports_audio]
        self._audio_sources = [s for s in self._sources if s.classifier.supports_audio]
        if clash := self._name_clash():
            raise ValueError(clash)
        # The bot's last reply, which the user's turn is read against.
        self._previous: str | None = None
        # For the sources that hear: the user's recent audio, and their reads
        # of the clips that end with the user's latest speech.
        self._audio = _RecentAudio([source.classifier for source in self._audio_sources])
        self._clip_reads: _ClipReads | None = None
        # Transcriptions held until their speech is read, and the task that
        # pushes them on once the STT's latency has passed.
        self._held: list[_HeldTranscription] = []
        self._release_task: asyncio.Task | None = None
        self._stt_latency = 0.0
        # Every read still running.
        self._reads: set[asyncio.Task] = set()
        for source in self._sources:
            source.classifier.add_event_handler("on_metrics", self._on_classifier_metrics)

    @property
    def sources(self) -> list[PerceptionSource]:
        """The classifiers asked about each turn, each with its questions."""
        return list(self._sources)

    @property
    def questions(self) -> dict[str, PerceptionQuestion]:
        """The questions asked about each turn, by the name of the signal each reports."""
        return {
            name: question
            for source in self._sources
            for name, question in source.questions.items()
        }

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the processor and its classifiers.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        self._audio.set_sample_rate(setup.audio_in_sample_rate)
        try:
            for source in self._sources:
                await source.classifier.setup(self.task_manager)
        except ClassifierError as e:
            await self.push_error(f"{self}: {e}", e, force_treat_as_permanent=True)

    async def cleanup(self):
        """Clean up the processor and its classifiers."""
        await super().cleanup()
        if self._release_task:
            await self.cancel_task(self._release_task)
        for read in list(self._reads):
            await self.cancel_task(read)
        self._clip_reads = None
        for source in self._sources:
            await source.classifier.cleanup()

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
        if self._audio_sources:
            self._audio.append(frame.audio)

    def _holds(self, frame: TranscriptionFrame) -> bool:
        """Whether the transcription waits for an audio reading."""
        return bool(self._audio_sources and frame.text.strip() and self.is_usable)

    async def _hold_transcription(self, frame: TranscriptionFrame):
        # The text sources read it while it waits for its speech to be heard.
        reads = {
            source: self._start_read(self._read_text(source, frame.text))
            for source in self._text_sources
        }
        self._held.append(_HeldTranscription(frame, reads))
        if not self._release_task:
            self._release_task = self.create_task(self._release_transcriptions())

    async def _handle_transcription(self, frame: TranscriptionFrame):
        if frame.text.strip() and self.is_usable:
            readings = await asyncio.gather(
                *(self._read_text(source, frame.text) for source in self._text_sources)
            )
            frame.perception = self._reading(dict(zip(self._text_sources, readings)))

    async def _handle_stt_metadata(self, frame: STTMetadataFrame):
        self._stt_latency = frame.ttfs_p99_latency
        # Transcriptions held for an audio reading reach the user aggregator up
        # to the longest timeout later than the STT sends them, so the stop
        # strategies wait that much longer.
        if self._audio_sources and self.is_usable:
            frame.ttfs_p99_latency += max(source.timeout for source in self._sources)

    async def _handle_vad_user_started_speaking(self, frame: VADUserStartedSpeakingFrame):
        self._audio.start(frame.start_secs)

    async def _handle_vad_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        self._audio.stop(frame.stop_secs)
        if self._audio_sources and self.is_usable:
            self._read_speech()

    async def _handle_mute(self, frame: UserMuteStartedFrame | UserMuteStoppedFrame):
        muted = isinstance(frame, UserMuteStartedFrame)
        self._audio.mute(muted)
        if muted:
            await self._cancel_clip_reads()

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

    async def _read_text(
        self, source: PerceptionSource, text: str
    ) -> list[PerceptionSignal] | None:
        """A source's reading of a transcription, or None when it's late or its classifier fails."""
        questions = source.questions
        state = {"assistant": self._previous, "user": text} if self._previous else {"user": text}
        signals = None
        try:
            results = await asyncio.wait_for(
                source.classifier.ask(state, self._classifier_questions(questions)),
                source.timeout,
            )
            signals = self._signals(results, questions)
            await self.broadcast_frame(PerceptionFrame, perception=signals)
        except TimeoutError:
            logger.warning(
                f"{self}: no reading from {source.classifier} after {source.timeout}s, "
                "the turn goes on without it"
            )
        except ClassifierError as e:
            await self.push_error(f"{self}: {e}", e)
        return signals

    def _read_speech(self):
        """Start each audio source's read of the clip that ends with the speech just finished."""
        self._clip_reads = None
        ended_at = self._audio.ended_at
        if ended_at is None:
            return
        state = {"assistant": self._previous} if self._previous else {}
        reads: dict[PerceptionSource, asyncio.Task] = {}
        for source in self._audio_sources:
            # Each classifier takes clips of its own lengths.
            clip = self._audio.clip(source.classifier)
            if clip:
                logger.debug(
                    f"{self}: {source.classifier} is reading {clip.duration_secs:.1f}s "
                    "of the user's audio"
                )
                reads[source] = self._start_read(self._read_audio(source, state, clip))
        if reads:
            self._clip_reads = _ClipReads(ended_at, reads)

    async def _read_audio(
        self, source: PerceptionSource, state: dict[str, Any], clip: ClassifierAudio
    ) -> list[PerceptionSignal] | None:
        """A source's reading of a clip, broadcast once made, or None when its classifier fails."""
        questions = source.questions
        signals = None
        try:
            results = await source.classifier.ask(
                state, self._classifier_questions(questions), inputs=[clip]
            )
            signals = self._signals(results, questions)
            await self.broadcast_frame(PerceptionFrame, perception=signals)
        except ClassifierError as e:
            await self.push_error(f"{self}: {e}", e)
        return signals

    def _start_read(self, read: Coroutine[Any, Any, list[PerceptionSignal] | None]) -> asyncio.Task:
        """Run a read in a task of its own, kept until it's done so cleanup can stop it."""
        task = self.create_task(read)
        self._reads.add(task)
        task.add_done_callback(self._reads.discard)
        return task

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
        heard = await self._clip_readings(speech_end)
        readings = [
            self._reading({**heard, **await self._text_readings(transcription)})
            for transcription in held
        ]
        # The stop strategies end the turn at once on a finalized transcription,
        # so only the speech's last one is, and none when the user has spoken
        # again since.
        over = speech_end is not None and self._audio.ended_at == speech_end
        for transcription, reading in zip(held, readings):
            frame = transcription.frame
            frame.perception = reading
            frame.finalized = over and transcription is held[-1]
            await self.push_frame(frame)

    async def _wait_for_speech_end(self) -> float | None:
        """When the user stopped speaking, once the STT's latency has passed; None if not yet."""
        if self._audio.speaking:
            logger.debug(f"{self}: transcriptions are waiting for the user to stop speaking")
        speech_end = await self._audio.wait_for_end(_SPEECH_END_WAIT_SECS)
        if speech_end is not None:
            await asyncio.sleep(max(0.0, speech_end + self._stt_latency - time.monotonic()))
        return speech_end

    async def _clip_readings(self, speech_end: float | None) -> "_Readings":
        """What each audio source heard in the speech, waited for up to its timeout after it."""
        # Reads are used once, and only for the speech their clips end with, so
        # a transcription of no new speech gets no reading.
        clip_reads, self._clip_reads = self._clip_reads, None
        if not clip_reads or clip_reads.ended_at != speech_end:
            return {}
        # The shortest timeout first, so the wait ends once every read is done
        # or out of time.
        for source, read in sorted(clip_reads.reads.items(), key=lambda item: item[0].timeout):
            if read.done():
                continue
            logger.debug(f"{self}: transcriptions are waiting for {source.classifier}'s reading")
            deadline = clip_reads.ended_at + source.timeout
            # A wait that doesn't cancel the read when it's late, nor fails
            # when a mute cancels it.
            await asyncio.wait({read}, timeout=max(0.0, deadline - time.monotonic()))
            if not read.done():
                logger.warning(
                    f"{self}: {source.classifier} hadn't read the clip after "
                    f"{source.timeout}s, the speech goes on without it"
                )
        return {source: self._found(read) for source, read in clip_reads.reads.items()}

    async def _text_readings(self, transcription: "_HeldTranscription") -> "_Readings":
        """What each text source read in a held transcription, each within its own timeout."""
        if transcription.reads:
            await asyncio.wait(transcription.reads.values())
        return {source: self._found(read) for source, read in transcription.reads.items()}

    def _found(self, read: asyncio.Task) -> list[PerceptionSignal] | None:
        """What a read found, or None when it's late, was stopped or failed."""
        if read.done() and not read.cancelled() and read.exception() is None:
            return read.result()
        return None

    def _reading(self, readings: "_Readings") -> list[PerceptionSignal] | None:
        """The sources' readings as one, in the sources' order, or None when none read the turn."""
        read = [
            signals for source in self._sources if (signals := readings.get(source)) is not None
        ]
        return [signal for signals in read for signal in signals] if read else None

    async def _cancel_clip_reads(self):
        """Stop the reads of the latest clips still running, if any."""
        if self._clip_reads:
            for read in self._clip_reads.reads.values():
                if not read.done():
                    logger.debug(f"{self}: dropping a clip still being read")
                    await self.cancel_task(read)
        self._clip_reads = None

    def _name_clash(self) -> str | None:
        """Why the sources can't share a reading, if two report a signal of the same name."""
        seen: set[str] = set()
        for source in self._sources:
            for name in source.names:
                if name in seen:
                    return f"two sources report {name!r}, and a reading has one signal of a name"
                seen.add(name)
        return None

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


#: What each source read in a turn: its signals, or None when it read nothing.
_Readings = dict[PerceptionSource, list[PerceptionSignal] | None]


class _HeldTranscription(NamedTuple):
    """A transcription held until its speech is read, and each text source's read of it."""

    frame: TranscriptionFrame
    reads: dict[PerceptionSource, asyncio.Task]


class _ClipReads(NamedTuple):
    """Each audio source's read of its clip, and when the user stopped the speech they end with."""

    ended_at: float
    reads: dict[PerceptionSource, asyncio.Task]


class _RecentAudio:
    """The user's recent audio as it was heard, cut into clips each time they stop speaking.

    A clip is cut for a classifier. It ends where the user stopped and holds
    their speech, and the audio heard before it when that's shorter than the
    classifier's ``min_audio_secs``, so a short reply is heard with what came
    just before it. At the start of a call there's less audio than that, and
    the clip starts with silence. Audio heard while the user is muted is kept
    as silence.
    """

    def __init__(self, classifiers: Sequence[BaseClassifier]):
        # How far back the classifiers' clips reach between them, and the
        # audio's sample rate.
        longest = [classifier.max_audio_secs for classifier in classifiers]
        bounded = [secs for secs in longest if secs is not None]
        self._min_secs = max((classifier.min_audio_secs for classifier in classifiers), default=0.0)
        self._max_secs = max(bounded) if bounded and len(bounded) == len(longest) else None
        self._sample_rate = 0
        # The recent audio, the bytes heard before it, and where the latest
        # speech starts and ends, counted in bytes heard.
        self._audio = bytearray()
        self._dropped = 0
        self._speech_start = 0
        self._speech_end = 0
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

    def append(self, audio: bytes):
        """Add the user's audio, as silence while they're muted."""
        self._audio += bytes(len(audio)) if self._muted else audio
        self._trim()

    def start(self, start_secs: float):
        """Mark the start of the user's speech, start_secs before the VAD said so."""
        self._speech_start = max(self._dropped, self._heard - self._bytes(start_secs))
        self._ended.clear()
        self._ended_at = None

    def stop(self, stop_secs: float):
        """Mark the end of the user's speech, stop_secs before the VAD said so."""
        self._speech_end = max(self._speech_start, self._heard - self._bytes(stop_secs))
        # The stop strategies count from when the user stopped, too.
        self._ended_at = time.monotonic() - stop_secs
        self._ended.set()

    def clip(self, classifier: BaseClassifier) -> ClassifierAudio | None:
        """A classifier's clip of the user's latest speech, or None when none was heard unmuted."""
        if self._muted or self._speech_end <= self._speech_start:
            return None
        start = min(self._speech_start, self._speech_end - self._bytes(classifier.min_audio_secs))
        if classifier.max_audio_secs is not None:
            start = max(start, self._speech_end - self._bytes(classifier.max_audio_secs))
        silence = bytes(max(0, self._dropped - start))
        audio = self._audio[max(0, start - self._dropped) : self._speech_end - self._dropped]
        return ClassifierAudio(audio=silence + bytes(audio), sample_rate=self._sample_rate)

    def mute(self, muted: bool):
        """Keep the user's audio as silence while they're muted, ending any speech when they are."""
        self._muted = muted
        if muted and self.speaking:
            self._speech_end = self._heard
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

    @property
    def _heard(self) -> int:
        """The bytes of audio heard so far."""
        return self._dropped + len(self._audio)

    def _trim(self):
        """Drop the audio no clip can reach: what's before the latest speech and a clip before it."""
        reach = self._bytes(self._min_secs + 1.0)
        start = self._speech_start if self.speaking else self._heard
        keep_from = start - reach
        if self._max_secs is not None:
            keep_from = max(keep_from, self._heard - self._bytes(self._max_secs) - reach)
        if keep_from > self._dropped:
            del self._audio[: keep_from - self._dropped]
            self._dropped = keep_from

    def _bytes(self, secs: float) -> int:
        """How many bytes of 16-bit mono audio make secs seconds."""
        return int(secs * self._sample_rate) * 2
