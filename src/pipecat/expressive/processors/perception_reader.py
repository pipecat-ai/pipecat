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
    BotStartedSpeakingFrame,
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
    hears the user's speech each time they stop: the latest, with what they
    said before it since the bot last spoke when that's shorter than the
    classifier answers best. It
    hears one clip at a time, and of the clips that come while it's busy, only
    the newest. The transcriptions of that speech wait for its answers, up to
    its source's ``timeout`` after the user stopped.

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
        names = [name for source in sources for name in source.names]
        if clash := next((name for name in names if names.count(name) > 1), None):
            raise ValueError(
                f"{self}: two sources report {clash!r}, and a reading has one signal of a name"
            )
        # Who to ask: the sources that read transcriptions, and the ones that hear.
        self._sources = list(sources)
        self._text_sources = [s for s in self._sources if not s.classifier.supports_audio]
        self._audio_sources = [s for s in self._sources if s.classifier.supports_audio]
        # The bot's last reply, which the user's turn is read against.
        self._previous: str | None = None
        # For the sources that hear: the user's recent speech, their reads of
        # the clips that end with the user's latest speech, and each one's
        # latest read, which its next one waits for.
        self._audio = _RecentSpeech([source.classifier for source in self._audio_sources])
        self._clip_reads: _Reads | None = None
        self._clip_tasks: dict[PerceptionSource, asyncio.Task] = {}
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
        elif isinstance(frame, BotStartedSpeakingFrame):
            await self._handle_bot_started_speaking(frame)
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
        self._held.append(_HeldTranscription(frame, self._read_text(frame.text)))
        if not self._release_task:
            self._release_task = self.create_task(self._release_transcriptions())

    async def _handle_transcription(self, frame: TranscriptionFrame):
        if frame.text.strip() and self.is_usable:
            frame.perception = self._reading(await self._wait(self._read_text(frame.text)))

    async def _handle_stt_metadata(self, frame: STTMetadataFrame):
        self._stt_latency = frame.ttfs_p99_latency
        # Transcriptions held for an audio reading reach the user aggregator up
        # to the longest timeout later than the STT sends them, so the stop
        # strategies wait that much longer.
        if self._audio_sources and self.is_usable:
            frame.ttfs_p99_latency += max(source.timeout for source in self._sources)

    async def _handle_vad_user_started_speaking(self, frame: VADUserStartedSpeakingFrame):
        self._audio.start()

    async def _handle_vad_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        self._audio.stop(frame.stop_secs)
        if self._audio_sources and self.is_usable:
            self._read_speech()

    async def _handle_mute(self, frame: UserMuteStartedFrame | UserMuteStoppedFrame):
        muted = isinstance(frame, UserMuteStartedFrame)
        self._audio.mute(muted)
        if muted:
            await self._cancel_clip_reads()

    async def _handle_bot_started_speaking(self, frame: BotStartedSpeakingFrame):
        self._audio.bot_started_speaking()

    async def _handle_assistant_turn(self, frame: LLMContextAssistantTurnFrame):
        self._previous = frame.text

    async def _declare_signals(self):
        """Push the signals the reader reports, if it can report them."""
        if self.is_usable:
            await self.push_frame(
                PerceptionMetadataFrame(service_name=self.name, perception=self._capabilities())
            )

    def _capabilities(self) -> PerceptionCapabilities:
        """The signals the reader reports, from its sources' questions."""
        return PerceptionCapabilities(
            signals=[
                PerceptionTerm(name, question.description, labels=self._labels(question.question))
                for source in self._sources
                for name, question in source.questions.items()
            ]
        )

    def _labels(self, question: ClassifierQuestion) -> list[str]:
        """The labels a signal can report: a choice's options, or none."""
        return list(question.options) if isinstance(question, ChoiceQuestion) else []

    def _read_text(self, text: str) -> "_Reads":
        """Start each text source's read of a transcription, with the bot's reply before it."""
        state = {"assistant": self._previous, "user": text} if self._previous else {"user": text}
        reads = {
            source: self._start_read(self._read(source, state)) for source in self._text_sources
        }
        return _Reads(time.monotonic(), reads)

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
                reads[source] = self._read_clip(source, state, clip)
        if reads:
            self._clip_reads = _Reads(ended_at, reads)

    def _read_clip(
        self, source: PerceptionSource, state: dict[str, Any], clip: ClassifierAudio
    ) -> asyncio.Task:
        """Start a source's read of a clip, which asks once the source's earlier read is done."""
        read = self._start_read(
            self._read_in_turn(self._clip_tasks.get(source), source, state, clip)
        )
        self._clip_tasks[source] = read
        return read

    async def _read_in_turn(
        self,
        earlier: asyncio.Task | None,
        source: PerceptionSource,
        state: dict[str, Any],
        clip: ClassifierAudio,
    ) -> list[PerceptionSignal] | None:
        """A source's reading of a clip after its earlier read, or None if a newer clip came first.

        A classifier that hears is asked about one clip at a time, as Hume's
        Audio Decisions asks of live audio, and of the clips that came while it
        was busy, only about the newest.
        """
        if earlier:
            # A wait that doesn't cancel the earlier read, nor fails when a
            # mute cancels it.
            await asyncio.wait({earlier})
        superseded = earlier is not None and self._clip_tasks[source] is not asyncio.current_task()
        return None if superseded else await self._read(source, state, [clip])

    def _start_read(self, read: Coroutine[Any, Any, list[PerceptionSignal] | None]) -> asyncio.Task:
        """Run a read in a task of its own, kept until it's done so cleanup can stop it."""
        task = self.create_task(read)
        self._reads.add(task)
        task.add_done_callback(self._reads.discard)
        return task

    async def _read(
        self,
        source: PerceptionSource,
        state: dict[str, Any],
        inputs: Sequence[ClassifierAudio] = (),
    ) -> list[PerceptionSignal] | None:
        """A source's reading, broadcast once made, late or not, or None when its classifier fails."""
        questions = source.questions
        signals = None
        try:
            results = await source.classifier.ask(
                state, self._classifier_questions(questions), inputs=inputs
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
        heard = await self._wait(self._take_clip_reads(speech_end))
        readings = [
            self._reading({**heard, **await self._wait(transcription.reads)})
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

    def _take_clip_reads(self, speech_end: float | None) -> "_Reads | None":
        """The reads of the clips that end with the speech, if any."""
        # Reads are used once, and only for the speech their clips end with, so
        # a transcription of no new speech gets no reading.
        clip_reads, self._clip_reads = self._clip_reads, None
        return clip_reads if clip_reads and clip_reads.started_at == speech_end else None

    async def _wait(self, reads: "_Reads | None") -> "_Readings":
        """What each source read, waited for up to its timeout from when the reads started."""
        if reads is None:
            return {}
        for source, read in reads.tasks.items():
            deadline = reads.started_at + source.timeout
            # A wait that doesn't cancel a late read, which still broadcasts its
            # reading, nor fails when a mute cancels it.
            await asyncio.wait({read}, timeout=max(0.0, deadline - time.monotonic()))
            if not read.done():
                logger.warning(
                    f"{self}: no reading from {source.classifier} after {source.timeout}s, "
                    "the turn goes on without it"
                )
        return {source: self._found(read) for source, read in reads.tasks.items()}

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
            for read in self._clip_reads.tasks.values():
                if not read.done():
                    logger.debug(f"{self}: dropping a clip still being read")
                    await self.cancel_task(read)
        self._clip_reads = None

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


class _Reads(NamedTuple):
    """The sources' reads of a turn, and when they started: each timeout counts from then.

    A transcription's reads start when it arrives, and the reads of the clips
    of the user's speech when the user stopped it.
    """

    started_at: float
    tasks: dict[PerceptionSource, asyncio.Task]


class _HeldTranscription(NamedTuple):
    """A transcription held until its speech is read, and the text sources' reads of it."""

    frame: TranscriptionFrame
    reads: _Reads


class _RecentSpeech:
    """The user's recent speech, kept to cut a clip from each time they stop speaking.

    Each stretch of speech is kept with the silence around it: up to a second
    of the audio heard before the VAD reported it, and the silence the VAD
    waited for after it. A classifier's clip is the latest stretch, with the
    ones before it joined on until it's as long as the classifier answers
    best. Speech from before the bot spoke or before a mute is never joined to
    speech after it, so a reply is heard on its own: speech from before the
    bot's turn says how the user sounded then, not how they reply. Audio heard
    while the user is muted isn't kept.
    """

    def __init__(self, classifiers: Sequence[BaseClassifier]):
        # How much speech the classifiers' clips reach back for between them,
        # and the longest a stretch of speech is kept.
        preferred = [classifier.preferred_audio_secs for classifier in classifiers]
        longest = [classifier.max_audio_secs for classifier in classifiers]
        bounded = [secs for secs in longest if secs is not None]
        self._reach_secs = max(preferred, default=0.0)
        max_secs = max(bounded) if bounded and len(bounded) == len(longest) else None
        self._segment = SpeechSegmentBuffer(max_secs=max_secs)
        self._sample_rate = 0
        # The user's recent stretches of speech, the latest last.
        self._stretches: list[bytes] = []
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
        self._segment.set_sample_rate(sample_rate)

    def append(self, audio: bytes):
        """Add the user's audio, unless they're muted."""
        if not self._muted:
            self._segment.append(audio)

    def start(self):
        """Mark the start of the user's speech."""
        self._segment.start()
        self._ended.clear()
        self._ended_at = None

    def stop(self, stop_secs: float):
        """Keep the user's speech, which ended stop_secs before the VAD said so."""
        if self.speaking:
            self._keep(self._segment.stop())
        # The stop strategies count from when the user stopped, too.
        self._ended_at = time.monotonic() - stop_secs
        self._ended.set()

    def clip(self, classifier: BaseClassifier) -> ClassifierAudio | None:
        """A classifier's clip of the user's latest speech, or None if it's too short for it."""
        audio = self._latest(classifier.preferred_audio_secs)
        if classifier.max_audio_secs is not None:
            audio = audio[-self._bytes(classifier.max_audio_secs) :]
        long_enough = audio and len(audio) >= self._bytes(classifier.min_audio_secs)
        return ClassifierAudio(audio=audio, sample_rate=self._sample_rate) if long_enough else None

    def bot_started_speaking(self):
        """Keep the speech from before the bot spoke out of the clips after it.

        Speech the user is in the middle of is kept whole.
        """
        self._stretches.clear()

    def mute(self, muted: bool):
        """Drop the speech kept when the user is muted, and end any speech they're in."""
        self._muted = muted
        if muted:
            self._segment.stop()
            self._stretches.clear()
            if self.speaking:
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

    def _keep(self, stretch: bytes):
        """Keep a stretch of speech, and the ones before it a clip can still reach."""
        if stretch:
            self._stretches.append(stretch)
        reach = self._bytes(self._reach_secs)
        while len(self._stretches) > 1 and sum(map(len, self._stretches[1:])) >= reach:
            del self._stretches[0]

    def _latest(self, secs: float) -> bytes:
        """The latest stretches of speech joined, from the newest back until they last secs."""
        joined = b""
        for stretch in reversed(self._stretches):
            if joined and len(joined) >= self._bytes(secs):
                break
            joined = stretch + joined
        return joined

    def _bytes(self, secs: float) -> int:
        """How many bytes of 16-bit mono audio make secs seconds."""
        return int(secs * self._sample_rate) * 2
