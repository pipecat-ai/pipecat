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
    UserStoppedSpeakingFrame,
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

    Goes between the STT service and the user aggregator. The answers are set
    as each final transcription's ``perception``: one signal per question,
    named after it. The signals are declared at start in a
    :class:`~pipecat.frames.frames.PerceptionMetadataFrame`.

    A text classifier is asked about each final transcription, with the bot's
    reply before it. A classifier that can hear, as said by its
    ``supports_audio``, hears the user's speech in clips instead, with the bot's reply before it
    as the state. The reader keeps the user's recent speech across turns: each
    stretch the VAD reports, from a second before it, without the silence in
    between. Each time the VAD reports silence, it sends the speech since the
    last clip, topped up with the user's earlier speech when that's shorter
    than the classifier's ``min_audio_secs``. A clip isn't sent while the last
    one is still being read: its speech goes in the next. Each final
    transcription gets the answers about the turn's latest clip, or about the
    turn's latest clip answered when that one is late. No speech is kept while
    the user is muted.

    Every reading is also broadcast as soon as it's made, in a
    :class:`~pipecat.frames.frames.PerceptionFrame`, so processors and clients
    can act on it before a transcription carries it.

    With a text classifier, a final transcription goes on as soon as it's
    read, up to ``timeout`` seconds after it arrived. With an audio classifier,
    the reader holds the final transcriptions of the user's speech until the
    user stops speaking, up to the 5 seconds the user aggregator keeps a turn
    open without user activity by default, the STT has had its
    ``ttfs_p99_latency`` to send the rest, and the latest clip is read, up to
    ``timeout`` seconds after the user stopped. They go on together, the last
    one marked ``finalized`` unless the user has spoken again since, so the
    user turn stop strategies end the turn as soon as it arrives. The reader
    then adds the timeout to the ``ttfs_p99_latency`` of the
    :class:`~pipecat.frames.frames.STTMetadataFrame` it passes on, so the stop
    strategies wait for the transcriptions it holds. A transcription passes
    unread when no reading is ready in time or the classifier fails. A
    classifier that can't be set up is reported as an error that leaves the
    reader unusable, and transcriptions pass unread.

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
        self._classifier = classifier
        self._questions = dict(questions)
        self._previous: str | None = None
        # The user's speech, kept and read for a classifier that hears audio:
        # the stretch the VAD is reporting, and the recent ones before it.
        self._audio_classifier = classifier if classifier.supports_audio else None
        if timeout is None:
            timeout = 2.0 if self._audio_classifier else 0.5
        self._timeout = timeout
        max_secs = self._audio_classifier.max_audio_secs if self._audio_classifier else None
        self._speech_buffer = SpeechSegmentBuffer(max_secs=max_secs)
        self._speech = bytearray()
        self._speech_ended = asyncio.Event()
        self._speech_ended.set()
        self._speech_ended_at: float | None = None
        self._sample_rate = 0
        self._muted = False
        self._unclipped_bytes = 0
        # Counts the user's turns, so a clip read after its turn ended isn't
        # given to the next turn's transcriptions.
        self._turn = 0
        self._audio_read: asyncio.Task | None = None
        self._latest_reading: list[PerceptionSignal] | None = None
        self._stt_latency = 0.0
        # Final transcriptions waiting for an audio reading, and the task that
        # pushes them on.
        self._held: list[TranscriptionFrame] = []
        self._release_task: asyncio.Task | None = None
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
        self._sample_rate = setup.audio_in_sample_rate
        self._speech_buffer.set_sample_rate(self._sample_rate)
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
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._handle_user_stopped_speaking(frame)
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
        if self._audio_classifier and not self._muted:
            self._speech_buffer.append(frame.audio)

    def _holds(self, frame: TranscriptionFrame) -> bool:
        """Whether the transcription waits for an audio reading."""
        return bool(self._audio_classifier and frame.text.strip() and self.is_usable)

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
        if self._audio_classifier and self.is_usable:
            frame.ttfs_p99_latency += self._timeout

    async def _handle_vad_user_started_speaking(self, frame: VADUserStartedSpeakingFrame):
        self._speech_buffer.start()
        self._speech_ended.clear()
        self._speech_ended_at = None

    async def _handle_vad_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        segment = self._speech_buffer.stop()
        if self._audio_classifier and self.is_usable and segment:
            self._keep_speech(segment)
            self._read_speech(self._audio_classifier)
        # The user stopped speaking stop_secs before the VAD reports it, which
        # is when the stop strategies start waiting for transcriptions.
        self._speech_ended_at = time.monotonic() - frame.stop_secs
        self._speech_ended.set()

    async def _handle_user_stopped_speaking(self, frame: UserStoppedSpeakingFrame):
        self._turn += 1
        self._latest_reading = None

    async def _handle_mute(self, frame: UserMuteStartedFrame | UserMuteStoppedFrame):
        self._muted = isinstance(frame, UserMuteStartedFrame)
        if self._muted:
            await self._cancel_read()
            self._latest_reading = None
            self._forget_speech()

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

    def _keep_speech(self, segment: bytes):
        """Add a stretch of the user's speech to the recent speech."""
        self._speech += segment
        self._unclipped_bytes += len(segment)

    def _read_speech(self, classifier: BaseClassifier):
        """Start reading the speech since the last clip, unless a read is still waiting."""
        if self._audio_read and not self._audio_read.done():
            logger.debug(f"{self}: a clip is still being read, the new speech waits for the next")
            return
        min_bytes = int(classifier.min_audio_secs * self._sample_rate) * 2
        size = max(self._unclipped_bytes, min_bytes)
        if classifier.max_audio_secs is not None:
            size = min(size, int(classifier.max_audio_secs * self._sample_rate) * 2)
        if len(self._speech) >= size:
            clip = ClassifierAudio(
                audio=bytes(self._speech[len(self._speech) - size :]), sample_rate=self._sample_rate
            )
            logger.debug(
                f"{self}: reading {clip.duration_secs:.1f}s of the user's speech, "
                f"{self._unclipped_bytes / (2 * self._sample_rate):.1f}s of it new"
            )
            state = {"assistant": self._previous} if self._previous else {}
            self._unclipped_bytes = 0
            # A later clip only ever tops up with the end of this one.
            del self._speech[: len(self._speech) - min_bytes]
            self._audio_read = self.create_task(
                self._read_audio(classifier, state, clip, self._turn)
            )
        else:
            logger.debug(
                f"{self}: {len(self._speech) / (2 * self._sample_rate):.1f}s of the user's speech "
                f"so far, {classifier.min_audio_secs:g}s needed for a clip"
            )

    async def _read_audio(
        self,
        classifier: BaseClassifier,
        state: dict[str, Any],
        clip: ClassifierAudio,
        turn: int,
    ):
        """Read a clip and broadcast its answers, keeping them as its turn's latest reading."""
        questions = self.questions
        try:
            results = await classifier.ask(state, self._classifier_questions(questions), audio=clip)
            signals = self._signals(results, questions)
            if turn == self._turn:
                self._latest_reading = signals
            await self.broadcast_frame(PerceptionFrame, perception=signals)
        except ClassifierError as e:
            await self.push_error(f"{self}: {e}", e)

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
        await self._wait_for_latest_clip(speech_end)
        # The stop strategies end the turn at once on a finalized transcription,
        # so only the speech's last one is, and none when the user has spoken
        # again since.
        over = speech_end is not None and self._speech_ended_at == speech_end
        for frame in held:
            frame.perception = self._latest_reading
            frame.finalized = over and frame is held[-1]
            await self.push_frame(frame)

    async def _wait_for_speech_end(self) -> float | None:
        """When the user stopped speaking, once the STT's latency has passed; None if not yet."""
        if not self._speech_ended.is_set():
            logger.debug(f"{self}: transcriptions are waiting for the user to stop speaking")
            try:
                await asyncio.wait_for(self._speech_ended.wait(), _SPEECH_END_WAIT_SECS)
            except TimeoutError:
                logger.debug(f"{self}: the user is still speaking after {_SPEECH_END_WAIT_SECS}s")
        speech_end = self._speech_ended_at
        if speech_end is not None:
            await asyncio.sleep(max(0.0, speech_end + self._stt_latency - time.monotonic()))
        return speech_end

    async def _wait_for_latest_clip(self, speech_end: float | None):
        """Wait for the latest clip's reading, up to the timeout after the user stopped speaking."""
        read = self._audio_read
        if read and not read.done():
            logger.debug(f"{self}: transcriptions are waiting for the latest clip")
            deadline = (speech_end if speech_end is not None else time.monotonic()) + self._timeout
            # A wait that doesn't cancel the read when it's late, nor fails
            # when a mute cancels it.
            await asyncio.wait({read}, timeout=max(0.0, deadline - time.monotonic()))
            if not read.done():
                fallback = "an earlier clip's reading" if self._latest_reading else "no reading"
                logger.warning(
                    f"{self}: the latest clip wasn't read after {self._timeout}s, using {fallback}"
                )

    async def _cancel_read(self):
        """Stop the read still running, if any."""
        if self._audio_read and not self._audio_read.done():
            logger.debug(f"{self}: dropping a clip still being read")
            await self.cancel_task(self._audio_read)
        self._audio_read = None

    def _forget_speech(self):
        """Drop the user's speech kept so far."""
        self._speech_buffer.stop()
        self._speech.clear()
        self._unclipped_bytes = 0
        self._speech_ended_at = time.monotonic()
        self._speech_ended.set()

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
