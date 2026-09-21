#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Voicemail detection for outbound calls.

A bot that places a call needs to know whether a person answered or the call
went to voicemail. :class:`VoicemailDetector` listens to what the other side
says and asks a classifier; until it has an answer, :class:`TTSGate` holds
the bot's speech back so a voicemail greeting is never talked over.

Any classifier will do. An
:class:`~pipecat.classifiers.llm.classifier.LLMClassifier` needs a service
with ``run_inference()``, so a realtime LLM cannot back the detector.
"""

import asyncio
import warnings
from typing import Literal

from loguru import logger

from pipecat.classifiers.base_classifier import (
    BaseClassifier,
    ChoiceQuestion,
    ChoiceResult,
    ClassifierError,
)
from pipecat.classifiers.llm.classifier import DEFAULT_INSTRUCTIONS, LLMClassifier
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    StopFrame,
    SystemFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    WorkerFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.services.llm_service import LLMService
from pipecat.utils.sync.base_notifier import BaseNotifier
from pipecat.utils.sync.event_notifier import EventNotifier

# Lifecycle frames that must still flow after voicemail is detected.
_CLOSED_GATE_ALLOWLIST = (SystemFrame, EndFrame, StopFrame, WorkerFrame)

_Verdict = Literal["conversation", "voicemail"]

#: The question put to the classifier, with the transcript so far as the state.
VOICEMAIL_QUESTION = ChoiceQuestion(
    instructions=(
        "A bot has placed an outbound phone call. This is what was heard after the "
        "call connected. Decide whether a person answered or the call went to "
        "voicemail."
    ),
    options={
        "conversation": (
            "a person answered: a greeting such as 'hello?', 'hi', 'yeah?' or "
            "'John speaking'; a question to the caller such as 'who is this?' or "
            "'can I help you?'; spontaneous speech that expects a reply"
        ),
        "voicemail": (
            "an automated greeting or carrier message: 'you've reached', 'leave a "
            "message', 'I'm not available right now', 'call me back', 'mailbox is "
            "full', 'not in service', 'all circuits are busy', 'our office is "
            "currently closed'"
        ),
    },
)


class TTSGate(FrameProcessor):
    """Holds the bot's speech until the voicemail decision is made.

    Placed right after the TTS service. TTS frames are buffered while the
    decision is pending; every other frame passes through. A conversation
    verdict releases the buffered frames in order, a voicemail verdict
    discards them, since they were meant for a person.
    """

    def __init__(self, conversation_notifier: BaseNotifier, voicemail_notifier: BaseNotifier):
        """Initialize the TTS gate.

        Args:
            conversation_notifier: Signals that a person answered and the
                buffered frames should play.
            voicemail_notifier: Signals that the call went to voicemail and
                the buffered frames should be dropped.
        """
        super().__init__()
        self._conversation_notifier = conversation_notifier
        self._voicemail_notifier = voicemail_notifier
        self._frame_buffer: list[tuple[Frame, FrameDirection]] = []
        self._gating_active = True
        self._conversation_task: asyncio.Task | None = None
        self._voicemail_task: asyncio.Task | None = None

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the processor with required components.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        self._conversation_task = self.create_task(self._wait_for_conversation())
        self._voicemail_task = self.create_task(self._wait_for_voicemail())

    async def cleanup(self):
        """Clean up the processor resources."""
        await super().cleanup()
        if self._conversation_task:
            await self.cancel_task(self._conversation_task)
            self._conversation_task = None
        if self._voicemail_task:
            await self.cancel_task(self._voicemail_task)
            self._voicemail_task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Buffer TTS frames while the decision is pending; pass the rest.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if self._gating_active and isinstance(
            frame, (TTSStartedFrame, TTSStoppedFrame, TTSTextFrame, TTSAudioRawFrame)
        ):
            self._frame_buffer.append((frame, direction))
        else:
            await self.push_frame(frame, direction)

    async def _wait_for_conversation(self):
        await self._conversation_notifier.wait()
        self._gating_active = False
        for frame, direction in self._frame_buffer:
            await self.push_frame(frame, direction)
        self._frame_buffer.clear()

    async def _wait_for_voicemail(self):
        await self._voicemail_notifier.wait()
        self._gating_active = False
        self._frame_buffer.clear()


class VoicemailDetector(FrameProcessor):
    """Decides whether a person answered an outbound call or it went to voicemail.

    Placed after the STT service. It passes every frame through and collects
    what the other side says. After each transcription it asks its classifier,
    in the background, whether the call reached a person or a voicemail, and
    keeps asking as more is said. Once the caller has been quiet for
    ``decision_timeout``, the latest answer decides and the detector acts:

    - CONVERSATION: the bot's held-back speech is released and the call goes
      on as normal.
    - VOICEMAIL: the held-back speech is dropped, the pipeline is interrupted,
      no further input reaches the conversation, and ``on_voicemail_detected``
      fires once the greeting has finished, so the handler can leave a
      message.

    Example::

        detector = VoicemailDetector(classifier=JevClassifier(api_key=...))

        @detector.event_handler("on_voicemail_detected")
        async def handle_voicemail(processor):
            await processor.push_frame(TTSSpeakFrame("Please call me back."))

        pipeline = Pipeline([
            transport.input(),
            stt,
            detector.detector(),          # Classification
            context_aggregator.user(),
            llm,
            tts,
            detector.gate(),              # TTS gating
            transport.output(),
            context_aggregator.assistant(),
        ])

    Event handlers available:

    - on_conversation_detected: A person answered. The handler receives the
      detector, which it can push frames through.
    - on_voicemail_detected: The call went to voicemail and the greeting has
      been quiet for ``voicemail_response_delay`` seconds. The handler
      receives the detector, which it can push frames through.
    """

    def __init__(
        self,
        *,
        classifier: BaseClassifier | None = None,
        voicemail_response_delay: float = 2.0,
        decision_timeout: float = 1.0,
        llm: LLMService | None = None,
        custom_system_prompt: str | None = None,
    ):
        """Initialize the voicemail detector.

        Args:
            classifier: What decides between a person and a voicemail. It is
                asked a ``choice`` question with the transcript so far.
            voicemail_response_delay: Seconds of silence after a voicemail
                verdict before ``on_voicemail_detected`` fires, so the message
                is left after the greeting ends and the recording starts.
            decision_timeout: Seconds of silence after the caller stops speaking
                before the latest answer decides. A greeting resumes after its
                pauses while a person stays quiet, so a verdict on a fragment
                such as "hi, this is Sam" is not acted on until the caller has
                really stopped.
            llm: LLM service used for the classification.

                .. deprecated:: 1.12.0
                    Use ``classifier`` instead. Will be removed in 2.0.0.
            custom_system_prompt: System prompt for the ``llm``.

                .. deprecated:: 1.12.0
                    Use ``classifier`` instead. Will be removed in 2.0.0.
        """
        super().__init__()
        if llm is not None:
            warnings.warn(
                "VoicemailDetector's `llm` parameter is deprecated since 1.12.0 and will be "
                "removed in 2.0.0. Use `classifier` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if custom_system_prompt is not None:
            warnings.warn(
                "VoicemailDetector's `custom_system_prompt` parameter is deprecated since 1.12.0 "
                "and will be removed in 2.0.0. Use `classifier` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if classifier is None:
            if llm is None:
                raise ValueError("VoicemailDetector needs a classifier")
            # The old prompt asked for a one-word reply; the classifier's own
            # instructions, which ask for a JSON object, come last and win.
            instructions = None
            if custom_system_prompt:
                instructions = f"{custom_system_prompt}\n\n{DEFAULT_INSTRUCTIONS}"
            classifier = LLMClassifier(llm=llm, instructions=instructions)
        self._classifier = classifier
        self._voicemail_response_delay = voicemail_response_delay
        self._decision_timeout = decision_timeout

        # The gate that holds the bot's speech until the verdict.
        self._conversation_notifier = EventNotifier()
        self._voicemail_notifier = EventNotifier()
        self._tts_gate = TTSGate(self._conversation_notifier, self._voicemail_notifier)

        # What was heard so far, and what the classifier made of it. Transcriptions
        # are queued for the classifying task, which owns the transcript and
        # the answer.
        self._segments: asyncio.Queue[str] = asyncio.Queue()
        self._classify_task: asyncio.Task | None = None
        self._transcript: list[str] = []
        self._last_result: ChoiceResult | None = None
        self._decision: _Verdict | None = None

        # The silence timer: after the caller stops, it acts on the best answer.
        self._user_speaking = False
        self._fallback_task: asyncio.Task | None = None

        # The voicemail handler fires once the greeting has been quiet for
        # the response delay; speech resets the wait.
        self._voicemail_task: asyncio.Task | None = None
        self._voicemail_event = asyncio.Event()
        self._voicemail_event.set()

        self._register_event_handler("on_conversation_detected")
        self._register_event_handler("on_voicemail_detected")

    def detector(self) -> "VoicemailDetector":
        """The processor to place after the STT service.

        Returns:
            This detector.
        """
        return self

    def gate(self) -> TTSGate:
        """The processor to place after the TTS service.

        Returns:
            The gate that holds speech until the decision is made.
        """
        return self._tts_gate

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the processor and its classifier.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._classifier.setup(self.task_manager)
        self._classify_task = self.create_task(self._classify_segments())
        self._voicemail_task = self.create_task(self._delayed_voicemail_handler())

    async def cleanup(self):
        """Clean up the processor and its classifier."""
        await super().cleanup()
        if self._classify_task:
            await self.cancel_task(self._classify_task)
            self._classify_task = None
        await self._cancel_fallback()
        if self._voicemail_task:
            await self.cancel_task(self._voicemail_task)
            self._voicemail_task = None
        await self._classifier.cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Collect transcriptions, keep the voicemail timer, and gate after a voicemail.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            self._segments.put_nowait(frame.text.strip())
            # A transcription often lands after the caller has stopped; the
            # silence timer restarts from it.
            if not self._user_speaking and self._decision is None:
                await self._restart_fallback()
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._user_speaking = True
            await self._cancel_fallback()
            if self._decision == "voicemail":
                self._voicemail_event.set()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_speaking = False
            if self._decision is None:
                await self._restart_fallback()
            elif self._decision == "voicemail":
                self._voicemail_event.clear()

        # After a voicemail verdict nothing more should reach the conversation,
        # only the frames that end or control the pipeline.
        if self._decision == "voicemail" and not isinstance(frame, _CLOSED_GATE_ALLOWLIST):
            return
        await self.push_frame(frame, direction)

    async def _classify_segments(self):
        """Classify the transcript every time it grows.

        Segments that arrive during a classification are taken together, so
        a burst of transcriptions costs one call on the full transcript.
        """
        while True:
            # Wait for a segment, then take any that arrived meanwhile.
            segments = [await self._segments.get()]
            while not self._segments.empty():
                segments.append(self._segments.get_nowait())
            self._transcript.extend(segments)
            # Nothing to ask once the verdict is in.
            if self._decision is None:
                await self._classify(" ".join(self._transcript))
            # Lets the silence timer know the transcript is fully classified.
            for _ in segments:
                self._segments.task_done()

    async def _classify(self, transcript: str):
        """Ask the classifier about the transcript and keep its answer."""
        try:
            results = await self._classifier.choice(transcript, {"voicemail": VOICEMAIL_QUESTION})
        except ClassifierError as e:
            logger.warning(f"{self}: classification failed: {e}")
            return
        result = results["voicemail"]
        logger.debug(f"{self}: {result.choice} ({result.confidence:.2f}) for {transcript!r}")
        # No verdict acts from here: "hi, this is Sam" is what a person says
        # and how a greeting starts, and only the silence that follows tells
        # them apart, so the silence timer decides.
        self._last_result = result

    async def _restart_fallback(self):
        """Start the silence timer over."""
        await self._cancel_fallback()
        self._fallback_task = self.create_task(self._decide_after_silence())

    async def _cancel_fallback(self):
        """Stop the silence timer, if it is running."""
        if self._fallback_task is not None:
            await self.cancel_task(self._fallback_task)
            self._fallback_task = None

    async def _decide_after_silence(self):
        """Act on the latest answer once the caller has been quiet long enough.

        A greeting goes on after its pauses; a person stops and waits. So this
        is how every verdict comes to act.
        """
        await asyncio.sleep(self._decision_timeout)
        # The answer for everything heard so far, once a call in flight is done.
        await self._segments.join()
        if self._decision is not None:
            return
        if self._last_result is not None:
            logger.info(
                f"{self}: {self._last_result.choice} ({self._last_result.confidence:.2f}) "
                f"after {self._decision_timeout}s of silence"
            )
            choice = self._last_result.choice
            await self._decide("voicemail" if choice == "voicemail" else "conversation")
        else:
            logger.warning(
                f"{self}: no answer from the classifier after {self._decision_timeout}s of "
                "silence, assuming a conversation"
            )
            await self._decide("conversation")

    async def _decide(self, label: _Verdict):
        """Record the verdict and act on it: release or drop the held speech."""
        self._decision = label
        if label == "voicemail":
            logger.info(f"{self}: VOICEMAIL detected")
            await self._voicemail_notifier.notify()
            await self.broadcast_interruption()
            self._voicemail_event.clear()
        else:
            logger.info(f"{self}: CONVERSATION detected")
            await self._conversation_notifier.notify()
            await self._call_event_handler("on_conversation_detected")

    async def _delayed_voicemail_handler(self):
        """Fire ``on_voicemail_detected`` once the greeting has been quiet for the delay."""
        while True:
            try:
                await asyncio.wait_for(
                    self._voicemail_event.wait(), timeout=self._voicemail_response_delay
                )
                await asyncio.sleep(0.1)
            except TimeoutError:
                await self._call_event_handler("on_voicemail_detected")
                break
