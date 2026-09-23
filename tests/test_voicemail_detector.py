#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.extensions.voicemail.voicemail_detector import (
    ClassifierGate,
    ConversationGate,
    NotifierGate,
    VoicemailDetector,
)
from pipecat.frames.frames import (
    CancelWorkerFrame,
    EndWorkerFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StopWorkerFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.utils.sync.base_notifier import BaseNotifier
from pipecat.utils.sync.event_notifier import EventNotifier

# Long enough for the classification verdict to reach the delayed voicemail
# handler, which fires VOICEMAIL_DELAY after the verdict.
VERDICT_SETTLE = 1.0
VOICEMAIL_DELAY = 0.1


class _Passthrough(FrameProcessor):
    """Stands in for a real processor, forwarding every frame unchanged.

    Used both as the classifier LLM, so ClassificationProcessor runs its real
    path with no network, and as a processor downstream of the detector.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


def _names(frames) -> list[str]:
    return [type(f).__name__ for f in frames]


#
# Gate-level tests
#
# Each gate is built with its closing notifier already fired, so the gate's
# waiter closes it during setup. Every closed-gate case also sends a data frame
# the gate must swallow: if the gate were still open that frame escapes and the
# case fails, so a passing lifecycle assertion cannot be vacuous.
#

GATE_CLASSES = (NotifierGate, ClassifierGate, ConversationGate)


def _gate(gate_cls, notifier: BaseNotifier) -> NotifierGate:
    """Build a gate of the given class that closes when `notifier` fires."""
    if gate_cls is ClassifierGate:
        # The second notifier reports conversation detection, not gate closure.
        return ClassifierGate(notifier, EventNotifier())
    return gate_cls(notifier)


class TestClosedGateLifecycleFrames(unittest.IsolatedAsyncioTestCase):
    async def _run_closed_gate(self, gate_cls, frame: Frame, direction: FrameDirection):
        """Send `frame` and a data frame through an already-closed gate."""
        notifier = EventNotifier()
        await notifier.notify()
        gate = _gate(gate_cls, notifier)

        blocked = LLMTextFrame(text="should not escape a closed gate")
        down, up = await run_test(
            gate,
            # The data frame goes first: the lifecycle frames end the pipeline
            # worker, so anything queued behind them may never be pushed.
            frames_to_send=[SleepFrame(0.1), blocked, frame],
            frames_to_send_direction=direction,
            start_timeout=5.0,
        )
        return down if direction == FrameDirection.DOWNSTREAM else up

    async def test_closed_gate_passes_end_worker_frame(self):
        for gate_cls in GATE_CLASSES:
            for direction in (FrameDirection.DOWNSTREAM, FrameDirection.UPSTREAM):
                with self.subTest(gate=gate_cls.__name__, direction=direction.name):
                    received = await self._run_closed_gate(
                        gate_cls, EndWorkerFrame(reason="Voicemail detected."), direction
                    )
                    self.assertTrue(
                        any(isinstance(f, EndWorkerFrame) for f in received),
                        f"EndWorkerFrame did not cross the closed gate: {_names(received)}",
                    )
                    self.assertFalse(
                        any(isinstance(f, LLMTextFrame) for f in received),
                        f"Closed gate leaked a data frame: {_names(received)}",
                    )

    async def test_closed_gate_passes_stop_worker_frame(self):
        for gate_cls in GATE_CLASSES:
            with self.subTest(gate=gate_cls.__name__):
                received = await self._run_closed_gate(
                    gate_cls, StopWorkerFrame(), FrameDirection.DOWNSTREAM
                )
                self.assertTrue(
                    any(isinstance(f, StopWorkerFrame) for f in received),
                    f"StopWorkerFrame did not cross the closed gate: {_names(received)}",
                )

    async def test_closed_gate_passes_cancel_worker_frame(self):
        for gate_cls in GATE_CLASSES:
            with self.subTest(gate=gate_cls.__name__):
                received = await self._run_closed_gate(
                    gate_cls, CancelWorkerFrame(reason="Caller hung up."), FrameDirection.DOWNSTREAM
                )
                self.assertTrue(
                    any(isinstance(f, CancelWorkerFrame) for f in received),
                    f"CancelWorkerFrame did not cross the closed gate: {_names(received)}",
                )

    async def test_open_gate_passes_data_frames(self):
        for gate_cls in GATE_CLASSES:
            with self.subTest(gate=gate_cls.__name__):
                gate = _gate(gate_cls, EventNotifier())
                down, _up = await run_test(
                    gate,
                    frames_to_send=[LLMTextFrame(text="hello")],
                    start_timeout=5.0,
                )
                self.assertTrue(
                    any(isinstance(f, LLMTextFrame) for f in down),
                    f"Open gate dropped a data frame: {_names(down)}",
                )


#
# Detector-level tests
#


def _verdict_frames(verdict: str) -> list[Frame]:
    """Frames a streaming LLM emits for a one-word classification verdict."""
    return [
        LLMFullResponseStartFrame(),
        LLMTextFrame(text=verdict),
        LLMFullResponseEndFrame(),
        SleepFrame(VERDICT_SETTLE),
    ]


def _detector() -> VoicemailDetector:
    return VoicemailDetector(
        llm=_Passthrough(),  # type: ignore[arg-type]
        voicemail_response_delay=VOICEMAIL_DELAY,
    )


class TestVoicemailDetectorEndWorkerFrame(unittest.IsolatedAsyncioTestCase):
    async def test_handler_pushing_upstream_ends_worker(self):
        detector = _detector()

        @detector.event_handler("on_voicemail_detected")
        async def _on_voicemail(processor: FrameProcessor):
            await processor.push_frame(
                EndWorkerFrame(reason="Voicemail detected."), FrameDirection.UPSTREAM
            )

        _down, up = await run_test(
            detector, frames_to_send=_verdict_frames("VOICEMAIL"), start_timeout=5.0
        )
        self.assertTrue(
            any(isinstance(f, EndWorkerFrame) for f in up),
            f"EndWorkerFrame did not escape upstream: {_names(up)}",
        )

    async def test_handler_pushing_downstream_ends_worker(self):
        detector = _detector()

        @detector.event_handler("on_voicemail_detected")
        async def _on_voicemail(processor: FrameProcessor):
            await processor.push_frame(
                EndWorkerFrame(reason="Voicemail detected."), FrameDirection.DOWNSTREAM
            )

        down, _up = await run_test(
            detector, frames_to_send=_verdict_frames("VOICEMAIL"), start_timeout=5.0
        )
        self.assertTrue(
            any(isinstance(f, EndWorkerFrame) for f in down),
            f"EndWorkerFrame did not escape downstream: {_names(down)}",
        )

    async def test_voicemail_verdict_lets_upstream_end_from_main_pipeline(self):
        detector = _detector()
        ender = _Passthrough()

        @detector.event_handler("on_voicemail_detected")
        async def _on_verdict(_processor: FrameProcessor):
            await ender.push_frame(
                EndWorkerFrame(reason="VOICEMAIL detected."), FrameDirection.UPSTREAM
            )

        _down, up = await run_test(
            Pipeline([detector, ender]),
            frames_to_send=_verdict_frames("VOICEMAIL"),
            start_timeout=5.0,
        )
        self.assertTrue(
            any(isinstance(f, EndWorkerFrame) for f in up),
            f"EndWorkerFrame did not escape upstream after VOICEMAIL: {_names(up)}",
        )

    async def test_conversation_verdict_lets_upstream_end_from_main_pipeline(self):
        detector = _detector()
        ender = _Passthrough()

        @detector.event_handler("on_conversation_detected")
        async def _on_verdict(_processor: FrameProcessor):
            await ender.push_frame(
                EndWorkerFrame(reason="CONVERSATION detected."), FrameDirection.UPSTREAM
            )

        _down, up = await run_test(
            Pipeline([detector, ender]),
            frames_to_send=_verdict_frames("CONVERSATION"),
            start_timeout=5.0,
        )
        self.assertTrue(
            any(isinstance(f, EndWorkerFrame) for f in up),
            f"EndWorkerFrame did not escape upstream after CONVERSATION: {_names(up)}",
        )


#
# Classification timeout tests
#
# The detector's classification branch runs a _Passthrough in place of the
# LLM, so the test injects the classifier's reply frames directly. A
# _SpeakOnce sits between the detector and its TTS gate, where the main LLM
# and TTS would be, and emits one TTS response for the gate to hold. Whether
# that audio reaches the sink is the observable outcome of every case.
#

CLASSIFICATION_TIMEOUT = 0.3
# Long enough past the timeout for the fallback verdict to release the gate.
TIMEOUT_SETTLE = CLASSIFICATION_TIMEOUT + 0.5


class _SpeakOnce(FrameProcessor):
    """Emits a single TTS response the first time the user stops speaking."""

    def __init__(self):
        super().__init__()
        self._spoken = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if isinstance(frame, UserStoppedSpeakingFrame) and not self._spoken:
            self._spoken = True
            await self.push_frame(TTSStartedFrame())
            await self.push_frame(
                TTSAudioRawFrame(audio=b"\x01\x00" * 800, sample_rate=8000, num_channels=1)
            )
            await self.push_frame(TTSStoppedFrame())


def _user_turn() -> list[Frame]:
    return [UserStartedSpeakingFrame(), UserStoppedSpeakingFrame()]


def _reply_frames(text: str) -> list[Frame]:
    """Frames a streaming LLM emits for a reply, without any settle time."""
    return [LLMFullResponseStartFrame(), LLMTextFrame(text=text), LLMFullResponseEndFrame()]


class TestClassificationTimeout(unittest.IsolatedAsyncioTestCase):
    def _detector_and_events(self, timeout: float | None) -> tuple[VoicemailDetector, list[str]]:
        detector = VoicemailDetector(
            llm=_Passthrough(),  # type: ignore[arg-type]
            voicemail_response_delay=VOICEMAIL_DELAY,
            classification_timeout=timeout,
        )
        events: list[str] = []

        @detector.event_handler("on_conversation_detected")
        async def _on_conversation(_processor: FrameProcessor):
            events.append("on_conversation_detected")

        @detector.event_handler("on_voicemail_detected")
        async def _on_voicemail(_processor: FrameProcessor):
            events.append("on_voicemail_detected")

        return detector, events

    async def _run(self, detector: VoicemailDetector, frames: list[Frame]):
        down, _up = await run_test(
            Pipeline([detector, _SpeakOnce(), detector.gate()]),
            frames_to_send=frames,
            start_timeout=5.0,
        )
        return [f for f in down if isinstance(f, TTSAudioRawFrame)]

    async def test_no_label_reply_defaults_to_conversation_after_timeout(self):
        detector, events = self._detector_and_events(CLASSIFICATION_TIMEOUT)
        audio = await self._run(
            detector,
            _user_turn() + _reply_frames("I'm not sure.") + [SleepFrame(TIMEOUT_SETTLE)],
        )
        self.assertEqual(len(audio), 1, "Timeout did not release the gated TTS audio")
        self.assertEqual(events, ["on_conversation_detected"])

    async def test_no_label_reply_holds_gate_without_timeout(self):
        detector, events = self._detector_and_events(None)
        audio = await self._run(
            detector,
            _user_turn() + _reply_frames("I'm not sure.") + [SleepFrame(TIMEOUT_SETTLE)],
        )
        self.assertEqual(audio, [], "Gate released audio with no verdict and no timeout")
        self.assertEqual(events, [])

    async def test_timeout_does_not_override_earlier_verdict(self):
        detector, events = self._detector_and_events(CLASSIFICATION_TIMEOUT)
        audio = await self._run(
            detector,
            _user_turn() + _reply_frames("VOICEMAIL") + [SleepFrame(TIMEOUT_SETTLE)],
        )
        self.assertEqual(audio, [], "VOICEMAIL verdict did not clear the gated TTS audio")
        self.assertEqual(events, ["on_voicemail_detected"])

    async def test_timeout_is_not_restarted_by_later_turns(self):
        # Two no-label turns, each shorter than the timeout but together
        # longer than it. A deadline restarted by the second turn would still
        # be pending when the pipeline ends.
        detector, events = self._detector_and_events(CLASSIFICATION_TIMEOUT)
        half = CLASSIFICATION_TIMEOUT * 0.7
        audio = await self._run(
            detector,
            _user_turn()
            + _reply_frames("I'm not sure.")
            + [SleepFrame(half)]
            + _user_turn()
            + _reply_frames("Still not sure.")
            + [SleepFrame(half)],
        )
        self.assertEqual(len(audio), 1, "Deadline was pushed out by the second user turn")
        self.assertEqual(events, ["on_conversation_detected"])

    async def test_timeout_is_armed_by_the_first_user_turn(self):
        # With no user turn there is nothing to classify, so the deadline is
        # never armed and the gate keeps holding.
        detector, events = self._detector_and_events(CLASSIFICATION_TIMEOUT)
        down, _up = await run_test(
            Pipeline([detector, detector.gate()]),
            frames_to_send=[
                TTSStartedFrame(),
                TTSAudioRawFrame(audio=b"\x01\x00" * 800, sample_rate=8000, num_channels=1),
                TTSStoppedFrame(),
                SleepFrame(TIMEOUT_SETTLE),
            ],
            start_timeout=5.0,
        )
        self.assertFalse(any(isinstance(f, TTSAudioRawFrame) for f in down))
        self.assertEqual(events, [])
