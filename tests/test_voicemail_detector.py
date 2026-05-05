# ABOUTME: Tests for VoicemailDetector reset functionality.
# ABOUTME: Verifies detection, re-classification after reset, and TTS gating behavior.

#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest
from typing import List, Optional

from pipecat.extensions.voicemail.voicemail_detector import VoicemailDetector
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class MockClassifierLLM(FrameProcessor):
    """Mock LLM that responds to context frames with configurable classifications.

    Returns responses from the provided list in order. When the list is
    exhausted, repeats the last response.
    """

    def __init__(self, responses: List[str]):
        """Initialize the mock classifier LLM.

        Args:
            responses: Ordered list of classification responses to return.
        """
        super().__init__()
        self._responses = list(responses)
        self._call_count = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Process frames, responding to LLMContextFrame with classification text.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            idx = min(self._call_count, len(self._responses) - 1)
            response = self._responses[idx]
            self._call_count += 1
            await self.push_frame(LLMFullResponseStartFrame())
            await self.push_frame(LLMTextFrame(text=response))
            await self.push_frame(LLMFullResponseEndFrame())
        else:
            await self.push_frame(frame, direction)


class FrameCollector(FrameProcessor):
    """Collects frames passing through for test assertions.

    Stores all non-system frames that flow downstream for later inspection.
    """

    def __init__(self) -> None:
        """Initialize the frame collector."""
        super().__init__()
        self.frames: List[Frame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Collect frames and pass them through.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            self.frames.append(frame)
        await self.push_frame(frame, direction)


class TestVoicemailDetector(unittest.IsolatedAsyncioTestCase):
    """Tests for VoicemailDetector reset functionality."""

    def _create_detector(
        self,
        responses: List[str],
        voicemail_response_delay: float = 0.1,
    ) -> VoicemailDetector:
        """Create a VoicemailDetector with a mock LLM.

        Args:
            responses: Ordered list of LLM classification responses.
            voicemail_response_delay: Delay before voicemail event fires.

        Returns:
            Configured VoicemailDetector instance.
        """
        mock_llm = MockClassifierLLM(responses=responses)
        return VoicemailDetector(
            llm=mock_llm,
            voicemail_response_delay=voicemail_response_delay,
        )

    async def _run_pipeline(
        self,
        detector: VoicemailDetector,
        send_frames_fn,
        collector: Optional[FrameCollector] = None,
    ) -> None:
        """Run a pipeline with the detector and execute a frame-sending coroutine.

        Args:
            detector: The VoicemailDetector under test.
            send_frames_fn: Async callable that receives the PipelineTask and
                sends frames through it. Must send EndFrame when done.
            collector: Optional FrameCollector to append after the gate.
        """
        processors = [detector.detector(), detector.gate()]
        if collector:
            processors.append(collector)
        pipeline = Pipeline(processors)
        task = PipelineTask(pipeline, cancel_on_idle_timeout=False)
        runner = PipelineRunner()

        async def drive():
            await asyncio.sleep(0.01)
            await send_frames_fn(task)

        await asyncio.gather(runner.run(task), drive())

    async def _send_detection_frames(
        self,
        task: PipelineTask,
        text: str = "Hello?",
    ) -> None:
        """Send the frame sequence that triggers one detection cycle.

        Args:
            task: The pipeline task to queue frames on.
            text: Transcription text to send.
        """
        await task.queue_frame(UserStartedSpeakingFrame())
        await task.queue_frame(TranscriptionFrame(text=text, user_id="test", timestamp="0"))
        await task.queue_frame(UserStoppedSpeakingFrame())

    async def test_initial_detection_conversation(self) -> None:
        """Baseline: LLM responds CONVERSATION, on_conversation_detected fires."""
        detector = self._create_detector(["CONVERSATION"])

        detected = asyncio.Event()

        async def on_conv(processor: FrameProcessor) -> None:
            detected.set()

        detector.add_event_handler("on_conversation_detected", on_conv)

        async def send(task: PipelineTask) -> None:
            await self._send_detection_frames(task, text="Hello?")
            await asyncio.wait_for(detected.wait(), timeout=5.0)
            await task.queue_frame(EndFrame())

        await self._run_pipeline(detector, send)
        self.assertTrue(detected.is_set())

    async def test_initial_detection_voicemail(self) -> None:
        """Baseline: LLM responds VOICEMAIL, on_voicemail_detected fires."""
        detector = self._create_detector(["VOICEMAIL"])

        detected = asyncio.Event()

        async def on_vm(processor: FrameProcessor) -> None:
            detected.set()

        detector.add_event_handler("on_voicemail_detected", on_vm)

        async def send(task: PipelineTask) -> None:
            await self._send_detection_frames(task, text="You've reached the voicemail of John.")
            await asyncio.wait_for(detected.wait(), timeout=5.0)
            await task.queue_frame(EndFrame())

        await self._run_pipeline(detector, send)
        self.assertTrue(detected.is_set())

    async def test_reset_clears_state(self) -> None:
        """After CONVERSATION decision, reset() re-opens all gates and clears state."""
        detector = self._create_detector(["CONVERSATION"])

        detected = asyncio.Event()

        async def on_conv(processor: FrameProcessor) -> None:
            detected.set()

        detector.add_event_handler("on_conversation_detected", on_conv)

        async def send(task: PipelineTask) -> None:
            await self._send_detection_frames(task)
            await asyncio.wait_for(detected.wait(), timeout=5.0)

            await detector.reset()

            # Verify gates are open
            self.assertTrue(detector._classifier_gate._gate_opened)
            self.assertTrue(detector._conversation_gate._gate_opened)
            # Verify classification state is cleared
            self.assertFalse(detector._classification_processor._decision_made)
            # Verify TTS gate is re-armed
            self.assertTrue(detector._voicemail_gate._gating_active)

            await task.queue_frame(EndFrame())

        await self._run_pipeline(detector, send)

    async def test_reset_conversation_then_voicemail(self) -> None:
        """Detect CONVERSATION, reset, then detect VOICEMAIL on second run."""
        detector = self._create_detector(["CONVERSATION", "VOICEMAIL"])

        events: List[str] = []
        conv_detected = asyncio.Event()
        vm_detected = asyncio.Event()

        async def on_conv(processor: FrameProcessor) -> None:
            events.append("conversation")
            conv_detected.set()

        async def on_vm(processor: FrameProcessor) -> None:
            events.append("voicemail")
            vm_detected.set()

        detector.add_event_handler("on_conversation_detected", on_conv)
        detector.add_event_handler("on_voicemail_detected", on_vm)

        async def send(task: PipelineTask) -> None:
            # First detection: CONVERSATION
            await self._send_detection_frames(task, text="Hello?")
            await asyncio.wait_for(conv_detected.wait(), timeout=5.0)

            # Reset for second detection
            await detector.reset()

            # Second detection: VOICEMAIL
            await self._send_detection_frames(task, text="You've reached the voicemail of John.")
            await asyncio.wait_for(vm_detected.wait(), timeout=5.0)
            await task.queue_frame(EndFrame())

        await self._run_pipeline(detector, send)
        self.assertEqual(events, ["conversation", "voicemail"])

    async def test_reset_voicemail_then_conversation(self) -> None:
        """Detect VOICEMAIL, reset, then detect CONVERSATION on second run."""
        detector = self._create_detector(["VOICEMAIL", "CONVERSATION"])

        events: List[str] = []
        vm_detected = asyncio.Event()
        conv_detected = asyncio.Event()

        async def on_vm(processor: FrameProcessor) -> None:
            events.append("voicemail")
            vm_detected.set()

        async def on_conv(processor: FrameProcessor) -> None:
            events.append("conversation")
            conv_detected.set()

        detector.add_event_handler("on_voicemail_detected", on_vm)
        detector.add_event_handler("on_conversation_detected", on_conv)

        async def send(task: PipelineTask) -> None:
            # First detection: VOICEMAIL
            await self._send_detection_frames(task, text="You've reached the voicemail of John.")
            await asyncio.wait_for(vm_detected.wait(), timeout=5.0)

            # Reset for second detection
            await detector.reset()

            # Second detection: CONVERSATION
            await self._send_detection_frames(task, text="Hello?")
            await asyncio.wait_for(conv_detected.wait(), timeout=5.0)
            await task.queue_frame(EndFrame())

        await self._run_pipeline(detector, send)
        self.assertEqual(events, ["voicemail", "conversation"])

    async def test_multiple_resets(self) -> None:
        """Three sequential resets, each followed by detection, all fire correctly."""
        detector = self._create_detector(["CONVERSATION", "VOICEMAIL", "CONVERSATION"])

        events: List[str] = []
        detection_events = [asyncio.Event() for _ in range(3)]
        detection_idx = 0

        async def on_conv(processor: FrameProcessor) -> None:
            nonlocal detection_idx
            events.append("conversation")
            detection_events[detection_idx].set()
            detection_idx += 1

        async def on_vm(processor: FrameProcessor) -> None:
            nonlocal detection_idx
            events.append("voicemail")
            detection_events[detection_idx].set()
            detection_idx += 1

        detector.add_event_handler("on_conversation_detected", on_conv)
        detector.add_event_handler("on_voicemail_detected", on_vm)

        async def send(task: PipelineTask) -> None:
            # First detection: CONVERSATION
            await self._send_detection_frames(task, text="Hello?")
            await asyncio.wait_for(detection_events[0].wait(), timeout=5.0)

            # Reset and second detection: VOICEMAIL
            await detector.reset()
            await self._send_detection_frames(task, text="You've reached the voicemail of John.")
            await asyncio.wait_for(detection_events[1].wait(), timeout=5.0)

            # Reset and third detection: CONVERSATION
            await detector.reset()
            await self._send_detection_frames(task, text="Yeah, who is this?")
            await asyncio.wait_for(detection_events[2].wait(), timeout=5.0)

            await task.queue_frame(EndFrame())

        await self._run_pipeline(detector, send)
        self.assertEqual(events, ["conversation", "voicemail", "conversation"])

    async def test_reset_tts_gating_true(self) -> None:
        """After reset with tts_gating=True, TTS frames are buffered during classification."""
        detector = self._create_detector(["CONVERSATION", "CONVERSATION"])
        collector = FrameCollector()

        first_detected = asyncio.Event()
        second_detected = asyncio.Event()

        async def on_conv(processor: FrameProcessor) -> None:
            if not first_detected.is_set():
                first_detected.set()
            else:
                second_detected.set()

        detector.add_event_handler("on_conversation_detected", on_conv)

        async def send(task: PipelineTask) -> None:
            # First detection: CONVERSATION
            await self._send_detection_frames(task, text="Hello?")
            await asyncio.wait_for(first_detected.wait(), timeout=5.0)

            # Reset with TTS gating enabled
            await detector.reset(tts_gating=True)
            collector.frames.clear()

            # Send TTS frame — should be buffered by the gate
            tts_frame = TTSAudioRawFrame(audio=b"\x00\x00" * 160, sample_rate=16000, num_channels=1)
            await task.queue_frame(tts_frame)
            await asyncio.sleep(0.1)

            # TTS frame should NOT have passed through yet
            tts_in_output = [f for f in collector.frames if isinstance(f, TTSAudioRawFrame)]
            self.assertEqual(len(tts_in_output), 0, "TTS frame should be buffered")

            # Now trigger second detection to release the buffer
            await self._send_detection_frames(task, text="Hi there")
            await asyncio.wait_for(second_detected.wait(), timeout=5.0)
            await asyncio.sleep(0.1)

            # TTS frame should now have been released
            tts_in_output = [f for f in collector.frames if isinstance(f, TTSAudioRawFrame)]
            self.assertEqual(len(tts_in_output), 1, "TTS frame should be released after detection")

            await task.queue_frame(EndFrame())

        await self._run_pipeline(detector, send, collector=collector)

    async def test_reset_tts_gating_false(self) -> None:
        """After reset with tts_gating=False, TTS frames flow through immediately."""
        detector = self._create_detector(["CONVERSATION", "CONVERSATION"])
        collector = FrameCollector()

        first_detected = asyncio.Event()

        async def on_conv(processor: FrameProcessor) -> None:
            first_detected.set()

        detector.add_event_handler("on_conversation_detected", on_conv)

        async def send(task: PipelineTask) -> None:
            # First detection: CONVERSATION
            await self._send_detection_frames(task, text="Hello?")
            await asyncio.wait_for(first_detected.wait(), timeout=5.0)

            # Reset with TTS gating disabled
            await detector.reset(tts_gating=False)
            collector.frames.clear()

            # Send TTS frame — should flow through immediately
            tts_frame = TTSAudioRawFrame(audio=b"\x00\x00" * 160, sample_rate=16000, num_channels=1)
            await task.queue_frame(tts_frame)
            await asyncio.sleep(0.1)

            # TTS frame should have passed through immediately
            tts_in_output = [f for f in collector.frames if isinstance(f, TTSAudioRawFrame)]
            self.assertEqual(len(tts_in_output), 1, "TTS frame should flow through immediately")

            await task.queue_frame(EndFrame())

        await self._run_pipeline(detector, send, collector=collector)


if __name__ == "__main__":
    unittest.main()
