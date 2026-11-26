#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import struct
import unittest
from typing import Any

from pipecat.frames.frames import EndFrame, InputAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.resembleai.detect import (
    DETECT_SAMPLE_RATE,
    DetectionResult,
    ResembleDetect,
)
from pipecat.tests.utils import run_test


class _FakeTransport:
    def __init__(self, *scores: float, consistency: float = 92.0) -> None:
        self._scores = list(scores)
        self._consistency = consistency
        self.calls: list[dict[str, Any]] = []

    async def submit(
        self,
        pcm16: bytes,
        *,
        frame_length: int,
        request_timeout: float,
    ) -> dict[str, Any]:
        index = len(self.calls)
        score = self._scores[index]
        self.calls.append(
            {
                "pcm16": pcm16,
                "frame_length": frame_length,
                "request_timeout": request_timeout,
            }
        )
        return {
            "uuid": f"detect-{index}",
            "status": "completed",
            "metrics": {
                "score": [score],
                "aggregated_score": score,
                "label": "fake" if score >= 0.7 else "real",
                "consistency": self._consistency,
            },
        }


def _audio_frame(seconds: float = 2.0, *, amplitude: int = 1200) -> InputAudioRawFrame:
    sample_count = int(DETECT_SAMPLE_RATE * seconds)
    audio = struct.pack(f"<{sample_count}h", *([amplitude] * sample_count))
    return InputAudioRawFrame(audio=audio, sample_rate=DETECT_SAMPLE_RATE, num_channels=1)


async def _drain_event_handlers() -> None:
    await asyncio.sleep(0)
    await asyncio.sleep(0)


class TestResembleDetectProcessor(unittest.IsolatedAsyncioTestCase):
    async def test_audio_frames_pass_through_and_result_event_fires(self) -> None:
        transport = _FakeTransport(0.82)
        processor = ResembleDetect(
            security="spot",
            transport=transport,
            window_seconds=2.0,
            silence_rms_threshold=0,
        )
        result_event = asyncio.Event()
        captured: dict[str, DetectionResult] = {}

        async def on_result(_, result: DetectionResult) -> None:
            captured["result"] = result
            result_event.set()

        processor.add_event_handler("on_result", on_result)

        down_frames, _ = await run_test(
            processor,
            frames_to_send=[_audio_frame()],
            expected_down_frames=[InputAudioRawFrame],
        )
        await asyncio.wait_for(result_event.wait(), timeout=1)

        self.assertEqual(len(down_frames), 1)
        self.assertEqual(captured["result"].normalized_label, "synthetic")
        self.assertEqual(captured["result"].score, 0.82)
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(len(transport.calls[0]["pcm16"]), DETECT_SAMPLE_RATE * 2 * 2)

    async def test_standard_requires_agreement_before_confirming_synthetic(self) -> None:
        transport = _FakeTransport(0.92, 0.05, 0.88)
        processor = ResembleDetect(transport=transport)
        raw_hits = []
        confirmed_hits = []
        processor.add_event_handler("on_fake_detected", lambda _, result: raw_hits.append(result))
        processor.add_event_handler(
            "on_synthetic_detected", lambda _, result: confirmed_hits.append(result)
        )

        await processor._analyze_window(
            b"\0\0",
            index=0,
            window_start=0.0,
            participant_identity="caller",
        )
        await _drain_event_handlers()

        self.assertEqual(len(raw_hits), 1)
        self.assertEqual(len(confirmed_hits), 0)
        self.assertEqual(processor.verdict.label, "inconclusive")

        await processor._analyze_window(
            b"\0\0",
            index=1,
            window_start=4.0,
            participant_identity="caller",
        )
        await processor._analyze_window(
            b"\0\0",
            index=2,
            window_start=8.0,
            participant_identity="caller",
        )
        await _drain_event_handlers()

        self.assertEqual(len(raw_hits), 2)
        self.assertEqual(len(confirmed_hits), 1)
        self.assertEqual(processor.verdict.label, "fake")
        self.assertEqual(processor.verdict.normalized_label, "synthetic")

    async def test_spot_confirms_from_one_fake_result(self) -> None:
        transport = _FakeTransport(0.91)
        processor = ResembleDetect(security="spot", transport=transport)
        confirmed_hits = []
        processor.add_event_handler(
            "on_synthetic_detected", lambda _, result: confirmed_hits.append(result)
        )

        await processor._analyze_window(
            b"\0\0",
            index=0,
            window_start=0.0,
            participant_identity="caller",
        )
        await _drain_event_handlers()

        self.assertEqual(len(confirmed_hits), 1)
        self.assertEqual(processor.verdict.label, "fake")

    async def test_forced_check_can_confirm_immediately_when_configured(self) -> None:
        transport = _FakeTransport(0.94)
        processor = ResembleDetect(transport=transport, force_immediate_fake=True)
        confirmed_hits = []
        processor.add_event_handler(
            "on_synthetic_detected", lambda _, result: confirmed_hits.append(result)
        )

        await processor._analyze_window(
            b"\0\0",
            index=0,
            window_start=0.0,
            participant_identity="caller",
            forced=True,
        )
        await _drain_event_handlers()

        self.assertEqual(len(confirmed_hits), 1)
        self.assertTrue(confirmed_hits[0].forced)

    async def test_final_verdict_preserves_confirmed_synthetic_alert(self) -> None:
        transport = _FakeTransport(0.91, 0.92, 0.02)
        processor = ResembleDetect(security="spot", transport=transport)
        confirmed_hits = []
        processor.add_event_handler(
            "on_synthetic_detected", lambda _, result: confirmed_hits.append(result)
        )

        await processor._analyze_window(
            b"\0\0",
            index=0,
            window_start=0.0,
            participant_identity="caller",
        )
        await processor._analyze_window(
            b"\0\0",
            index=1,
            window_start=4.0,
            participant_identity="caller",
        )
        await processor._analyze_window(
            b"\0\0",
            index=2,
            window_start=8.0,
            participant_identity="caller",
        )
        await _drain_event_handlers()

        self.assertEqual(len(confirmed_hits), 1)
        self.assertEqual(processor.verdict.label, "fake")

    async def test_result_payload_uses_stable_developer_shape(self) -> None:
        transport = _FakeTransport(0.82, consistency=91.0)
        processor = ResembleDetect(security="spot", transport=transport)

        await processor._analyze_window(
            b"\0\0",
            index=0,
            window_start=12.0,
            participant_identity="caller",
        )

        result = processor.results[0]
        self.assertEqual(result.normalized_label, "synthetic")
        self.assertEqual(result.score, 0.82)
        self.assertEqual(result.confidence, 0.91)
        self.assertEqual(result.scan_index, 1)
        self.assertEqual(result.window_ts, 16.0)
        self.assertEqual(result.recommended_action, "block")
        self.assertEqual(
            result.to_dict(),
            {
                "label": "synthetic",
                "raw_label": "fake",
                "score": 0.82,
                "confidence": 0.91,
                "window_ts": 16.0,
                "scan_index": 1,
                "is_final": False,
                "recommended_action": "block",
                "participant_identity": "caller",
                "detection_uuid": "detect-0",
                "latency": result.latency,
                "forced": False,
            },
        )

    async def test_custom_transport_receives_runtime_options(self) -> None:
        transport = _FakeTransport(0.1)
        processor = ResembleDetect(
            security="high",
            transport=transport,
            frame_length=3,
            request_timeout=12.0,
            sample_interval_seconds=11.0,
            agreement_window=2,
            min_fake_results=1,
        )

        await processor._analyze_window(
            b"\0\0",
            index=0,
            window_start=0.0,
            participant_identity="caller",
        )

        self.assertEqual(processor.security, "high")
        self.assertEqual(processor._opts.mode, "continuous")
        self.assertEqual(processor._opts.sample_interval_seconds, 11.0)
        self.assertEqual(transport.calls[0]["frame_length"], 3)
        self.assertEqual(transport.calls[0]["request_timeout"], 12.0)

    async def test_verdict_emits_on_end_frame_after_inflight_analysis(self) -> None:
        transport = _FakeTransport(0.1)
        processor = ResembleDetect(
            security="spot",
            transport=transport,
            window_seconds=2.0,
            silence_rms_threshold=0,
        )
        verdict_event = asyncio.Event()
        captured = {}

        async def on_verdict(_, verdict) -> None:
            captured["verdict"] = verdict
            verdict_event.set()

        processor.add_event_handler("on_verdict", on_verdict)

        await run_test(
            processor,
            frames_to_send=[_audio_frame(), EndFrame()],
            expected_down_frames=[InputAudioRawFrame, EndFrame],
            send_end_frame=False,
        )
        await asyncio.wait_for(verdict_event.wait(), timeout=1)

        self.assertEqual(captured["verdict"].label, "real")

    def test_invalid_explicit_overrides_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "samples must be >= 1"):
            ResembleDetect(transport=_FakeTransport(0.1), samples=0)

        with self.assertRaisesRegex(ValueError, "min_fake_results must be <= agreement_window"):
            ResembleDetect(
                transport=_FakeTransport(0.1),
                agreement_window=1,
                min_fake_results=2,
            )


if __name__ == "__main__":
    unittest.main()
