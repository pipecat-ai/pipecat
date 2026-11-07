#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Resemble AI Detect processor implementations."""

from __future__ import annotations

import asyncio
import inspect
import io
import math
import os
import struct
import time
import wave
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import aiohttp
import numpy as np
from loguru import logger

from pipecat.audio.resamplers.base_audio_resampler import BaseAudioResampler
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import CancelFrame, EndFrame, Frame, InputAudioRawFrame, StartFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

RESEMBLE_DETECT_API_URL = "https://app.resemble.ai/api/v2"

DETECT_SAMPLE_RATE = 16000
"""Sample rate (Hz) that audio is normalized to before analysis."""

DetectionMode = Literal["sampled", "first_n", "continuous"]
DetectionLabel = Literal["real", "fake", "inconclusive"]
DetectionNormalizedLabel = Literal["real", "synthetic", "inconclusive"]
DetectionSecurity = Literal["spot", "standard", "high"]
DetectionAction = Literal["continue", "watch", "verify", "block"]
DetectFormFieldValue = str | int | float | bool


class DetectTransport(Protocol):
    """Transport used by :class:`ResembleDetect` to submit one audio window."""

    async def submit(
        self,
        pcm16: bytes,
        *,
        frame_length: int,
        request_timeout: float,
    ) -> dict[str, Any]:
        """Submit mono 16 kHz PCM16 audio and return a completed Detect item payload."""


@dataclass(frozen=True)
class _SecurityPreset:
    mode: DetectionMode
    samples: int
    sample_interval_seconds: float
    analysis_budget_seconds: float
    agreement_window: int
    min_fake_results: int


_SECURITY_PRESETS: dict[DetectionSecurity, _SecurityPreset] = {
    "spot": _SecurityPreset(
        mode="sampled",
        samples=1,
        sample_interval_seconds=0.0,
        analysis_budget_seconds=4.0,
        agreement_window=1,
        min_fake_results=1,
    ),
    "standard": _SecurityPreset(
        mode="sampled",
        samples=3,
        sample_interval_seconds=30.0,
        analysis_budget_seconds=16.0,
        agreement_window=3,
        min_fake_results=2,
    ),
    "high": _SecurityPreset(
        mode="continuous",
        samples=3,
        sample_interval_seconds=0.0,
        analysis_budget_seconds=0.0,
        agreement_window=3,
        min_fake_results=2,
    ),
}

_DEFAULT_WINDOW_SECONDS = 4.0
_DEFAULT_FAKE_THRESHOLD = 0.7
_DEFAULT_LIKELY_SYNTHETIC_THRESHOLD = 0.5
_DEFAULT_UNCLEAR_THRESHOLD = 0.3


@dataclass
class DetectionResult:
    """Outcome of analyzing one audio window with Resemble Detect."""

    label: DetectionLabel
    """``"fake"`` or ``"real"`` as returned by the API."""
    aggregated_score: float
    """Aggregated fake-probability for the window (0.0 = real, 1.0 = fake)."""
    scores: list[float]
    """Per-frame fake-probabilities inside the window."""
    consistency: float | None
    """Model consistency metric for the window, when provided by the API."""
    window_index: int
    """0-based index of the analyzed window within this processor's stream."""
    window_start: float
    """Start of the analyzed window, in seconds since processing began."""
    window_end: float
    """End of the analyzed window, in seconds since processing began."""
    participant_identity: str | None
    """Optional identity for the monitored participant."""
    detection_uuid: str | None
    """UUID of the detection job on Resemble's side."""
    latency: float
    """Round-trip seconds spent on the API call for this window."""
    forced: bool = False
    """True if this result came from :meth:`ResembleDetect.check_now`."""
    raw: dict[str, Any] = field(repr=False, default_factory=dict)
    """Raw item payload returned by the API."""

    @property
    def score(self) -> float:
        """Return the fake/synthetic probability in ``[0, 1]``."""
        return self.aggregated_score

    @property
    def normalized_label(self) -> DetectionNormalizedLabel:
        """Return a developer-facing label with ``"synthetic"`` instead of ``"fake"``."""
        return _normalize_label(self.label)

    @property
    def confidence(self) -> float:
        """Return the best-effort confidence in ``[0, 1]``."""
        if self.consistency is not None:
            return _clamp01(self.consistency / 100 if self.consistency > 1 else self.consistency)
        return _clamp01(abs(self.aggregated_score - 0.5) * 2)

    @property
    def scan_index(self) -> int:
        """Return the 1-based scan index for UI/event payloads."""
        return self.window_index + 1

    @property
    def window_ts(self) -> float:
        """Return the end timestamp of the analyzed window."""
        return self.window_end

    @property
    def is_final(self) -> bool:
        """Return whether this payload is final."""
        return False

    @property
    def recommended_action(self) -> DetectionAction:
        """Return the recommended action band for this result."""
        if self.aggregated_score < _DEFAULT_UNCLEAR_THRESHOLD:
            return "continue"
        if self.aggregated_score < _DEFAULT_LIKELY_SYNTHETIC_THRESHOLD:
            return "watch"
        if self.aggregated_score < _DEFAULT_FAKE_THRESHOLD:
            return "verify"
        return "block"

    def to_dict(self) -> dict[str, Any]:
        """Return a small, stable payload suitable for app events or dashboards."""
        return {
            "label": self.normalized_label,
            "raw_label": self.label,
            "score": self.score,
            "confidence": self.confidence,
            "window_ts": self.window_ts,
            "scan_index": self.scan_index,
            "is_final": self.is_final,
            "recommended_action": self.recommended_action,
            "participant_identity": self.participant_identity,
            "detection_uuid": self.detection_uuid,
            "latency": self.latency,
            "forced": self.forced,
        }


@dataclass
class DetectionVerdict:
    """Aggregate verdict across analyzed windows."""

    label: DetectionLabel
    """Final aggregate label."""
    max_score: float | None
    """Highest window score observed."""
    analyzed_seconds: float
    """Total seconds of speech submitted for analysis."""
    results: list[DetectionResult]
    """Per-window results that informed the verdict."""

    @property
    def score(self) -> float:
        """Return the highest fake/synthetic probability observed."""
        return self.max_score or 0.0

    @property
    def normalized_label(self) -> DetectionNormalizedLabel:
        """Return a developer-facing label with ``"synthetic"`` instead of ``"fake"``."""
        return _normalize_label(self.label)

    @property
    def confidence(self) -> float:
        """Return confidence from the highest-scoring result."""
        if not self.results:
            return 0.0
        result = max(self.results, key=lambda r: r.aggregated_score)
        return result.confidence

    @property
    def scan_index(self) -> int:
        """Return the 1-based index of the latest scan represented by this verdict."""
        return len(self.results)

    @property
    def window_ts(self) -> float:
        """Return the end timestamp of the latest analyzed window."""
        if not self.results:
            return 0.0
        return self.results[-1].window_end

    @property
    def is_final(self) -> bool:
        """Return whether this payload is final."""
        return True

    def to_dict(self) -> dict[str, Any]:
        """Return a small, stable payload suitable for app events or dashboards."""
        return {
            "label": self.normalized_label,
            "raw_label": self.label,
            "score": self.score,
            "confidence": self.confidence,
            "window_ts": self.window_ts,
            "scan_index": self.scan_index,
            "is_final": self.is_final,
            "analyzed_seconds": self.analyzed_seconds,
        }


@dataclass
class _DetectOptions:
    security: DetectionSecurity
    window_seconds: float
    mode: DetectionMode
    analysis_budget_seconds: float
    samples: int
    sample_interval_seconds: float
    fake_threshold: float
    likely_synthetic_threshold: float
    unclear_threshold: float
    agreement_window: int
    min_fake_results: int
    force_immediate_fake: bool
    frame_length: int
    silence_rms_threshold: float
    request_timeout: float


class ResembleDetect(FrameProcessor):
    """Real-time deepfake detection processor powered by Resemble Detect.

    The processor passes every Pipecat frame through unchanged while listening to
    downstream :class:`InputAudioRawFrame` frames. Audio is normalized to mono
    16 kHz PCM, buffered into short windows, and submitted in background tasks.

    Event handlers available:

    - ``on_result``: one audio window was analyzed.
    - ``on_fake_detected``: a raw window crossed ``fake_threshold``.
    - ``on_synthetic_detected``: the security policy confirmed a synthetic speaker.
    - ``on_verdict``: the configured ambient analysis budget completed or the pipeline ended.
    """

    def __init__(
        self,
        *,
        security: DetectionSecurity = "standard",
        api_key: str | None = None,
        base_url: str = RESEMBLE_DETECT_API_URL,
        window_seconds: float | None = None,
        mode: DetectionMode | None = None,
        samples: int | None = None,
        sample_interval_seconds: float | None = None,
        analysis_budget_seconds: float | None = None,
        fake_threshold: float = _DEFAULT_FAKE_THRESHOLD,
        likely_synthetic_threshold: float = _DEFAULT_LIKELY_SYNTHETIC_THRESHOLD,
        unclear_threshold: float = _DEFAULT_UNCLEAR_THRESHOLD,
        agreement_window: int | None = None,
        min_fake_results: int | None = None,
        force_immediate_fake: bool = False,
        frame_length: int = 2,
        silence_rms_threshold: float = 0.0035,
        request_timeout: float = 30.0,
        http_session: aiohttp.ClientSession | None = None,
        transport: DetectTransport | None = None,
        zero_retention_mode: bool = True,
        extra_form_fields: Mapping[str, DetectFormFieldValue] | None = None,
        participant_identity: str | None = None,
        resampler: BaseAudioResampler | None = None,
        **kwargs,
    ) -> None:
        """Create a Resemble Detect processor.

        Args:
            security: Detection preset. ``"spot"`` performs one low-cost check,
                ``"standard"`` uses sampled 2-of-3 agreement, and ``"high"``
                monitors continuously.
            api_key: Resemble API key. If omitted, ``RESEMBLE_API_KEY`` is read
                from the environment. Not required when ``transport`` is provided.
            base_url: Resemble REST Detect API base URL.
            window_seconds: Seconds of normalized audio per detection request.
            mode: Detection mode override: ``"sampled"``, ``"first_n"``, or
                ``"continuous"``.
            samples: Number of ambient samples in sampled mode.
            sample_interval_seconds: Gap between ambient samples in sampled mode.
            analysis_budget_seconds: Speech budget for first-n mode.
            fake_threshold: Score at or above which a window is suspicious.
            likely_synthetic_threshold: Score band used in aggregate verdicts.
            unclear_threshold: Score band used in aggregate verdicts.
            agreement_window: Number of recent checks considered for agreement.
            min_fake_results: Suspicious results needed within ``agreement_window``.
            force_immediate_fake: Allow a forced check to confirm immediately.
            frame_length: Detect analysis sub-window length in seconds, from 1 to 4.
            silence_rms_threshold: Normalized RMS below which ambient windows are skipped.
            request_timeout: Per-request timeout in seconds.
            http_session: Existing aiohttp session for the default transport.
            transport: Custom Detect transport.
            zero_retention_mode: Enable zero-retention mode on the default transport.
            extra_form_fields: Extra form fields sent by the default transport.
            participant_identity: Optional identity surfaced in result payloads.
            resampler: Audio resampler used to normalize sample rate.
            **kwargs: Additional arguments passed to :class:`FrameProcessor`.
        """
        super().__init__(**kwargs)

        if security not in _SECURITY_PRESETS:
            raise ValueError(f"security must be one of {sorted(_SECURITY_PRESETS)}")

        preset = _SECURITY_PRESETS[security]
        resolved_window_seconds = (
            window_seconds if window_seconds is not None else _DEFAULT_WINDOW_SECONDS
        )
        resolved_mode = mode if mode is not None else preset.mode
        resolved_samples = samples if samples is not None else preset.samples
        resolved_sample_interval_seconds = (
            sample_interval_seconds
            if sample_interval_seconds is not None
            else preset.sample_interval_seconds
        )
        resolved_analysis_budget_seconds = (
            analysis_budget_seconds
            if analysis_budget_seconds is not None
            else preset.analysis_budget_seconds
        )
        resolved_agreement_window = (
            agreement_window if agreement_window is not None else preset.agreement_window
        )
        resolved_min_fake_results = (
            min_fake_results if min_fake_results is not None else preset.min_fake_results
        )

        if resolved_mode not in ("sampled", "first_n", "continuous"):
            raise ValueError("mode must be one of 'sampled', 'first_n', or 'continuous'")
        if resolved_window_seconds < 2.0:
            raise ValueError("window_seconds must be >= 2.0 (Detect scores 1-4s frames)")
        if not 1 <= frame_length <= 4:
            raise ValueError("frame_length must be between 1 and 4")
        if resolved_samples < 1:
            raise ValueError("samples must be >= 1")
        if resolved_sample_interval_seconds < 0:
            raise ValueError("sample_interval_seconds must be >= 0")
        if resolved_mode == "first_n" and resolved_analysis_budget_seconds <= 0:
            raise ValueError("analysis_budget_seconds must be > 0 in first_n mode")
        _validate_threshold("fake_threshold", fake_threshold)
        _validate_threshold("likely_synthetic_threshold", likely_synthetic_threshold)
        _validate_threshold("unclear_threshold", unclear_threshold)
        if not unclear_threshold <= likely_synthetic_threshold <= fake_threshold:
            raise ValueError(
                "thresholds must satisfy unclear_threshold <= likely_synthetic_threshold "
                "<= fake_threshold"
            )
        if resolved_agreement_window < 1:
            raise ValueError("agreement_window must be >= 1")
        if resolved_min_fake_results < 1:
            raise ValueError("min_fake_results must be >= 1")
        if resolved_min_fake_results > resolved_agreement_window:
            raise ValueError("min_fake_results must be <= agreement_window")

        if transport is None:
            api_key = api_key or os.environ.get("RESEMBLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "Resemble API key is required, either as argument or set "
                    "RESEMBLE_API_KEY environment variable"
                )
            transport = RestDetectTransport(
                api_key=api_key,
                base_url=base_url,
                http_session=http_session,
                zero_retention_mode=zero_retention_mode,
                extra_form_fields=extra_form_fields,
            )

        self._opts = _DetectOptions(
            security=security,
            window_seconds=resolved_window_seconds,
            mode=resolved_mode,
            analysis_budget_seconds=resolved_analysis_budget_seconds,
            samples=resolved_samples,
            sample_interval_seconds=resolved_sample_interval_seconds,
            fake_threshold=fake_threshold,
            likely_synthetic_threshold=likely_synthetic_threshold,
            unclear_threshold=unclear_threshold,
            agreement_window=resolved_agreement_window,
            min_fake_results=resolved_min_fake_results,
            force_immediate_fake=force_immediate_fake,
            frame_length=frame_length,
            silence_rms_threshold=silence_rms_threshold,
            request_timeout=request_timeout,
        )

        self._transport = transport
        self._participant_identity = participant_identity
        self._resampler = resampler or create_stream_resampler()
        self._analysis_tasks: set[asyncio.Task[Any]] = set()

        self._buffer = bytearray()
        self._stream_pos = 0.0
        self._cooldown_until = 0.0
        self._results: list[DetectionResult] = []
        self._analyzed_seconds = 0.0
        self._window_index = 0
        self._samples_taken = 0
        self._verdict_emitted = False
        self._synthetic_alert_emitted = False
        self._paused = False
        self._force_pending = False

        self._register_event_handler("on_result")
        self._register_event_handler("on_fake_detected")
        self._register_event_handler("on_synthetic_detected")
        self._register_event_handler("on_verdict")

    async def cleanup(self):
        """Clean up background analyses and owned transport resources."""
        await self._cancel_analysis_tasks()
        close = getattr(self._transport, "close", None)
        if callable(close):
            maybe_awaitable = close()
            if inspect.isawaitable(maybe_awaitable):
                await maybe_awaitable
        await super().cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process one frame while passing it through unchanged."""
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            self._reset_stream_state()

        if isinstance(frame, EndFrame):
            await self._flush_and_emit_verdict()

        await self.push_frame(frame, direction)

        if isinstance(frame, CancelFrame):
            await self._cancel_analysis_tasks()
            return

        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, InputAudioRawFrame):
            await self._handle_audio(frame)

    def check_now(self) -> None:
        """Force an immediate spot-check of the next full speech window."""
        self._force_pending = True

    def pause(self) -> None:
        """Pause collection and drop buffered audio until :meth:`resume` is called."""
        self._paused = True
        self._buffer.clear()

    def resume(self) -> None:
        """Resume collection after :meth:`pause`."""
        self._paused = False

    @property
    def paused(self) -> bool:
        """Return whether collection is paused."""
        return self._paused

    @property
    def security(self) -> DetectionSecurity:
        """Return the configured security preset."""
        return self._opts.security

    @property
    def samples(self) -> int:
        """Return the configured number of ambient samples for sampled mode."""
        return self._opts.samples

    @property
    def window_seconds(self) -> float:
        """Return the seconds of audio submitted per detection request."""
        return self._opts.window_seconds

    @property
    def results(self) -> list[DetectionResult]:
        """Return all per-window results collected so far."""
        return list(self._results)

    @property
    def verdict(self) -> DetectionVerdict:
        """Return the aggregate verdict over everything analyzed so far."""
        max_score = max((r.aggregated_score for r in self._results), default=None)
        if max_score is None:
            label: DetectionLabel = "inconclusive"
        elif self._synthetic_alert_emitted or self._has_confirmed_fake():
            label = "fake"
        elif max_score >= self._opts.unclear_threshold:
            label = "inconclusive"
        else:
            label = "real"
        return DetectionVerdict(
            label=label,
            max_score=max_score,
            analyzed_seconds=self._analyzed_seconds,
            results=self.results,
        )

    async def _handle_audio(self, frame: InputAudioRawFrame) -> None:
        if self._paused:
            self._buffer.clear()
            return

        self._buffer.extend(await self._normalize_audio(frame))
        window_bytes = int(self._opts.window_seconds * DETECT_SAMPLE_RATE) * 2

        while len(self._buffer) >= window_bytes:
            window = bytes(self._buffer[:window_bytes])
            del self._buffer[:window_bytes]
            window_start = self._stream_pos
            self._stream_pos += self._opts.window_seconds
            await self._maybe_analyze_window(window, window_start=window_start)

    async def _maybe_analyze_window(self, window: bytes, *, window_start: float) -> None:
        forced = self._force_pending

        if self._opts.mode == "sampled" and not forced:
            if self._stream_pos <= self._cooldown_until:
                return
            if self._samples_taken >= self._opts.samples:
                return

        if self._budget_exhausted():
            if self._opts.mode == "first_n":
                await self._flush_and_emit_verdict()
            return

        if self._is_silence(window) and not forced:
            logger.debug("skipping silent Detect window")
            return

        self._analyzed_seconds += self._opts.window_seconds
        index = self._window_index
        self._window_index += 1
        self._spawn_analysis(
            self._analyze_window(
                window,
                index=index,
                window_start=window_start,
                participant_identity=self._participant_identity,
                forced=forced,
            )
        )

        if self._opts.mode == "sampled":
            if forced:
                self._force_pending = False
            else:
                self._samples_taken += 1
                self._cooldown_until = self._stream_pos + self._opts.sample_interval_seconds
                if self._samples_taken >= self._opts.samples:
                    await self._flush_and_emit_verdict()

    def _spawn_analysis(self, coro: Any) -> None:
        task = self.create_task(coro)
        self._analysis_tasks.add(task)
        task.add_done_callback(self._analysis_tasks.discard)

    async def _flush_and_emit_verdict(self) -> None:
        while self._analysis_tasks:
            await asyncio.gather(*list(self._analysis_tasks), return_exceptions=True)
        await self._emit_verdict()

    async def _emit_verdict(self) -> None:
        if self._verdict_emitted:
            return
        self._verdict_emitted = True
        await self._call_event_handler("on_verdict", self.verdict)

    def _budget_exhausted(self) -> bool:
        return (
            self._opts.mode == "first_n"
            and self._analyzed_seconds >= self._opts.analysis_budget_seconds
        )

    async def _normalize_audio(self, frame: InputAudioRawFrame) -> bytes:
        mono = _downmix_to_mono(frame.audio, frame.num_channels)
        return await self._resampler.resample(mono, frame.sample_rate, DETECT_SAMPLE_RATE)

    def _is_silence(self, pcm16: bytes) -> bool:
        if self._opts.silence_rms_threshold <= 0:
            return False
        samples = struct.unpack(f"<{len(pcm16) // 2}h", pcm16)
        rms = math.sqrt(sum(s * s for s in samples) / len(samples)) / 32768.0
        return rms < self._opts.silence_rms_threshold

    async def _analyze_window(
        self,
        pcm16: bytes,
        *,
        index: int,
        window_start: float,
        participant_identity: str | None,
        forced: bool = False,
    ) -> None:
        try:
            started = time.monotonic()
            item = await self._submit(pcm16)
            latency = time.monotonic() - started
        except Exception as exc:
            logger.warning("resemble detect request failed", exc_info=exc)
            await self.push_error("Resemble Detect request failed", exception=exc)
            return

        metrics = item.get("metrics") or {}
        try:
            scores = [float(s) for s in metrics.get("score") or [] if s is not None]
            aggregated = float(metrics["aggregated_score"])
            label: DetectionLabel = "fake" if metrics["label"] == "fake" else "real"
        except (KeyError, TypeError, ValueError):
            await self.push_error(f"Unexpected Resemble Detect response shape: {item}")
            return

        result = DetectionResult(
            label=label,
            aggregated_score=aggregated,
            scores=scores,
            consistency=_opt_float(metrics.get("consistency")),
            window_index=index,
            window_start=window_start,
            window_end=window_start + self._opts.window_seconds,
            participant_identity=participant_identity,
            detection_uuid=item.get("uuid"),
            latency=latency,
            forced=forced,
            raw=item,
        )
        self._results.append(result)
        logger.debug(
            "detect window analyzed",
            extra={
                "label": result.label,
                "aggregated_score": result.aggregated_score,
                "window_index": index,
                "latency": round(latency, 3),
            },
        )
        await self._call_event_handler("on_result", result)
        if aggregated >= self._opts.fake_threshold:
            await self._call_event_handler("on_fake_detected", result)
        if self._should_emit_synthetic_detected(result):
            self._synthetic_alert_emitted = True
            await self._call_event_handler("on_synthetic_detected", result)

    async def _submit(self, pcm16: bytes) -> dict[str, Any]:
        return await self._transport.submit(
            pcm16,
            frame_length=self._opts.frame_length,
            request_timeout=self._opts.request_timeout,
        )

    def _has_confirmed_fake(self) -> bool:
        if not self._results:
            return False
        recent = self._results[-self._opts.agreement_window :]
        fake_results = [r for r in recent if r.aggregated_score >= self._opts.fake_threshold]
        return len(fake_results) >= self._opts.min_fake_results

    def _should_emit_synthetic_detected(self, result: DetectionResult) -> bool:
        if self._synthetic_alert_emitted:
            return False
        if result.aggregated_score < self._opts.fake_threshold:
            return False
        if result.forced and self._opts.force_immediate_fake:
            return True
        return self._has_confirmed_fake()

    def _reset_stream_state(self) -> None:
        self._buffer.clear()
        self._stream_pos = 0.0
        self._cooldown_until = 0.0
        self._results.clear()
        self._analyzed_seconds = 0.0
        self._window_index = 0
        self._samples_taken = 0
        self._verdict_emitted = False
        self._synthetic_alert_emitted = False
        self._force_pending = False

    async def _cancel_analysis_tasks(self) -> None:
        for task in list(self._analysis_tasks):
            await self.cancel_task(task)
        self._analysis_tasks.clear()


class RestDetectTransport:
    """Default transport for Resemble Detect's REST API."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = RESEMBLE_DETECT_API_URL,
        http_session: aiohttp.ClientSession | None = None,
        zero_retention_mode: bool = True,
        extra_form_fields: Mapping[str, DetectFormFieldValue] | None = None,
    ) -> None:
        """Create a REST transport.

        Args:
            api_key: Resemble API key.
            base_url: Resemble REST Detect API base URL.
            http_session: Existing aiohttp session. If omitted, the transport creates one.
            zero_retention_mode: Enable Detect's zero-retention mode.
            extra_form_fields: Extra form fields to include in every Detect request.
        """
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._session = http_session
        self._owns_session = http_session is None
        self._extra_form_fields = {
            "zero_retention_mode": _form_value(zero_retention_mode),
            **{key: _form_value(value) for key, value in (extra_form_fields or {}).items()},
        }

    async def submit(
        self,
        pcm16: bytes,
        *,
        frame_length: int,
        request_timeout: float,
    ) -> dict[str, Any]:
        """Submit mono 16 kHz PCM16 audio and return a completed Detect item payload."""
        form = aiohttp.FormData()
        form.add_field(
            "file",
            _wav_bytes(pcm16),
            filename="window.wav",
            content_type="audio/wav",
        )
        form.add_field("modality", "audio")
        form.add_field("frame_length", str(frame_length))
        for key, value in self._extra_form_fields.items():
            form.add_field(key, value)

        async with self._ensure_session().post(
            f"{self._base_url}/detect",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Prefer": "wait",
            },
            data=form,
            timeout=aiohttp.ClientTimeout(total=request_timeout),
        ) as resp:
            if resp.status not in (200, 201):
                body = await resp.text()
                raise RuntimeError(f"Resemble Detect request failed ({resp.status}): {body[:500]}")
            payload = await resp.json()

        item: dict[str, Any] = payload.get("item") or {}
        if item.get("status") == "completed":
            return item

        uuid = item.get("uuid")
        if not uuid:
            raise RuntimeError(f"Resemble Detect response missing uuid: {payload}")
        return await self._poll(uuid, request_timeout=request_timeout)

    async def close(self) -> None:
        """Close the owned aiohttp session, if one was created by this transport."""
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()

    async def _poll(self, uuid: str, *, request_timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + request_timeout
        while time.monotonic() < deadline:
            await asyncio.sleep(1.0)
            async with self._ensure_session().get(
                f"{self._base_url}/detect/{uuid}",
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=aiohttp.ClientTimeout(total=request_timeout),
            ) as resp:
                if resp.status != 200:
                    continue
                payload = await resp.json()
            item: dict[str, Any] = payload.get("item") or {}
            if item.get("status") == "completed":
                return item
        raise RuntimeError(f"Resemble Detect job {uuid} did not complete in time")

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = aiohttp.ClientSession()

        return self._session


def _downmix_to_mono(audio: bytes, num_channels: int) -> bytes:
    if num_channels < 1:
        raise ValueError("num_channels must be >= 1")
    if len(audio) % (num_channels * 2) != 0:
        raise ValueError("input audio must be 16-bit PCM aligned")
    if num_channels == 1:
        return audio

    samples = np.frombuffer(audio, dtype=np.int16)
    frames = samples.reshape(-1, num_channels)
    mono = frames.astype(np.int32).mean(axis=1).astype(np.int16)
    return mono.tobytes()


def _wav_bytes(pcm16: bytes) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(DETECT_SAMPLE_RATE)
        wf.writeframes(pcm16)
    return out.getvalue()


def _opt_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_label(label: DetectionLabel) -> DetectionNormalizedLabel:
    if label == "fake":
        return "synthetic"
    return label


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _validate_threshold(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def _form_value(value: DetectFormFieldValue) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
