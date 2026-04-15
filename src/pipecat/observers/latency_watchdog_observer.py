#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Type

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    MetricsFrame,
    TTSSpeakFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.llm_service import LLMService
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService


@dataclass
class _Subscription:
    start_processor: Optional[Type[FrameProcessor]]
    start_frame: Optional[Type[Frame]]
    stop_processor: Optional[Type[FrameProcessor]]
    stop_frame: Optional[Type[Frame]]
    threshold_secs: float
    handler: Callable


class LatencyWatchdogObserver(BaseObserver):
    """Observer that proactively monitors processing latency.

    It arms a timer when a specific input frame is pushed to a processor and
    disarms it when a TTFBMetricsData (Time to First Byte) is received from that
    same processor.

    If the timer expires before disarming, a registered callback is fired.
    """

    _DEFAULT_INPUT_FRAMES: Dict[Type[FrameProcessor], Type[Frame]] = {
        LLMService: LLMContextFrame,
        TTSService: TTSSpeakFrame,
        STTService: VADUserStoppedSpeakingFrame,
    }

    def __init__(self, *, cooldown_secs: float = 5.0, **kwargs):
        super().__init__(**kwargs)
        self._subscriptions: List[_Subscription] = []
        self._pending: Dict[Tuple[int, int], asyncio.TimerHandle] = {}
        self._last_fire: Dict[Tuple[int, int], float] = {}
        self._cooldown_secs = cooldown_secs

    def subscribe(
        self,
        processor_type: Optional[Type[FrameProcessor]] = None,
        *,
        threshold_secs: float,
        input_frame: Optional[Type[Frame]] = None,
        start_frame: Optional[Type[Frame]] = None,
        start_processor: Optional[Type[FrameProcessor]] = None,
        stop_frame: Optional[Type[Frame]] = None,
        stop_processor: Optional[Type[FrameProcessor]] = None,
    ) -> Callable:
        """Register a callback fired when a latency threshold is exceeded.

        This method supports both simple per-processor monitoring (backward compatible)
        and complex spanning or global monitoring.

        Args:
            processor_type: (Legacy) FrameProcessor subclass to watch.
            threshold_secs: Fire callback if threshold exceeded.
            input_frame: (Legacy) Alias for start_frame.
            start_frame: Frame type that signals the start of measurement.
            start_processor: Optional processor type that must receive the start_frame.
            stop_frame: Frame type that signals the end of measurement. None = TTFBMetricsData.
            stop_processor: Optional processor type that must emit the stop_signal.

        Example (Spanning)::

            @watchdog.subscribe(start_frame=VADUserStoppedSpeakingFrame, stop_processor=TTSService, threshold_secs=1.5)
            async def on_slow(name, elapsed):
                await task.queue_frame(TTSSpeakFrame("I'm still working on it..."))
        """

        # Backward compatibility logic
        final_start_processor = start_processor or processor_type
        final_stop_processor = stop_processor or (processor_type if stop_frame is None else None)
        final_start_frame = start_frame or input_frame

        def decorator(func: Callable) -> Callable:
            self._subscriptions.append(
                _Subscription(
                    start_processor=final_start_processor,
                    start_frame=final_start_frame,
                    stop_processor=final_stop_processor,
                    stop_frame=stop_frame,
                    threshold_secs=threshold_secs,
                    handler=func,
                )
            )
            return func

        return decorator

    async def on_push_frame(self, data: FramePushed) -> None:
        frame = data.frame

        # 1. Cancel all on interruption/end
        if isinstance(frame, (InterruptionFrame, CancelFrame, EndFrame)):
            self._cancel_all()
            return

        # 2. DISARM
        if isinstance(frame, MetricsFrame):
            for metrics_data in frame.data:
                if isinstance(metrics_data, TTFBMetricsData):
                    self._disarm(data.source, None)
        else:
            self._disarm(data.source, type(frame))

        # 3. ARM
        destination = data.destination
        if destination is None:
            return

        for idx, sub in enumerate(self._subscriptions):
            expected_start = self._resolve_start_frame(sub, destination)
            if expected_start is None or not isinstance(frame, expected_start):
                continue

            if sub.start_processor is not None and not isinstance(destination, sub.start_processor):
                continue

            # Keying: (sub_idx, proc_id)
            # proc_id is 0 if it's a global/spanning watchdog (single active timer per sub).
            # Otherwise it's id(processor) allowing multiple concurrent measurements for different processors.
            is_spanning = sub.start_processor != sub.stop_processor or sub.start_processor is None
            proc_id = 0 if is_spanning else id(destination)
            key = (idx, proc_id)

            if key in self._pending:
                continue  # IGNORE: already armed

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop()

            name = self._resolve_callback_name(sub, destination)
            handle = loop.call_later(
                sub.threshold_secs,
                lambda k=key, n=name, s=sub: asyncio.create_task(self._fire(k, n, s)),
            )
            self._pending[key] = handle

    def _disarm(self, source: FrameProcessor, frame_type: Optional[Type[Frame]]) -> None:
        # frame_type is None for TTFBMetricsData
        for idx, sub in enumerate(self._subscriptions):
            # Check stop frame
            if sub.stop_frame is None:
                if frame_type is not None:
                    continue
            elif frame_type != sub.stop_frame:
                continue

            # Check stop processor
            if sub.stop_processor is not None and not isinstance(source, sub.stop_processor):
                continue

            # Identify key to cancel
            is_spanning = sub.start_processor != sub.stop_processor or sub.start_processor is None
            proc_id = 0 if is_spanning else id(source)
            key = (idx, proc_id)

            handle = self._pending.pop(key, None)
            if handle:
                handle.cancel()

    def _resolve_start_frame(
        self, sub: _Subscription, processor: FrameProcessor
    ) -> Optional[Type[Frame]]:
        if sub.start_frame is not None:
            return sub.start_frame
        # Default fallback logic for legacy processor-based subscriptions
        if sub.start_processor is not None:
            for base_type, frame_type in self._DEFAULT_INPUT_FRAMES.items():
                if isinstance(processor, base_type):
                    return frame_type
        return None

    def _resolve_callback_name(self, sub: _Subscription, processor: FrameProcessor) -> str:
        if sub.start_processor and sub.stop_processor and sub.start_processor != sub.stop_processor:
            return f"{sub.start_processor.__name__} -> {sub.stop_processor.__name__}"
        if sub.start_processor and sub.stop_processor is None:
            return f"{sub.start_processor.__name__} -> Global"
        if sub.start_processor is None and sub.stop_processor:
            return f"Global -> {sub.stop_processor.__name__}"
        return processor.name

    async def _fire(self, key: Tuple[int, int], name: str, sub: _Subscription) -> None:
        self._pending.pop(key, None)
        now = asyncio.get_event_loop().time()
        if now - self._last_fire.get(key, 0.0) < self._cooldown_secs:
            return  # in cooldown
        self._last_fire[key] = now
        await sub.handler(name, sub.threshold_secs)

    def _cancel_all(self) -> None:
        for handle in self._pending.values():
            handle.cancel()
        self._pending.clear()
