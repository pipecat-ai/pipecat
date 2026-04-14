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
    processor_type: Optional[Type[FrameProcessor]]
    input_frame: Optional[Type[Frame]]
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
    ) -> Callable:
        """Register a callback fired when a processor exceeds threshold_secs TTFB.

        Args:
            processor_type: FrameProcessor subclass to watch. None = all known types
                (LLMService, TTSService, STTService). Subclasses are matched via isinstance.
            threshold_secs: Fire callback if no TTFB measured within this duration.
            input_frame: Frame type that signals the start of processing for this processor.
                Defaults to the known type for LLMService/TTSService/STTService.
                Required for custom processor types not in the default table.

        Example::

            @watchdog.subscribe(LLMService, threshold_secs=0.7)
            async def on_slow_llm(processor_name, elapsed_secs):
                await task.queue_frame(TTSSpeakFrame("One moment..."))
        """

        def decorator(func: Callable) -> Callable:
            self._subscriptions.append(
                _Subscription(
                    processor_type=processor_type,
                    input_frame=input_frame,
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

        # 2. DISARM: TTFBMetricsData = first byte received -> cancel timers
        if isinstance(frame, MetricsFrame):
            for metrics_data in frame.data:
                if isinstance(metrics_data, TTFBMetricsData):
                    proc_id = id(data.source)
                    for idx in range(len(self._subscriptions)):
                        key = (proc_id, idx)
                        handle = self._pending.pop(key, None)
                        if handle:
                            handle.cancel()
            return

        # 3. ARM: input frame -> start timers
        destination = data.destination
        if destination is None:
            return

        for idx, sub in enumerate(self._subscriptions):
            # Resolve the expected input frame type for this processor
            expected_input = self._resolve_input_frame(sub, destination)
            if expected_input is None:
                continue

            if not isinstance(frame, expected_input):
                continue

            # Verify that the processor matches the desired type
            if sub.processor_type is not None and not isinstance(destination, sub.processor_type):
                continue

            key = (id(destination), idx)
            if key in self._pending:
                continue  # already armed, measuring from first input

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop()

            handle = loop.call_later(
                sub.threshold_secs,
                lambda k=key, pn=destination.name, s=sub: asyncio.create_task(self._fire(k, pn, s)),
            )
            self._pending[key] = handle

    def _resolve_input_frame(
        self, sub: _Subscription, processor: FrameProcessor
    ) -> Optional[Type[Frame]]:
        if sub.input_frame is not None:
            return sub.input_frame
        for base_type, frame_type in self._DEFAULT_INPUT_FRAMES.items():
            if isinstance(processor, base_type):
                return frame_type
        return None

    async def _fire(self, key: Tuple[int, int], processor_name: str, sub: _Subscription) -> None:
        self._pending.pop(key, None)
        now = asyncio.get_event_loop().time()
        if now - self._last_fire.get(key, 0.0) < self._cooldown_secs:
            return  # in cooldown
        self._last_fire[key] = now
        await sub.handler(processor_name, sub.threshold_secs)

    def _cancel_all(self) -> None:
        for handle in self._pending.values():
            handle.cancel()
        self._pending.clear()
