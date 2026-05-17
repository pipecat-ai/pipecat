#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Proactive latency monitoring observer.

Fires a callback when a measurement window (between two frames, optionally
scoped to specific processor types) exceeds a threshold. Unlike ``TTFBMetricsData``
which is reactive (emitted after a response arrives), this observer fires
*before* the response arrives — so an application can react mid-wait (e.g.
play a filler TTS message).
"""

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, Optional, Tuple, Type

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    MetricsFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameProcessor

LatencyHandler = Callable[[str, float], Awaitable[None]]


@dataclass
class _Subscription:
    start_frame: Type[Frame]
    start_processor: Optional[Type[FrameProcessor]]
    stop_frame: Optional[Type[Frame]]
    stop_processor: Optional[Type[FrameProcessor]]
    threshold_secs: float
    handler: LatencyHandler


class LatencyWatchdogObserver(BaseObserver):
    """Observer that proactively monitors processing latency between pipeline components.

    Arms a timer when a specific frame is pushed (optionally to a specific processor
    type) and disarms it when a stop signal is received (either a ``TTFBMetricsData``
    or an explicit ``stop_frame``, optionally emitted by a specific processor type).

    If the timer expires before being disarmed, the registered callback is fired.
    Supports three monitoring modes:

    - **Per-processor**: arm and disarm on the same processor (e.g. LLM TTFB).
    - **Spanning**: measure latency across two different processors (e.g. STT → TTS).
    - **Global**: no processor constraint — arm/disarm on any processor matching the frame.

    Example (per-processor, the common case)::

        @watchdog.subscribe(
            start_processor=LLMService,
            start_frame=LLMContextFrame,
            threshold_secs=0.5,
        )
        async def on_slow_llm(name, elapsed_secs):
            logger.warning(f"{name} exceeded threshold ({elapsed_secs}s)")

    Example (spanning, from user stop-speaking to bot start-speaking)::

        @watchdog.subscribe(
            start_frame=VADUserStoppedSpeakingFrame,
            stop_frame=BotStartedSpeakingFrame,
            threshold_secs=1.5,
        )
        async def on_slow_pipeline(name, elapsed_secs):
            await task.queue_frame(TTSSpeakFrame("Still working on it..."))
    """

    def __init__(self, *, cooldown_secs: float = 5.0, **kwargs):
        """Initialize the latency watchdog.

        Args:
            cooldown_secs: Minimum seconds between two consecutive fires of the same
                subscription. Prevents callback flooding when latency stays above threshold.
            **kwargs: Passed to :class:`BaseObserver`.
        """
        super().__init__(**kwargs)
        self._subscriptions: List[_Subscription] = []
        self._pending: Dict[Tuple[int, int], asyncio.TimerHandle] = {}
        self._last_fire: Dict[Tuple[int, int], float] = {}
        self._cooldown_secs = cooldown_secs

    def subscribe(
        self,
        *,
        start_frame: Type[Frame],
        threshold_secs: float,
        start_processor: Optional[Type[FrameProcessor]] = None,
        stop_frame: Optional[Type[Frame]] = None,
        stop_processor: Optional[Type[FrameProcessor]] = None,
    ) -> Callable[[LatencyHandler], LatencyHandler]:
        """Register a callback fired when a latency threshold is exceeded.

        Args:
            start_frame: Frame type that arms the timer when pushed.
            threshold_secs: Fire the callback if the timer is not disarmed within this delay.
            start_processor: If set, only arm when the ``start_frame`` is pushed to an
                instance of this processor type (or a subclass).
            stop_frame: Frame type that disarms the timer. Defaults to ``TTFBMetricsData``.
            stop_processor: If set, only disarm when the stop signal comes from an instance
                of this processor type (or a subclass). If neither ``stop_frame`` nor
                ``stop_processor`` is specified and ``start_processor`` is set,
                ``stop_processor`` defaults to ``start_processor`` (per-processor mode).

        Returns:
            A decorator that registers the wrapped coroutine as the handler. The handler
            receives ``(name: str, elapsed: float)``.
        """
        # Smart default for per-processor mode: if the user specifies a start_processor but
        # nothing about the stop, assume they want to measure TTFB on that same processor.
        if stop_processor is None and stop_frame is None and start_processor is not None:
            stop_processor = start_processor

        def decorator(func: LatencyHandler) -> LatencyHandler:
            self._subscriptions.append(
                _Subscription(
                    start_frame=start_frame,
                    start_processor=start_processor,
                    stop_frame=stop_frame,
                    stop_processor=stop_processor,
                    threshold_secs=threshold_secs,
                    handler=func,
                )
            )
            return func

        return decorator

    async def on_push_frame(self, data: FramePushed) -> None:
        """Handle each frame push: cancel all on interruption, disarm matching, arm matching."""
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
            if not isinstance(frame, sub.start_frame):
                continue

            if sub.start_processor is not None and not isinstance(destination, sub.start_processor):
                continue

            # Keying: (sub_idx, proc_id).
            # proc_id is 0 for spanning/global subs (a single active timer per subscription).
            # Otherwise it's id(processor), allowing multiple concurrent measurements for
            # different instances of the same processor type.
            is_spanning = sub.start_processor != sub.stop_processor or sub.start_processor is None
            proc_id = 0 if is_spanning else id(destination)
            key = (idx, proc_id)

            if key in self._pending:
                continue  # already armed — ignore

            loop = asyncio.get_running_loop()
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

            is_spanning = sub.start_processor != sub.stop_processor or sub.start_processor is None
            proc_id = 0 if is_spanning else id(source)
            key = (idx, proc_id)

            handle = self._pending.pop(key, None)
            if handle:
                handle.cancel()

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
        now = asyncio.get_running_loop().time()
        if now - self._last_fire.get(key, 0.0) < self._cooldown_secs:
            return  # in cooldown
        self._last_fire[key] = now
        try:
            await sub.handler(name, sub.threshold_secs)
        except Exception as e:
            # A misbehaving handler must not break the observer for other subscriptions.
            logger.error(f"LatencyWatchdog handler raised: {e}")

    def _cancel_all(self) -> None:
        for handle in self._pending.values():
            handle.cancel()
        self._pending.clear()
