#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Service switcher for switching between different services at runtime, with different switching strategies."""

from abc import abstractmethod
from typing import Any, Dict, Generic, List, Optional, Set, Tuple, Type, TypeVar

from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    ManuallySwitchServiceFrame,
    ServiceMetadataFrame,
    ServiceSwitcherFrame,
    ServiceSwitcherRequestMetadataFrame,
    TTSErrorFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.base_object import BaseObject


class ServiceSwitcherStrategy(BaseObject):
    """Base class for service switching strategies.

    Note:
        Strategy classes are instantiated internally by ServiceSwitcher.
        Developers should pass the strategy class (not an instance) to ServiceSwitcher.

    Event handlers available:

    - on_service_switched: Called when the active service changes.

    Example::

        @strategy.event_handler("on_service_switched")
        async def on_service_switched(strategy, service):
            ...
    """

    def __init__(self, services: List[FrameProcessor]):
        """Initialize the service switcher strategy with a list of services.

        Note:
            This is called internally by ServiceSwitcher. Do not instantiate directly.

        Args:
            services: List of frame processors to switch between.
        """
        super().__init__()

        if len(services) == 0:
            raise Exception(f"ServiceSwitcherStrategy needs at least one service")

        self._services = services
        self._active_service = services[0]

        self._register_event_handler("on_service_switched")

    @property
    def services(self) -> List[FrameProcessor]:
        """Return the list of available services."""
        return self._services

    @property
    def active_service(self) -> FrameProcessor:
        """Return the currently active service."""
        return self._active_service

    async def handle_frame(
        self, frame: ServiceSwitcherFrame, direction: FrameDirection
    ) -> Optional[FrameProcessor]:
        """Handle a frame that controls service switching.

        The base implementation returns ``None`` for all frames. Subclasses
        override this to implement specific switching behaviors.

        Args:
            frame: The frame to handle.
            direction: The direction of the frame (upstream or downstream).

        Returns:
            The newly active service if a switch occurred, or None otherwise.
        """
        return None

    async def handle_error(self, error: ErrorFrame) -> Optional[FrameProcessor]:
        """Handle an error from the active service.

        Called by ``ServiceSwitcher`` when a non-fatal ``ErrorFrame`` is pushed
        upstream by the currently active service. Subclasses can override this
        to implement automatic failover.

        Args:
            error: The error frame pushed by the active service.

        Returns:
            The newly active service if a switch occurred, or None otherwise.
        """
        return None

    async def _set_active_if_available(self, service: FrameProcessor) -> Optional[FrameProcessor]:
        """Set the active service to the given one, if it is in the list of available services.

        If it's not in the list, the request is ignored, as it may have been
        intended for another ServiceSwitcher in the pipeline.

        Args:
            service: The service to set as active.

        Returns:
            The newly active service, or None if the service was not found.
        """
        if service in self.services:
            self._active_service = service
            await service.queue_frame(ServiceSwitcherRequestMetadataFrame(service=service))
            await self._call_event_handler("on_service_switched", service)
            return service
        return None


class ServiceSwitcherStrategyManual(ServiceSwitcherStrategy):
    """A strategy for switching between services manually.

    This strategy allows the user to manually select which service is active.
    The initial active service is the first one in the list.

    Example::

        stt_switcher = ServiceSwitcher(
            services=[stt_1, stt_2],
            strategy_type=ServiceSwitcherStrategyManual
        )
    """

    async def handle_frame(
        self, frame: ServiceSwitcherFrame, direction: FrameDirection
    ) -> Optional[FrameProcessor]:
        """Handle a frame that controls service switching.

        Args:
            frame: The frame to handle.
            direction: The direction of the frame (upstream or downstream).

        Returns:
            The newly active service if a switch occurred, or None otherwise.
        """
        if isinstance(frame, ManuallySwitchServiceFrame):
            return await self._set_active_if_available(frame.service)

        return None


class ServiceSwitcherStrategyFailover(ServiceSwitcherStrategyManual):
    """A strategy that retries the active service and fails over to the next on repeated errors.

    When a ``TTSErrorFrame`` is received, the strategy retries the current
    service up to ``max_retries`` times. Once the retry budget for a service
    is exhausted, it switches to the next service in the list and resets the
    per-service retry counter. If all services have been exhausted, the error
    is propagated upstream.

    The retry counter is tracked **per service** and is **not** reset on
    success — once a service has failed enough times, the switch is permanent.

    Example::

        tts_switcher = ServiceSwitcher(
            services=[tts_primary, tts_fallback],
            strategy_type=ServiceSwitcherStrategyFailover,
            strategy_kwargs={"max_retries": 3},
        )
    """

    def __init__(self, services: List[FrameProcessor], *, max_retries: int = 2):
        """Initialize the failover strategy.

        Args:
            services: List of frame processors to switch between.
            max_retries: Number of retries on the current service before switching.
        """
        super().__init__(services)
        self._max_retries = max_retries
        self._retry_count = 0
        self._failed_services: Set[FrameProcessor] = set()

    async def handle_frame(
        self, frame: ServiceSwitcherFrame, direction: FrameDirection
    ) -> Optional[FrameProcessor]:
        """Handle manual switch frames (pass-through to allow composition).

        Args:
            frame: The frame to handle.
            direction: The direction of the frame.

        Returns:
            The newly active service if a switch occurred, or None otherwise.
        """
        if isinstance(frame, ManuallySwitchServiceFrame):
            if frame.service in self.services:
                self._active_service = frame.service
                self._retry_count = 0
                await self._call_event_handler("on_service_switched", frame.service)
                return frame.service
        return None

    async def handle_error(
        self, error_frame: TTSErrorFrame
    ) -> Tuple[bool, Optional[FrameProcessor]]:
        """Decide whether to retry or fail over based on the error.

        Args:
            error_frame: The TTS error frame containing the failed text.

        Returns:
            A tuple of (should_retry, switched_service). If should_retry is True,
            the caller should re-send the text to the active service.
            switched_service is the new service if a switch occurred, or None.
        """
        self._retry_count += 1

        if self._retry_count <= self._max_retries:
            logger.warning(
                f"TTS error on {self._active_service.name}, "
                f"retry {self._retry_count}/{self._max_retries}"
            )
            return (True, None)

        # Current service exhausted — mark as failed and try the next one.
        self._failed_services.add(self._active_service)
        logger.warning(
            f"TTS retries exhausted on {self._active_service.name}, switching to next service"
        )

        next_service = self._find_next_service()
        if next_service is None:
            logger.error("All TTS services have been exhausted")
            return (False, None)

        self._active_service = next_service
        self._retry_count = 1  # First attempt on the new service counts
        await self._call_event_handler("on_service_switched", next_service)
        return (True, next_service)

    def _find_next_service(self) -> Optional[FrameProcessor]:
        """Find the next service that hasn't been marked as failed.

        Returns:
            The next available service, or None if all have failed.
        """
        for service in self._services:
            if service not in self._failed_services:
                return service
        return None


StrategyType = TypeVar("StrategyType", bound=ServiceSwitcherStrategy)


class ServiceSwitcher(ParallelPipeline, Generic[StrategyType]):
    """Parallel pipeline that routes frames to one active service at a time.

    Wraps each service in a pair of filters that gate frame flow based on
    which service is currently active. Switching is controlled by
    `ServiceSwitcherFrame` frames and delegated to a pluggable
    `ServiceSwitcherStrategy`.

    Example::

        switcher = ServiceSwitcher(services=[stt_1, stt_2])
    """

    def __init__(
        self,
        services: List[FrameProcessor],
        strategy_type: Type[StrategyType] = ServiceSwitcherStrategyManual,
        strategy_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the service switcher with a list of services and a switching strategy.

        Args:
            services: List of frame processors to switch between.
            strategy_type: The strategy class to use for switching between services.
                Defaults to ``ServiceSwitcherStrategyManual``.
            strategy_kwargs: Extra keyword arguments forwarded to the strategy constructor.
        """
        _strategy = strategy_type(services, **(strategy_kwargs or {}))
        super().__init__(*self._make_pipeline_definitions(services, _strategy))
        self._services = services
        self._strategy = _strategy

    @property
    def strategy(self) -> StrategyType:
        """Return the active switching strategy."""
        return self._strategy

    @property
    def services(self) -> List[FrameProcessor]:
        """Return the list of available services."""
        return self._services

    @staticmethod
    def _make_pipeline_definitions(
        services: List[FrameProcessor], strategy: ServiceSwitcherStrategy
    ) -> List[Any]:
        pipelines = []
        for service in services:
            pipelines.append(ServiceSwitcher._make_pipeline_definition(service, strategy))
        return pipelines

    @staticmethod
    def _make_pipeline_definition(
        service: FrameProcessor, strategy: ServiceSwitcherStrategy
    ) -> Any:
        async def filter(_: Frame) -> bool:
            return service == strategy.active_service

        # Layout: Filter → Service → Filter
        #
        # filter_system_frames: we want to run filter functions also on system
        # frames.
        #
        # enable_direct_mode: filter functions are quick so we don't need
        # additional tasks.
        return [
            FunctionFilter(
                filter=filter,
                direction=FrameDirection.DOWNSTREAM,
                filter_system_frames=True,
                enable_direct_mode=True,
            ),
            service,
            FunctionFilter(
                filter=filter,
                direction=FrameDirection.UPSTREAM,
                filter_system_frames=True,
                enable_direct_mode=True,
            ),
        ]

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame out of the service switcher.

        Suppresses `ServiceSwitcherRequestMetadataFrame` targeting the active
        service (since it has already been handled) and `ServiceMetadataFrame`
        from inactive services so only the active service's metadata reaches
        downstream processors. One case this happens is with `StartFrame` since
        all the filters let it pass, and `StartFrame` causes the service to
        generate `ServiceMetadataFrame`.

        Non-fatal ``ErrorFrame`` instances are forwarded to the strategy via
        ``handle_error`` so strategies like ``ServiceSwitcherStrategyFailover``
        can perform failover. For ``TTSErrorFrame`` frames, the failover
        strategy can retry the failed text on the same or next service.
        """
        # Consume ServiceSwitcherRequestMetadataFrame once the targeted service
        # has handled it (i.e. the active service).
        if isinstance(frame, ServiceSwitcherRequestMetadataFrame):
            if frame.service == self.strategy.active_service:
                return

        # Only let metadata from the active service escape.
        if isinstance(frame, ServiceMetadataFrame):
            if frame.service_name != self.strategy.active_service.name:
                return

        # Let the strategy react to non-fatal errors from the active service.
        if isinstance(frame, ErrorFrame) and not frame.fatal:
            result = await self.strategy.handle_error(frame)
            # If handle_error returned a (should_retry, switched_service) tuple
            # (e.g. from ServiceSwitcherStrategyFailover with TTS retry), handle retry.
            if isinstance(result, tuple):
                should_retry, switched_service = result
                if should_retry and isinstance(frame, TTSErrorFrame):
                    if switched_service:
                        await switched_service.queue_frame(
                            ServiceSwitcherRequestMetadataFrame(service=switched_service)
                        )
                    retry_frame = TTSSpeakFrame(text=frame.text)
                    await self._strategy.active_service.queue_frame(retry_frame)
                    return

        await super().push_frame(frame, direction)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame, handling frames which affect service switching.

        Args:
            frame: The frame to process.
            direction: The direction of the frame (upstream or downstream).
        """
        if isinstance(frame, ServiceSwitcherFrame):
            service = await self.strategy.handle_frame(frame, direction)

            # If we don't switch to a new service we need to keep processing the
            # frame. If we switched, we just swallow the frame.
            if not service:
                await super().process_frame(frame, direction)
        else:
            await super().process_frame(frame, direction)
