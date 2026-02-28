#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Service switcher for switching between different services at runtime, with different switching strategies."""

from abc import abstractmethod
from typing import Any, Generic, List, Optional, Type, TypeVar

from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    ManuallySwitchServiceFrame,
    ServiceMetadataFrame,
    ServiceSwitcherFrame,
    ServiceSwitcherRequestMetadataFrame,
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

    @abstractmethod
    async def handle_frame(
        self, frame: ServiceSwitcherFrame, direction: FrameDirection
    ) -> Optional[FrameProcessor]:
        """Handle a frame that controls service switching.

        Subclasses implement this to decide whether a switch should occur.

        Args:
            frame: The frame to handle.
            direction: The direction of the frame (upstream or downstream).

        Returns:
            The newly active service if a switch occurred, or None otherwise.
        """
        pass

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
            await self._call_event_handler("on_service_switched", service)
            return service
        return None


class ServiceSwitcherStrategyFailover(ServiceSwitcherStrategy):
    """A strategy that automatically switches to a backup service on failure.

    When the active service produces a non-fatal error, this strategy switches
    to the next available service in the list. Recovery and fallback policies
    are left to application code via the ``on_service_switched`` event.

    Manual switching via ``ManuallySwitchServiceFrame`` is still supported.

    Event handlers available:

    - on_service_switched: Called when the active service changes.

    Example::

        switcher = ServiceSwitcher(
            services=[primary_stt, backup_stt],
            strategy_type=ServiceSwitcherStrategyFailover,
        )

        @switcher.strategy.event_handler("on_service_switched")
        async def on_switched(strategy, service):
            # App decides when/how to recover the failed service
            ...
    """

    def __init__(self, services: List[FrameProcessor]):
        """Initialize the automatic service switcher strategy.

        Args:
            services: List of frame processors to switch between.
        """
        super().__init__(services)

    async def handle_frame(
        self, frame: ServiceSwitcherFrame, direction: FrameDirection
    ) -> Optional[FrameProcessor]:
        """Handle a frame that controls service switching.

        Supports ``ManuallySwitchServiceFrame`` for explicit overrides.

        Args:
            frame: The frame to handle.
            direction: The direction of the frame (upstream or downstream).

        Returns:
            The newly active service if a switch occurred, or None otherwise.
        """
        if isinstance(frame, ManuallySwitchServiceFrame):
            return await self._set_active_if_available(frame.service)
        return None

    async def handle_error(self, error: ErrorFrame) -> Optional[FrameProcessor]:
        """Handle an error from the active service by failing over.

        Switches to the next service in the list. The failed service remains
        in the list and can be switched back to manually or via application
        logic in the ``on_service_switched`` event handler.

        Args:
            error: The error frame pushed by the active service.

        Returns:
            The newly active service if a switch occurred, or None if no
            other service is available.
        """
        failed_service = self._active_service

        logger.warning(f"Service {failed_service.name} reported an error: {error.error}")

        next_service = self._next_service()
        if next_service is None:
            logger.error("No other service available to switch to")
            return None

        return await self._set_active_if_available(next_service)

    def _next_service(self) -> Optional[FrameProcessor]:
        """Find the next service in the list after the active one.

        Returns:
            The next service, or None if there is only one service.
        """
        if len(self._services) <= 1:
            return None
        current_idx = self._services.index(self._active_service)
        next_idx = (current_idx + 1) % len(self._services)
        return self._services[next_idx]

    async def _set_active_if_available(self, service: FrameProcessor) -> Optional[FrameProcessor]:
        """Set the active service if it is in the service list.

        Args:
            service: The service to set as active.

        Returns:
            The newly active service, or None if the service was not found.
        """
        if service in self.services:
            self._active_service = service
            await self._call_event_handler("on_service_switched", service)
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

        switcher = ServiceSwitcher(
            services=[stt_1, stt_2],
            strategy_type=ServiceSwitcherStrategyManual,
        )
    """

    def __init__(self, services: List[FrameProcessor], strategy_type: Type[StrategyType]):
        """Initialize the service switcher with a list of services and a switching strategy.

        Args:
            services: List of frame processors to switch between.
            strategy_type: The strategy class to use for switching between services.
        """
        _strategy = strategy_type(services)
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
        can perform failover. The error frame is still propagated upstream so
        that application-level error handlers can observe it.
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
            service = await self.strategy.handle_error(frame)
            if service:
                await service.queue_frame(ServiceSwitcherRequestMetadataFrame(service=service))

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

            # If we switched to a new service, request its metadata.
            if service:
                await service.queue_frame(ServiceSwitcherRequestMetadataFrame(service=service))
        else:
            await super().process_frame(frame, direction)
