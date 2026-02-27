#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Service switcher for switching between different services at runtime, with different switching strategies."""

import asyncio
from abc import abstractmethod
from typing import Any, Dict, Generic, List, Optional, Set, Type, TypeVar

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


class ServiceSwitcherStrategyAutomatic(ServiceSwitcherStrategy):
    """A strategy that automatically switches to a backup service on failure.

    When the active service produces a non-fatal error, this strategy marks it
    as unhealthy and switches to the next healthy service in the list. After a
    configurable recovery period the unhealthy service is marked healthy again.
    If ``prefer_primary`` is enabled (the default) and the primary service
    recovers, the strategy switches back to it automatically.

    Manual switching via ``ManuallySwitchServiceFrame`` is still supported.

    Event handlers available:

    - on_service_switched: Called when the active service changes.

    Example::

        switcher = ServiceSwitcher(
            services=[primary_stt, backup_stt],
            strategy_type=ServiceSwitcherStrategyAutomatic,
        )

        # Optionally customise parameters after construction:
        switcher.strategy.recovery_timeout = 60.0
        switcher.strategy.prefer_primary = False
    """

    def __init__(self, services: List[FrameProcessor]):
        """Initialize the automatic service switcher strategy.

        Args:
            services: List of frame processors to switch between.
                The first service is the primary.
        """
        super().__init__(services)
        self._primary_service: FrameProcessor = services[0]
        self._unhealthy_services: Set[FrameProcessor] = set()
        self._recovery_timers: Dict[FrameProcessor, asyncio.TimerHandle] = {}

        self.recovery_timeout: float = 30.0
        """Seconds before an unhealthy service is marked healthy again."""

        self.prefer_primary: bool = True
        """When True, automatically switch back to the primary once it recovers."""

    @property
    def unhealthy_services(self) -> Set[FrameProcessor]:
        """Return the set of services currently marked as unhealthy."""
        return set(self._unhealthy_services)

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

        Marks the active service as unhealthy, schedules a recovery timer,
        and switches to the next healthy service in the list.

        Args:
            error: The error frame pushed by the active service.

        Returns:
            The newly active service if a switch occurred, or None if no
            healthy backup is available.
        """
        failed_service = self._active_service

        logger.warning(
            f"Service {failed_service.name} reported an error, marking unhealthy: {error.error}"
        )

        self._unhealthy_services.add(failed_service)
        self._schedule_recovery(failed_service)

        next_service = self._next_healthy_service()
        if next_service is None:
            logger.error("All services are unhealthy, no backup available")
            return None

        return await self._set_active_if_available(next_service)

    def _next_healthy_service(self) -> Optional[FrameProcessor]:
        """Find the next healthy service in the list.

        Searches from the beginning of the service list so that
        higher-priority services are preferred.

        Returns:
            The next healthy service, or None if all are unhealthy.
        """
        for service in self._services:
            if service not in self._unhealthy_services:
                return service
        return None

    def _schedule_recovery(self, service: FrameProcessor):
        """Schedule a recovery timer for an unhealthy service.

        If a timer is already running for this service it is cancelled and
        restarted so that repeated errors extend the cooldown.

        Args:
            service: The service to schedule recovery for.
        """
        existing = self._recovery_timers.pop(service, None)
        if existing is not None:
            existing.cancel()

        loop = asyncio.get_event_loop()
        handle = loop.call_later(
            self.recovery_timeout,
            lambda s=service: asyncio.ensure_future(self._recover_service(s)),
        )
        self._recovery_timers[service] = handle

    async def _recover_service(self, service: FrameProcessor):
        """Mark a service as healthy after its recovery timeout expires.

        If ``prefer_primary`` is enabled and the recovered service is the
        primary, the strategy switches back to it automatically.

        Args:
            service: The service that has recovered.
        """
        self._unhealthy_services.discard(service)
        self._recovery_timers.pop(service, None)

        logger.info(f"Service {service.name} marked healthy after recovery timeout")

        if self.prefer_primary and service == self._primary_service:
            if self._active_service != self._primary_service:
                logger.info(f"Switching back to primary service {service.name}")
                await self._set_active_if_available(self._primary_service)

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
        ``handle_error`` so strategies like ``ServiceSwitcherStrategyAutomatic``
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
