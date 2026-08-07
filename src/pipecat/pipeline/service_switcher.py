#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Service switcher for switching between different services at runtime, with different switching strategies."""

from collections import deque
from dataclasses import replace
from typing import Any, Generic, TypeVar

from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    ManuallySwitchServiceFrame,
    ServiceMetadataFrame,
    ServiceSwitcherFrame,
    ServiceSwitcherRequestMetadataFrame,
    ServiceUpdateSettingsFrame,
)
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.base_object import BaseObject
from pipecat.utils.errors import ErrorCategory


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

    def __init__(self, services: list[FrameProcessor]):
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
    def services(self) -> list[FrameProcessor]:
        """Return the list of available services."""
        return self._services

    @property
    def active_service(self) -> FrameProcessor:
        """Return the currently active service."""
        return self._active_service

    @property
    def usable_services(self) -> list[FrameProcessor]:
        """Return the services that can still be given work, in order."""
        return [service for service in self._services if service.is_usable]

    async def handle_frame(
        self, frame: ServiceSwitcherFrame, direction: FrameDirection
    ) -> FrameProcessor | None:
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

    async def handle_error(self, error: ErrorFrame) -> FrameProcessor | None:
        """Handle an error from the active service.

        Called by ``ServiceSwitcher`` when the active service pushes a
        non-fatal ``ErrorFrame`` upstream that leaves it unable to do its job.
        Subclasses can override this to implement automatic failover.

        Args:
            error: The error frame pushed by the active service.

        Returns:
            The newly active service if a switch occurred, or None otherwise.
        """
        return None

    async def _set_active_if_available(self, service: FrameProcessor) -> FrameProcessor | None:
        """Set the active service to the given one, if it is in the list of available services.

        If it's not in the list, the request is ignored, as it may have been
        intended for another ServiceSwitcher in the pipeline. A service that
        can no longer do its job is refused, since making it active would only
        route work to something that can't handle it; call
        :meth:`~pipecat.processors.frame_processor.FrameProcessor.set_usable`
        on it first once whatever stopped it working has been dealt with.

        Args:
            service: The service to set as active.

        Returns:
            The newly active service, or None if the service was not found or
            can no longer be given work.
        """
        if service not in self.services:
            return None

        if not service.is_usable:
            logger.warning(f"Not switching to {service.name}: it can no longer do its job")
            return None

        self._active_service = service
        await service.queue_frame(ServiceSwitcherRequestMetadataFrame(service=service))
        await self._call_event_handler("on_service_switched", service)
        return service


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
    ) -> FrameProcessor | None:
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
    """A strategy that automatically switches to a backup service on failure.

    When the active service reports an error that leaves it unable to do its
    job, this strategy switches to the next service in the list that can still
    do its own. Errors a service can carry on from are left alone, so a
    provider hiccup doesn't cost a failover. Recovery and fallback policies are
    left to application code via the ``on_service_switched`` event.

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

    async def handle_error(self, error: ErrorFrame) -> FrameProcessor | None:
        """Handle an error from the active service by failing over.

        Switches to the next service in the list that can still do its job,
        wrapping around from the end. The failed service stays in the list and
        can be switched back to once it has been brought back with
        :meth:`~pipecat.processors.frame_processor.FrameProcessor.set_usable`.

        Args:
            error: The error frame pushed by the active service.

        Returns:
            The newly active service if a switch occurred, or None if no other
            service can be given work.
        """
        service_name = error.processor.name if error.processor else self._active_service.name
        logger.warning(f"Service {service_name} reported an error: {error.error}")

        # Walk the list from the one after the active service so failover
        # follows the order the services were given in.
        current_idx = self._services.index(self._active_service)
        for offset in range(1, len(self._services)):
            candidate = self._services[(current_idx + offset) % len(self._services)]
            if candidate.is_usable:
                return await self._set_active_if_available(candidate)

        logger.error("No other service available to switch to")
        return None


StrategyType = TypeVar("StrategyType", bound=ServiceSwitcherStrategy)


class ServiceSwitcher(ParallelPipeline, Generic[StrategyType]):
    """Parallel pipeline that routes frames to one active service at a time.

    Wraps each service in a pair of filters that gate frame flow based on
    which service is currently active. Switching is controlled by
    `ServiceSwitcherFrame` frames and delegated to a pluggable
    `ServiceSwitcherStrategy`.

    `ServiceUpdateSettingsFrame` is the exception to the gating. A settings
    update addressed to a member service (``service=``) reaches it whether or
    not it is active, and one marked ``reach_inactive_services`` reaches every
    member, so whichever service becomes active later is already configured. Any
    other settings update applies to the active service alone.

    Example::

        switcher = ServiceSwitcher(services=[stt_1, stt_2])
    """

    def __init__(
        self,
        services: list[FrameProcessor],
        strategy_type: type[StrategyType] = ServiceSwitcherStrategyManual,
    ):
        """Initialize the service switcher with a list of services and a switching strategy.

        Args:
            services: List of frame processors to switch between.
            strategy_type: The strategy class to use for switching between services.
                Defaults to ``ServiceSwitcherStrategyManual``.
        """
        _strategy = strategy_type(services)
        super().__init__(*self._make_pipeline_definitions(services, _strategy))
        self._services = services
        self._strategy = _strategy
        # Ids of the settings updates handed to services that weren't active, so
        # they can be consumed again on their way out. A small ring is enough: an
        # update crosses its service long before the ring wraps.
        self._inactive_service_updates: deque[int] = deque(maxlen=64)

    @property
    def strategy(self) -> StrategyType:
        """Return the active switching strategy."""
        return self._strategy

    @property
    def services(self) -> list[FrameProcessor]:
        """Return the list of available services."""
        return self._services

    @property
    def is_usable(self) -> bool:
        """Whether any of the switched services can still be given work.

        A switcher is only as dead as its last service: it can keep doing its
        job by moving work to a different one, so it reports itself unusable
        only once none of them can do theirs. Bringing a service back with
        :meth:`~pipecat.processors.frame_processor.FrameProcessor.set_usable`
        therefore brings the switcher back too — while calling that on the
        switcher itself takes it out of service regardless of what it holds.

        Returns:
            True while at least one service can be given work.
        """
        return super().is_usable and any(service.is_usable for service in self._services)

    @staticmethod
    def _make_pipeline_definitions(
        services: list[FrameProcessor], strategy: ServiceSwitcherStrategy
    ) -> list[Any]:
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

        Also suppresses the copies of a `ServiceUpdateSettingsFrame` handed to
        the inactive services, so that the update the rest of the pipeline sees
        is the one travelling the active service's branch.

        A non-fatal ``ErrorFrame`` that leaves the active service unable to do
        its job is forwarded to the strategy via ``handle_error``, so
        strategies like ``ServiceSwitcherStrategyFailover`` can perform
        failover. A successful failover absorbs the error: the switcher went on
        doing its job, so nothing upstream needs to hear about it. Without one,
        the error is reported against the switcher instead, so that the rest of
        the pipeline judges it by what the switcher has left rather than by the
        one service that failed. Every other error travels upstream as usual.
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

        # Consume the settings updates handed to the inactive services: they have
        # been delivered, and only the active service's copy travels on.
        if isinstance(frame, ServiceUpdateSettingsFrame):
            if frame.id in self._inactive_service_updates:
                return

        # Let the strategy react to errors that cost us the active service,
        # ignoring errors it can carry on from and errors just propagating
        # upstream from other processors.
        if isinstance(frame, ErrorFrame) and not frame.fatal:
            failed_service = frame.processor
            if (
                failed_service
                and failed_service == self.strategy.active_service
                and not failed_service.is_usable
            ):
                if await self.strategy.handle_error(frame):
                    return
                await self._report_service_failure(failed_service, frame)
                return

        await super().push_frame(frame, direction)

    async def _report_service_failure(self, failed_service: FrameProcessor, error: ErrorFrame):
        """Report a service failure the switcher could not switch away from.

        Re-attributes the error to the switcher, which is the processor the
        rest of the pipeline deals with. Whether losing this service matters
        depends on what the switcher has left, not on the service that just
        failed, so `is_usable` on the reported error answers for the switcher
        as a whole.

        Args:
            failed_service: The service that can no longer do its job.
            error: The error it reported.
        """
        await self.push_error(
            f"{failed_service.name} can no longer do its job: {error.error}",
            exception=error.exception,
            # The switcher's own configuration is never what a service's
            # rejection calls into question, and inheriting the category would
            # write the switcher off along with the one service that failed.
            category=ErrorCategory.UNKNOWN,
        )

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

            if isinstance(frame, ServiceUpdateSettingsFrame):
                await self._update_inactive_services(frame, direction)

    async def _update_inactive_services(
        self, frame: ServiceUpdateSettingsFrame, direction: FrameDirection
    ):
        """Hand a settings update to the member services that aren't active.

        The active service receives the update through its branch like any other
        frame. The inactive ones sit behind closed filters, so each is handed its
        own copy directly.

        Args:
            frame: The settings update to hand over.
            direction: The direction the settings update is travelling.
        """
        for service in self._inactive_update_targets(frame):
            # A copy carries the same update with an id of its own, which is how
            # push_frame tells it from the active service's.
            update = replace(frame)
            self._inactive_service_updates.append(update.id)
            await service.queue_frame(update, direction)

    def _inactive_update_targets(self, frame: ServiceUpdateSettingsFrame) -> list[FrameProcessor]:
        """Return the inactive member services that should apply a settings update.

        Args:
            frame: The settings update to route.

        Returns:
            The services to hand the update to, which may be empty.
        """
        inactive = [s for s in self.services if s is not self.strategy.active_service]

        # An addressed update goes to its service alone, active or not. One
        # addressed elsewhere in the pipeline is left for the switcher that
        # manages it.
        if frame.service is not None:
            return [frame.service] if frame.service in inactive else []

        # Any other update crosses to the inactive services only if it opts in,
        # since settings values are often specific to one provider.
        return inactive if frame.reach_inactive_services else []
