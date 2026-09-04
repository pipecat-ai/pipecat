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
            raise Exception("ServiceSwitcherStrategy needs at least one service")

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

        Called by ``ServiceSwitcher`` for every non-fatal ``ErrorFrame`` the
        active service pushes upstream, whether or not the service can carry on
        from it. Subclasses override this to decide which errors are worth
        switching away from: `error.processor.is_usable` tells a service that
        is finished apart from one having a bad moment, and a subclass is free
        to switch on either.

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
        can no longer do its job (`is_usable=False`) is refused, since making it active would only
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

    This strategy switches when the active service reports an error that leaves
    it unable to do its job (`is_usable=False`), moving to the next service in
    the list that can still do its own. Errors the service can carry on from
    are left alone, so a provider hiccup doesn't cost a failover. Recovery and
    fallback policies are left to application code via the
    ``on_service_switched`` event.

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

        Only an error that leaves the service unable to do its job is worth a
        failover; anything it can carry on from is ignored. When one does,
        switches to the next service in the list that can still do its job,
        wrapping around from the end. The failed service stays in the list and
        can be switched back to once it has been brought back with
        :meth:`~pipecat.processors.frame_processor.FrameProcessor.set_usable`.

        Args:
            error: The error frame pushed by the active service.

        Returns:
            The newly active service if a switch occurred, or None if the error
            didn't warrant one or no other service can be given work.
        """
        failed_service = error.processor or self._active_service

        # The service is still working, so there is nothing to fail over from.
        if failed_service.is_usable:
            return None

        logger.warning(f"Service {failed_service.name} reported an error: {error.error}")

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

        # `is_usable` is read from the services, so keep the last value
        # announced: a service changing its own doesn't always change the
        # switcher's.
        self._announced_usable = self.is_usable
        for service in services:
            service.add_event_handler("on_usable_changed", self._on_service_usable_changed)

    async def set_usable(self, is_usable: bool):
        """Ignore an attempt to set this switcher's usability directly.

        A switcher has no usability of its own to set: what it reports is a
        reading of its services. Bring an individual service back with its own
        :meth:`~pipecat.processors.frame_processor.FrameProcessor.set_usable`,
        which brings the switcher back with it.

        Args:
            is_usable: Whether the switcher can be given work. Ignored.
        """
        # Not calling super(): it would set a flag this switcher never reads,
        # and announce a change that didn't happen.
        logger.debug(
            f"{self}: ignoring set_usable({is_usable}); a switcher reports whether any of "
            "its services can be given work"
        )

    async def _on_service_usable_changed(self, service: FrameProcessor, is_usable: bool):
        """Announce this switcher's usability after a service changed its own.

        Args:
            service: The service whose usability changed.
            is_usable: Whether that service can now be given work.
        """
        await self._announce_usable()

    async def _announce_usable(self):
        """Raise ``on_usable_changed`` if what this switcher reports has moved."""
        is_usable = self.is_usable
        if is_usable == self._announced_usable:
            return

        self._announced_usable = is_usable
        logger.debug(f"{self}: {'usable' if is_usable else 'no longer usable'}")
        await self._call_event_handler("on_usable_changed", is_usable)

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
        only once none of them can do theirs. This is a reading of the services
        rather than a state of its own, so bringing a service back with
        :meth:`~pipecat.processors.frame_processor.FrameProcessor.set_usable`
        brings the switcher back too, and calling that on the switcher itself
        does nothing. The switcher raises ``on_usable_changed`` for itself as
        the reading moves, so watching it is enough to hear about the services
        it holds.

        Returns:
            True while at least one service can be given work.
        """
        return any(service.is_usable for service in self._services)

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

        Non-fatal ``ErrorFrame`` instances from the services being switched
        between are answered for here rather than travelling on, since the rest
        of the pipeline deals with the switcher and not with what it holds.
        Errors from a service the switcher isn't using stop here outright.
        Every error from the active service is forwarded to the strategy via
        ``handle_error``, which decides whether it warrants a switch; a
        successful switch absorbs it, the switcher having gone on doing its
        job. An error that leaves the active service unable to do its job with
        no switch to be made is reported against the switcher, so that the rest
        of the pipeline judges it by what the switcher has left rather than by
        the one service that failed. Every other error travels upstream as
        usual.
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

        # Errors from the switched services are the switcher's to answer for.
        # Errors from anywhere else are just passing through, and travel on.
        if isinstance(frame, ErrorFrame) and not frame.fatal and frame.processor is not None:
            failed_service = frame.processor
            active_service = self.strategy.active_service
            # A service the switcher isn't using can't stop it doing its job,
            # so its errors go no further. Watch a service's own
            # ``on_usable_changed`` to hear about the ones held in reserve.
            if failed_service in self._services and failed_service is not active_service:
                return
            if failed_service is active_service:
                # The strategy decides which errors are worth switching away
                # from. A switch means the switcher recovered on its own, so
                # the error goes no further.
                if await self.strategy.handle_error(frame):
                    return
                # No switch was made. If the service can no longer work, the
                # switcher has nowhere left to send the work, so it reports the
                # failure as its own. Any other error travels upstream.
                if not failed_service.is_usable:
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
            # A permanent category would cost the switcher its own `is_usable`,
            # writing it off along with the one service that failed.
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
