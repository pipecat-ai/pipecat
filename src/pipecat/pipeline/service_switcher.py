#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Service switcher for switching between different services at runtime, with different switching strategies."""

from dataclasses import dataclass
from typing import Any, Generic, List, Optional, Type, TypeVar

from pipecat.frames.frames import (
    ControlFrame,
    Frame,
    ManuallySwitchServiceFrame,
    RequestMetadataFrame,
    ServiceMetadataFrame,
    ServiceSwitcherFrame,
    StartFrame,
)
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class ServiceSwitcherStrategy:
    """Base class for service switching strategies.

    Note:
        Strategy classes are instantiated internally by ServiceSwitcher.
        Developers should pass the strategy class (not an instance) to ServiceSwitcher.
    """

    def __init__(self, services: List[FrameProcessor]):
        """Initialize the service switcher strategy with a list of services.

        Note:
            This is called internally by ServiceSwitcher. Do not instantiate directly.

        Args:
            services: List of frame processors to switch between.
        """
        self.services = services
        self.active_service: Optional[FrameProcessor] = None

    def handle_frame(self, frame: ServiceSwitcherFrame, direction: FrameDirection):
        """Handle a frame that controls service switching.

        This method can be overridden by subclasses to implement specific logic
        for handling frames that control service switching.

        Args:
            frame: The frame to handle.
            direction: The direction of the frame (upstream or downstream).
        """
        raise NotImplementedError("Subclasses must implement this method.")


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

    def __init__(self, services: List[FrameProcessor]):
        """Initialize the manual service switcher strategy with a list of services.

        Note:
            This is called internally by ServiceSwitcher. Do not instantiate directly.

        Args:
            services: List of frame processors to switch between.
        """
        super().__init__(services)
        self.active_service = services[0] if services else None

    def handle_frame(self, frame: ServiceSwitcherFrame, direction: FrameDirection):
        """Handle a frame that controls service switching.

        Args:
            frame: The frame to handle.
            direction: The direction of the frame (upstream or downstream).
        """
        if isinstance(frame, ManuallySwitchServiceFrame):
            self._set_active_if_available(frame.service)
        else:
            raise ValueError(f"Unsupported frame type: {type(frame)}")

    def _set_active_if_available(self, service: FrameProcessor):
        """Set the active service to the given one, if it is in the list of available services.

        If it's not in the list, the request is ignored, as it may have been
        intended for another ServiceSwitcher in the pipeline.

        Args:
            service: The service to set as active.
        """
        if service in self.services:
            self.active_service = service


StrategyType = TypeVar("StrategyType", bound=ServiceSwitcherStrategy)


class ServiceSwitcher(ParallelPipeline, Generic[StrategyType]):
    """A pipeline that switches between different services at runtime."""

    def __init__(self, services: List[FrameProcessor], strategy_type: Type[StrategyType]):
        """Initialize the service switcher with a list of services and a switching strategy.

        Args:
            services: List of frame processors to switch between.
            strategy_type: The strategy class to use for switching between services.
        """
        strategy = strategy_type(services)
        super().__init__(*self._make_pipeline_definitions(services, strategy))
        self.services = services
        self.strategy = strategy

    class ServiceSwitcherFilter(FunctionFilter):
        """An internal filter that gates frame flow based on active service.

        Two filters "sandwich" each service (one upstream, one downstream),
        allowing frames through only when the wrapped service is active.
        """

        def __init__(
            self,
            wrapped_service: FrameProcessor,
            active_service: FrameProcessor,
            direction: FrameDirection,
        ):
            """Initialize the service switcher filter with a strategy and direction.

            Args:
                wrapped_service: The service that this filter wraps.
                active_service: The currently active service.
                direction: The direction of frame flow to filter.
            """
            self._wrapped_service = wrapped_service
            self._active_service = active_service
            self._startup_complete = False

            async def filter(_: Frame) -> bool:
                return self._wrapped_service == self._active_service

            super().__init__(filter, direction, filter_system_frames=True)

        async def process_frame(self, frame, direction):
            """Process a frame through the filter, handling special internal filter-updating frames."""
            if isinstance(frame, ServiceSwitcher.ServiceSwitcherFilterFrame):
                old_active = self._active_service
                old_startup_complete = self._startup_complete
                self._active_service = frame.active_service
                self._startup_complete = True  # True after StartFrame is released
                # Two ServiceSwitcherFilters "sandwich" a service. Push the
                # frame only to update the other side of the sandwich, but
                # otherwise don't let it leave the sandwich.
                if direction == self._direction:
                    await self.push_frame(frame, direction)
                # If this is the upstream filter for the active service, request
                # metadata when: (1) startup just completed, or (2) service became
                # active. Startup triggers metadata request because ParallelPipeline
                # synchronizes StartFrame release, so we must request metadata after
                # to maintain ordering for downstream processors.
                elif (
                    self._direction == FrameDirection.UPSTREAM
                    and self._wrapped_service == frame.active_service
                    and (not old_startup_complete or old_active != self._wrapped_service)
                ):
                    await self.push_frame(RequestMetadataFrame(), FrameDirection.UPSTREAM)
                return

            # RequestMetadataFrame is internal to ServiceSwitcher - only let it
            # through to the active service and don't let it leave.
            if isinstance(frame, RequestMetadataFrame):
                if direction == self._direction and self._wrapped_service == self._active_service:
                    await self.push_frame(frame, direction)
                return

            # Special case: ServiceMetadataFrame must be filtered in BOTH directions.
            # Block if: (1) from inactive service, or (2) during startup (before
            # StartFrame is released) to prevent ordering issues.
            if isinstance(frame, ServiceMetadataFrame):
                if self._wrapped_service != self._active_service or not self._startup_complete:
                    return
                await self.push_frame(frame, direction)
                return

            await super().process_frame(frame, direction)

    @dataclass
    class ServiceSwitcherFilterFrame(ControlFrame):
        """An internal frame used to update filter state.

        Sent on startup (after StartFrame is released) and on service switch.
        Updates active service and marks startup as complete, then triggers
        metadata emission from the active service.
        """

        active_service: FrameProcessor

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
        return [
            ServiceSwitcher.ServiceSwitcherFilter(
                wrapped_service=service,
                active_service=strategy.active_service,
                direction=FrameDirection.DOWNSTREAM,
            ),
            service,
            ServiceSwitcher.ServiceSwitcherFilter(
                wrapped_service=service,
                active_service=strategy.active_service,
                direction=FrameDirection.UPSTREAM,
            ),
        ]

    async def _parallel_push_frame(self, frame: Frame, direction: FrameDirection):
        """Push frames while handling StartFrame completion for metadata ordering.

        When StartFrame is released (after all branches have processed it),
        we send a filter frame to mark startup as complete and trigger metadata
        emission. This ensures ServiceMetadataFrame arrives after StartFrame.
        """
        await super()._parallel_push_frame(frame, direction)

        if isinstance(frame, StartFrame):
            # StartFrame has been released. Send filter frame to mark startup
            # complete and trigger metadata from the active service.
            await super().process_frame(
                ServiceSwitcher.ServiceSwitcherFilterFrame(
                    active_service=self.strategy.active_service
                ),
                FrameDirection.DOWNSTREAM,
            )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame, handling frames which affect service switching.

        Args:
            frame: The frame to process.
            direction: The direction of the frame (upstream or downstream).
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, ServiceSwitcherFrame):
            self.strategy.handle_frame(frame, direction)
            service_switcher_filter_frame = ServiceSwitcher.ServiceSwitcherFilterFrame(
                active_service=self.strategy.active_service
            )
            await super().process_frame(service_switcher_filter_frame, direction)
