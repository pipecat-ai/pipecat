#
# Copyright (c) 2025, Daily
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
    ServiceSwitcherFrame,
)
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class ServiceSwitcherStrategy:
    """Base class for service switching strategies."""

    def __init__(self, services: List[FrameProcessor]):
        """Initialize the service switcher strategy with a list of services."""
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
    """

    def __init__(self, services: List[FrameProcessor]):
        """Initialize the manual service switcher strategy with a list of services."""
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
        """Initialize the service switcher with a list of services and a switching strategy."""
        strategy = strategy_type(services)
        super().__init__(*self._make_pipeline_definitions(services, strategy))
        self.services = services
        self.strategy = strategy

    class ServiceSwitcherFilter(FunctionFilter):
        """An internal filter that allows frames to pass through to the wrapped service only if it's the active service."""

        def __init__(
            self,
            wrapped_service: FrameProcessor,
            active_service: FrameProcessor,
            direction: FrameDirection,
        ):
            """Initialize the service switcher filter with a strategy and direction."""

            async def filter(_: Frame) -> bool:
                return self._wrapped_service == self._active_service

            super().__init__(filter, direction)
            self._wrapped_service = wrapped_service
            self._active_service = active_service

        async def process_frame(self, frame, direction):
            """Process a frame through the filter, handling special internal filter-updating frames."""
            if isinstance(frame, ServiceSwitcher.ServiceSwitcherFilterFrame):
                self._active_service = frame.active_service
                # Two ServiceSwitcherFilters "sandwich" a service. Push the
                # frame only to update the other side of the sandwich, but
                # otherwise don't let it leave the sandwich.
                if direction == self._direction:
                    await self.push_frame(frame, direction)
                return

            await super().process_frame(frame, direction)

    @dataclass
    class ServiceSwitcherFilterFrame(ControlFrame):
        """An internal frame used by ServiceSwitcher to filter frames based on active service."""

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
