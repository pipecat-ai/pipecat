#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Service switcher for switching between different services at runtime, with different switching strategies."""

from typing import Any, Generic, List, Optional, Type, TypeVar

from pipecat.frames.frames import Frame, ManuallySwitchServiceFrame, ServiceSwitcherFrame
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class ServiceSwitcherStrategy:
    """Base class for service switching strategies."""

    def __init__(self, services: List[FrameProcessor]):
        """Initialize the service switcher strategy with a list of services."""
        self.services = services
        self.active_service: Optional[FrameProcessor] = None

    def is_active(self, service: FrameProcessor) -> bool:
        """Determine if the given service is the currently active one.

        This method should be overridden by subclasses to implement specific logic.

        Args:
            service: The service to check.

        Returns:
            True if the given service is the active one, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement this method.")

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

    def is_active(self, service: FrameProcessor) -> bool:
        """Check if the given service is the currently active one.

        Args:
            service: The service to check.

        Returns:
            True if the given service is the active one, False otherwise.
        """
        return service == self.active_service

    def handle_frame(self, frame: ServiceSwitcherFrame, direction: FrameDirection):
        """Handle a frame that controls service switching.

        Args:
            frame: The frame to handle.
            direction: The direction of the frame (upstream or downstream).
        """
        if isinstance(frame, ManuallySwitchServiceFrame):
            self._set_active(frame.service)
        else:
            raise ValueError(f"Unsupported frame type: {type(frame)}")

    def _set_active(self, service: FrameProcessor):
        """Set the active service to the given one.

        Args:
            service: The service to set as active.
        """
        if service in self.services:
            self.active_service = service
        else:
            raise ValueError(f"Service {service} is not in the list of available services.")


StrategyType = TypeVar("StrategyType", bound=ServiceSwitcherStrategy)


class ServiceSwitcher(ParallelPipeline, Generic[StrategyType]):
    """A pipeline that switches between different services at runtime."""

    def __init__(self, services: List[FrameProcessor], strategy_type: Type[StrategyType]):
        """Initialize the service switcher with a list of services and a switching strategy."""
        strategy = strategy_type(services)
        super().__init__(*self._make_pipeline_definitions(services, strategy))
        self.services = services
        self.strategy = strategy

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
        async def filter(frame) -> bool:
            _ = frame
            return strategy.is_active(service)

        return [
            FunctionFilter(filter, direction=FrameDirection.DOWNSTREAM),
            service,
            FunctionFilter(filter, direction=FrameDirection.UPSTREAM),
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
