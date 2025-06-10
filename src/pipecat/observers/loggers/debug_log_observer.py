#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from dataclasses import fields, is_dataclass
from enum import Enum, auto
from typing import Dict, Optional, Set, Tuple, Type, Union

from loguru import logger

from pipecat.frames.frames import Frame
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection


class FrameEndpoint(Enum):
    """Specifies which endpoint (source or destination) to filter on."""

    SOURCE = auto()
    DESTINATION = auto()


class DebugLogObserver(BaseObserver):
    """Observer that logs frame activity with detailed content to the console.

    Automatically extracts and formats data from any frame type, making it useful
    for debugging pipeline behavior without needing frame-specific observers.

    Args:
        frame_types: Optional tuple of frame types to log, or a dict with frame type
            filters. If None, logs all frame types.
        exclude_fields: Optional set of field names to exclude from logging.

    Examples:
        Log all frames from all services:
        ```python
        observers = DebugLogObserver()
        ```

        Log specific frame types from any source/destination:
        ```python
        from pipecat.frames.frames import TranscriptionFrame, InterimTranscriptionFrame
        observers=[
            DebugLogObserver(frame_types=(LLMTextFrame,TranscriptionFrame,)),
        ],
        ```

        Log frames with specific source/destination filters:
        ```python
        from pipecat.frames.frames import StartInterruptionFrame, UserStartedSpeakingFrame, LLMTextFrame
        from pipecat.transports.base_output_transport import BaseOutputTransport
        from pipecat.services.stt_service import STTService

        observers=[
            DebugLogObserver(
                frame_types={
                    # Only log StartInterruptionFrame when source is BaseOutputTransport
                    StartInterruptionFrame: (BaseOutputTransport, FrameEndpoint.SOURCE),
                    # Only log UserStartedSpeakingFrame when destination is STTService
                    UserStartedSpeakingFrame: (STTService, FrameEndpoint.DESTINATION),
                    # Log LLMTextFrame regardless of source or destination type
                    LLMTextFrame: None,
                }
            ),
        ],
        ```
    """

    def __init__(
        self,
        frame_types: Optional[
            Union[Tuple[Type[Frame], ...], Dict[Type[Frame], Optional[Tuple[Type, FrameEndpoint]]]]
        ] = None,
        exclude_fields: Optional[Set[str]] = None,
        **kwargs,
    ):
        """Initialize the debug log observer.

        Args:
            frame_types: Tuple of frame types to log, or a dict mapping frame types to
                filter configurations. Filter configs can be:
                - None to log all instances of the frame type
                - A tuple of (service_type, endpoint) to filter on a specific service
                  and endpoint (SOURCE or DESTINATION)
                If None is provided instead of a tuple/dict, log all frames.
            exclude_fields: Set of field names to exclude from logging. If None, only binary
                data fields are excluded.
        """
        super().__init__(**kwargs)

        # Process frame filters
        self.frame_filters = {}

        if frame_types is not None:
            if isinstance(frame_types, tuple):
                # Tuple of frame types - log all instances
                self.frame_filters = {frame_type: None for frame_type in frame_types}
            else:
                # Dict of frame types with filters
                self.frame_filters = frame_types

        # By default, exclude binary data fields that would clutter logs
        self.exclude_fields = (
            exclude_fields
            if exclude_fields is not None
            else {
                "audio",  # Skip binary audio data
                "image",  # Skip binary image data
                "images",  # Skip lists of images
            }
        )

    def _format_value(self, value):
        """Format a value for logging.

        Args:
            value: The value to format.

        Returns:
            str: A string representation of the value suitable for logging.
        """
        if value is None:
            return "None"
        elif isinstance(value, str):
            return f"{value!r}"
        elif isinstance(value, (list, tuple)):
            if len(value) == 0:
                return "[]"
            if isinstance(value[0], dict) and len(value) > 3:
                # For message lists, just show count
                return f"{len(value)} items"
            return str(value)
        elif isinstance(value, (bytes, bytearray)):
            return f"{len(value)} bytes"
        elif hasattr(value, "get_messages_for_logging") and callable(
            getattr(value, "get_messages_for_logging")
        ):
            # Special case for OpenAI context
            return f"{value.__class__.__name__} with messages: {value.get_messages_for_logging()}"
        else:
            return str(value)

    def _should_log_frame(self, frame, src, dst):
        """Determine if a frame should be logged based on filters.

        Args:
            frame: The frame being processed
            src: The source component
            dst: The destination component

        Returns:
            bool: True if the frame should be logged, False otherwise
        """
        # If no filters, log all frames
        if not self.frame_filters:
            return True

        # Check if this frame type is in our filters
        for frame_type, filter_config in self.frame_filters.items():
            if isinstance(frame, frame_type):
                # If filter is None, log all instances of this frame type
                if filter_config is None:
                    return True

                # Otherwise, check the specific filter
                service_type, endpoint = filter_config

                if endpoint == FrameEndpoint.SOURCE:
                    return isinstance(src, service_type)
                elif endpoint == FrameEndpoint.DESTINATION:
                    return isinstance(dst, service_type)

        return False

    async def on_push_frame(self, data: FramePushed):
        """Process a frame being pushed into the pipeline.

        Logs frame details to the console with all relevant fields and values.

        Args:
            data: Event data containing the frame, source, destination, direction, and timestamp.
        """
        src = data.source
        dst = data.destination
        frame = data.frame
        direction = data.direction
        timestamp = data.timestamp

        # Check if we should log this frame
        if not self._should_log_frame(frame, src, dst):
            return

        # Format direction arrow
        arrow = "→" if direction == FrameDirection.DOWNSTREAM else "←"

        time_sec = timestamp / 1_000_000_000
        class_name = frame.__class__.__name__

        # Build frame representation
        frame_details = []

        # If dataclass, extract fields
        if is_dataclass(frame):
            for field in fields(frame):
                if field.name in self.exclude_fields:
                    continue

                value = getattr(frame, field.name)
                if value is None:
                    continue

                formatted_value = self._format_value(value)
                frame_details.append(f"{field.name}: {formatted_value}")

        # Format the message
        if frame_details:
            details = ", ".join(frame_details)
            message = f"{class_name} {details} at {time_sec:.2f}s"
        else:
            message = f"{class_name} at {time_sec:.2f}s"

        # Log the message
        logger.debug(f"{src} {arrow} {dst}: {message}")
