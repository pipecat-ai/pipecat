#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base pipeline implementation for frame processing."""

from abc import abstractmethod
from typing import List

from pipecat.processors.frame_processor import FrameProcessor


class BasePipeline(FrameProcessor):
    """Base class for all pipeline implementations.

    Provides the foundation for pipeline processors that need to support
    metrics collection from their contained processors.
    """

    def __init__(self):
        """Initialize the base pipeline."""
        super().__init__()

    @abstractmethod
    def processors_with_metrics(self) -> List[FrameProcessor]:
        """Return processors that can generate metrics.

        Implementing classes should collect and return all processors within
        their pipeline that support metrics generation.

        Returns:
            List of frame processors that support metrics collection.
        """
        pass
