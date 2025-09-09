#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base pipeline implementation for frame processing."""

from pipecat.processors.frame_processor import FrameProcessor


class BasePipeline(FrameProcessor):
    """Base class for all pipeline implementations."""

    def __init__(self, **kwargs):
        """Initialize the base pipeline."""
        super().__init__(**kwargs)
