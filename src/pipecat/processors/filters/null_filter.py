#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from pipecat.processors.frame_processor import FrameProcessor


class NullFilter(FrameProcessor):
    """This filter doesn't allow passing any frames up or downstream."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
