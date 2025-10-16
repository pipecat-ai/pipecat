#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Custom frames for AWS Nova Sonic LLM service."""

from dataclasses import dataclass

from pipecat.frames.frames import DataFrame, FunctionCallResultFrame


@dataclass
class AWSNovaSonicFunctionCallResultFrame(DataFrame):
    """Frame containing function call result for AWS Nova Sonic processing.

    This frame wraps a standard function call result frame to enable
    AWS Nova Sonic-specific handling and context updates.

    Parameters:
        result_frame: The underlying function call result frame.
    """

    result_frame: FunctionCallResultFrame
