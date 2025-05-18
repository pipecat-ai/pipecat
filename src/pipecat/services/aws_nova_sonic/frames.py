#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from dataclasses import dataclass

from pipecat.frames.frames import DataFrame, FunctionCallResultFrame


@dataclass
class AWSNovaSonicFunctionCallResultFrame(DataFrame):
    result_frame: FunctionCallResultFrame
