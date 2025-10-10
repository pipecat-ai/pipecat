#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Custom frames for AWS Nova Sonic LLM service."""

import warnings

from pipecat.services.aws.nova_sonic.frames import *

with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "Types in pipecat.services.aws_nova_sonic.frames are deprecated. "
        "Please use the equivalent types from "
        "pipecat.services.aws.nova_sonic.frames instead.",
        DeprecationWarning,
        stacklevel=2,
    )
