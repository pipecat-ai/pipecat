#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Context management for AWS Nova Sonic LLM service.

This module provides specialized context aggregators and message handling for AWS Nova Sonic,
including conversation history management and role-specific message processing.
"""

import warnings

from pipecat.services.aws.nova_sonic.context import *

with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "Types in pipecat.services.aws_nova_sonic.context are deprecated. "
        "Please use the equivalent types from "
        "pipecat.services.aws.nova_sonic.context instead.",
        DeprecationWarning,
        stacklevel=2,
    )
