#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Context management for AWS Nova Sonic LLM service.

This module provides specialized context aggregators and message handling for AWS Nova Sonic,
including conversation history management and role-specific message processing.

.. deprecated:: 0.0.91
    AWS Nova Sonic no longer uses types from this module under the hood.
    It now uses `LLMContext` and `LLMContextAggregatorPair`.
    Using the new patterns should allow you to not need types from this module.

    See deprecation warning in pipecat.services.aws.nova_sonic.context for more
    details.
"""

from pipecat.services.aws.nova_sonic.context import *
