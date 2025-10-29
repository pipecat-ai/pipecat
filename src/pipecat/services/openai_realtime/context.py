#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI Realtime LLM context and aggregator implementations.

.. deprecated:: 0.0.91
    OpenAI Realtime no longer uses types from this module under the hood.
    It now uses `LLMContext` and `LLMContextAggregatorPair`.
    Using the new patterns should allow you to not need types from this module.

    See deprecation warning in pipecat.services.openai.realtime.context for
    more details.
"""

from pipecat.services.openai.realtime.context import *
