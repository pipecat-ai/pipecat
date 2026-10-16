#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Grok Realtime LLM service.

.. deprecated:: 0.0.108
    Use :mod:`pipecat.services.xai.realtime.llm` instead.
    Will be removed in 2.0.0.
"""

from pipecat.services.xai.realtime.llm import *  # noqa: F401,F403
from pipecat.utils.deprecation import warn_deprecated

warn_deprecated(
    "`pipecat.services.grok.realtime.llm` is deprecated since 0.0.108 and will be removed in "
    "2.0.0. Use `pipecat.services.xai.realtime.llm` instead.",
    stacklevel=2,
)
