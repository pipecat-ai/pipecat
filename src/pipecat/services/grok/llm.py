#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Grok LLM service implementation.

.. deprecated:: 0.0.108
    Use :mod:`pipecat.services.xai.llm` instead.
    Will be removed in 2.0.0.
"""

import warnings

from pipecat.services.xai.llm import *  # noqa: F401,F403

with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "pipecat.services.grok.llm is deprecated. Please use pipecat.services.xai.llm instead.",
        DeprecationWarning,
        stacklevel=2,
    )
