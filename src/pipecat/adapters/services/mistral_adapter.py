#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Mistral LLM adapter for Pipecat.

.. deprecated:: 1.14.0
    Moved to :mod:`pipecat.adapters.llm.services.mistral_adapter`.
    Will be removed in 2.0.0.
"""

from pipecat.adapters.llm.services.mistral_adapter import *  # noqa: F401,F403
from pipecat.utils.deprecation import warn_deprecated

warn_deprecated(
    "`pipecat.adapters.services.mistral_adapter` is deprecated since 1.14.0 and will be "
    "removed in 2.0.0. Use `pipecat.adapters.llm.services.mistral_adapter` instead.",
    stacklevel=2,
)
