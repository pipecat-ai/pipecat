#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base adapter for LLM provider integration.

.. deprecated:: 1.14.0
    Moved to :mod:`pipecat.adapters.llm.base_llm_adapter`.
    Will be removed in 2.0.0.
"""

from pipecat.adapters.llm.base_llm_adapter import *  # noqa: F401,F403
from pipecat.utils.deprecation import warn_deprecated

warn_deprecated(
    "`pipecat.adapters.base_llm_adapter` is deprecated since 1.14.0 and will be removed in "
    "2.0.0. Use `pipecat.adapters.llm.base_llm_adapter` instead.",
    stacklevel=2,
)
