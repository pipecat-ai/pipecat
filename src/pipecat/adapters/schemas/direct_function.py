#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Direct function wrapper utilities for LLM function calling.

.. deprecated:: 1.14.0
    Moved to :mod:`pipecat.adapters.llm.schemas.direct_function`.
    Will be removed in 2.0.0.
"""

from pipecat.adapters.llm.schemas.direct_function import *  # noqa: F401,F403
from pipecat.utils.deprecation import warn_deprecated

warn_deprecated(
    "`pipecat.adapters.schemas.direct_function` is deprecated since 1.14.0 and will be "
    "removed in 2.0.0. Use `pipecat.adapters.llm.schemas.direct_function` instead.",
    stacklevel=2,
)
