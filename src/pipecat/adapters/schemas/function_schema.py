#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Function schema utilities for AI tool definitions.

.. deprecated:: 1.14.0
    Moved to :mod:`pipecat.adapters.llm.schemas.function_schema`.
    Will be removed in 2.0.0.
"""

from pipecat.adapters.llm.schemas.function_schema import *  # noqa: F401,F403
from pipecat.utils.deprecation import warn_deprecated

warn_deprecated(
    "`pipecat.adapters.schemas.function_schema` is deprecated since 1.14.0 and will be "
    "removed in 2.0.0. Use `pipecat.adapters.llm.schemas.function_schema` instead.",
    stacklevel=2,
)
