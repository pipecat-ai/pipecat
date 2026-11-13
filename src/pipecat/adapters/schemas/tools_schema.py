#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tools schema definitions for function calling adapters.

.. deprecated:: 1.14.0
    Moved to :mod:`pipecat.adapters.llm.schemas.tools_schema`.
    Will be removed in 2.0.0.
"""

from pipecat.adapters.llm.schemas.tools_schema import *  # noqa: F401,F403
from pipecat.utils.deprecation import warn_deprecated

warn_deprecated(
    "`pipecat.adapters.schemas.tools_schema` is deprecated since 1.14.0 and will be removed"
    " in 2.0.0. Use `pipecat.adapters.llm.schemas.tools_schema` instead.",
    stacklevel=2,
)
