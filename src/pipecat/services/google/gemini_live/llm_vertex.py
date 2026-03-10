#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Deprecated: use ``pipecat.services.google.gemini_live.vertex.llm`` instead."""

import warnings

warnings.warn(
    "Module `pipecat.services.google.gemini_live.llm_vertex` is deprecated, "
    "use `pipecat.services.google.gemini_live.vertex.llm` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from pipecat.services.google.gemini_live.vertex.llm import *  # noqa: E402, F401, F403
