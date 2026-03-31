#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Deprecated: use ``pipecat.services.google.openai.llm`` instead."""

import warnings

warnings.warn(
    "Module `pipecat.services.google.llm_openai` is deprecated, "
    "use `pipecat.services.google.openai.llm` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from pipecat.services.google.openai.llm import *  # noqa: E402, F401, F403
