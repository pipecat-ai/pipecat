#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "Module `pipecat.services.together` is deprecated, use `pipecat.services.together.llm` instead",
        DeprecationWarning,
    )

from .llm import *
