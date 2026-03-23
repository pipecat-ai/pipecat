#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Deprecated: use ``pipecat.services.deepgram.sagemaker.tts`` instead."""

import warnings

warnings.warn(
    "Module `pipecat.services.deepgram.tts_sagemaker` is deprecated, "
    "use `pipecat.services.deepgram.sagemaker.tts` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from pipecat.services.deepgram.sagemaker.tts import *  # noqa: E402, F401, F403
