#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Deprecated: use ``pipecat.services.deepgram.sagemaker.stt`` instead."""

import warnings

warnings.warn(
    "Module `pipecat.services.deepgram.stt_sagemaker` is deprecated, "
    "use `pipecat.services.deepgram.sagemaker.stt` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from pipecat.services.deepgram.sagemaker.stt import *  # noqa: E402, F401, F403
