#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "Package `pipecat.vad` is deprecated, use `pipecat.audio.vad` instead", DeprecationWarning
    )

from ..audio.vad.vad_analyzer import VADAnalyzer, VADParams, VADState
