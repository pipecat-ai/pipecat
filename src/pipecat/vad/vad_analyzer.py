#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from loguru import logger

logger.warning("DEPRECATED: Package `pipecat.vad` is deprecated, use `pipecat.audio.vad` instead.")

from ..audio.vad.vad_analyzer import VADAnalyzer, VADParams, VADState
