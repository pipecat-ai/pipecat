#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import sys
from importlib.metadata import version

from loguru import logger

__version__ = version("pipecat-ai")

logger.info(f"ᓚᘏᗢ Pipecat {__version__} (Python {sys.version}) ᓚᘏᗢ")
