#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import sys

from pipecat.services import DeprecatedModuleProxy

from .llm import *

sys.modules[__name__] = DeprecatedModuleProxy(globals(), "ollama", "ollama.llm")
