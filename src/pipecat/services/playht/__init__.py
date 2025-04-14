#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import sys

from pipecat.services import DeprecatedModuleProxy

from .tts import *

sys.modules[__name__] = DeprecatedModuleProxy(globals(), "playht", "playht.tts")
