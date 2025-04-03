#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import sys

from pipecat.services import DeprecatedModuleProxy

from .image import *
from .llm import *
from .stt import *
from .tts import *

sys.modules[__name__] = DeprecatedModuleProxy(globals(), "openai", "openai.[image,llm,stt,tts]")
