#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Google services module for Pipecat."""

import sys

from pipecat.services import DeprecatedModuleProxy

from .frames import *
from .image import *
from .llm import *
from .llm_openai import *
from .llm_vertex import *
from .rtvi import *
from .stt import *
from .tts import *

sys.modules[__name__] = DeprecatedModuleProxy(
    globals(), "google", "google.[frames,image,llm,llm_openai,llm_vertex,rtvi,stt,tts]"
)
