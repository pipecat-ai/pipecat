#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Google services module for Pipecat."""

import sys

from pipecat.services import DeprecatedModuleProxy

from .frames import *
from .image import *
from .llm import *
from .openai import *
from .rtvi import *
from .stt import *
from .tts import *
from .vertex import *

sys.modules[__name__] = DeprecatedModuleProxy(
    globals(), "google", "google.[frames,image,llm,openai,vertex,rtvi,stt,tts]"
)
