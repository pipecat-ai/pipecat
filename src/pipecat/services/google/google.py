#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "Module `pipecat.services.google.google` is deprecated, use `pipecat.services.google.[frames,image,llm,llm_openai,llm_vertex,rtvi,stt,tts]` instead",
        DeprecationWarning,
    )

from .frames import *
from .image import *
from .llm import *
from .llm_openai import *
from .llm_vertex import *
from .rtvi import *
from .stt import *
from .tts import *
