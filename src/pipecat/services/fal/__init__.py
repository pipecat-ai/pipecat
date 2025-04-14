#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import sys

from pipecat.services import DeprecatedModuleProxy

from .image import *
from .stt import *

sys.modules[__name__] = DeprecatedModuleProxy(globals(), "fal", "fal.[image,stt]")
