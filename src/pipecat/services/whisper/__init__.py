#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import sys

from pipecat.services import DeprecatedModuleProxy

from .base_stt import *
from .stt import *

sys.modules[__name__] = DeprecatedModuleProxy(globals(), "whisper", "whisper.stt")
