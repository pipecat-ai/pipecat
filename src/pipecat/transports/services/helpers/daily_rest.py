#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Daily REST Helpers.

Methods that wrap the Daily API to create rooms, check room URLs, and get meeting tokens.
"""

import warnings

from pipecat.transports.daily.utils import *

with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "Module `pipecat.transports.services.helpers.daily_rest` is deprecated, "
        "use `pipecat.transports.daily.utils` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
