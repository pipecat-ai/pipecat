#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Grok Realtime event models.

.. deprecated:: 0.0.108
    This module is deprecated. Please use pipecat.services.xai.realtime.events instead.
"""

import warnings

from pipecat.services.xai.realtime.events import *  # noqa: F401,F403

with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "pipecat.services.grok.realtime.events is deprecated. "
        "Please use pipecat.services.xai.realtime.events instead.",
        DeprecationWarning,
        stacklevel=2,
    )
