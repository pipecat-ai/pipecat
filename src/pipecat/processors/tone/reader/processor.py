#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The processor that reads how each user turn was meant with Pipecat Tone.

.. deprecated:: 1.16.0
    Moved to :mod:`pipecat.expressive.processors.tone_reader`.
    Will be removed in 2.0.0.
"""

from pipecat.expressive.processors.tone_reader import *  # noqa: F401,F403
from pipecat.utils.deprecation import warn_deprecated

warn_deprecated(
    "`pipecat.processors.tone.reader.processor` is deprecated since 1.16.0 and will be removed "
    "in 2.0.0. Use `pipecat.expressive.processors.tone_reader` instead.",
    stacklevel=2,
)
