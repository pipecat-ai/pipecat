#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Tone's tagger model, on the CPU.

.. deprecated:: 1.16.0
    Moved to :mod:`pipecat.expressive.processors.tone_tagger.model`.
    Will be removed in 2.0.0.
"""

from pipecat.expressive.processors.tone_tagger.model import *  # noqa: F401,F403
from pipecat.utils.deprecation import warn_deprecated

warn_deprecated(
    "`pipecat.processors.tone.tagger.model` is deprecated since 1.16.0 and will be removed in "
    "2.0.0. Use `pipecat.expressive.processors.tone_tagger.model` instead.",
    stacklevel=2,
)
