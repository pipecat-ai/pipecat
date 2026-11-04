#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Tone's reader: how each user turn was meant, read from its words.

.. deprecated:: 1.16.0
    Moved to :mod:`pipecat.expressive.processors.tone_reader`.
    Will be removed in 2.0.0.
"""

from pipecat.expressive.processors.tone_reader import PipecatToneReader
from pipecat.utils.deprecation import warn_deprecated

__all__ = ["PipecatToneReader"]

warn_deprecated(
    "`pipecat.processors.tone.reader` is deprecated since 1.16.0 and will be removed in 2.0.0. "
    "Use `pipecat.expressive.processors.tone_reader` instead.",
    stacklevel=2,
)
