#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Tone's tagger: expression tags for an LLM's plain replies.

.. deprecated:: 1.16.0
    Moved to :mod:`pipecat.expressive.processors.tone_tagger`.
    Will be removed in 2.0.0.
"""

from pipecat.expressive.processors.tone_tagger import PipecatToneTagger
from pipecat.utils.deprecation import warn_deprecated

__all__ = ["PipecatToneTagger"]

warn_deprecated(
    "`pipecat.processors.tone.tagger` is deprecated since 1.16.0 and will be removed in 2.0.0. "
    "Use `pipecat.expressive.processors.tone_tagger` instead.",
    stacklevel=2,
)
