#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Tone's reader model, on the CPU.

.. deprecated:: 1.16.0
    Moved to :mod:`pipecat.classifiers.tone.model`.
    Will be removed in 2.0.0.
"""

from pipecat.classifiers.tone.model import *  # noqa: F401,F403
from pipecat.utils.deprecation import warn_deprecated

warn_deprecated(
    "`pipecat.processors.tone.reader.model` is deprecated since 1.16.0 and will be removed in "
    "2.0.0. Use `pipecat.classifiers.tone.model` instead.",
    stacklevel=2,
)
