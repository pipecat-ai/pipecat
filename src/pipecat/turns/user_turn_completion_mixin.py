#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Turn completion markers and configuration, at their former module path.

.. deprecated:: 1.16.0
    Moved to :mod:`pipecat.turns.user_turn_completion`. Will be removed in
    2.0.0.
"""

from pipecat.turns.user_turn_completion import *  # noqa: F401,F403
from pipecat.utils.deprecation import warn_deprecated

warn_deprecated(
    "`pipecat.turns.user_turn_completion_mixin` is deprecated since 1.16.0 and will be "
    "removed in 2.0.0. Use `pipecat.turns.user_turn_completion` instead.",
    stacklevel=2,
)
