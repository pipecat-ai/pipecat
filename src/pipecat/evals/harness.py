#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Eval session, at its former module path.

.. deprecated:: 1.9.0
    Moved to :mod:`pipecat.evals.session` (the session and its default
    timeout) and :mod:`pipecat.evals.script_session` (the rest).
    Will be removed in 2.0.0.
"""

# Everything the module used to define or import at its old path, from where it
# lives now; the star import covers the scripted session's constants.
from pipecat.evals.client import BOT_READY_TIMEOUT_S  # noqa: F401
from pipecat.evals.results import (  # noqa: F401
    FAILURE_KINDS,
    TURN_STATUSES,
    EvalAssertionFailure,
    EvalResult,
    EvalTurnProgress,
    EvalTurnResult,
)
from pipecat.evals.script import EvalScenario, EvalTurn  # noqa: F401
from pipecat.evals.script_driver import SEND_AFTER_MAX_WAIT_S, SEND_AFTER_POLL_S  # noqa: F401
from pipecat.evals.script_session import *  # noqa: F401,F403
from pipecat.evals.session import DEFAULT_EVENT_TIMEOUT_MS, EvalSession  # noqa: F401
from pipecat.utils.deprecation import warn_deprecated

warn_deprecated(
    "`pipecat.evals.harness` is deprecated since 1.9.0 and will be removed in 2.0.0. "
    "Use `pipecat.evals.session` and `pipecat.evals.script_session` instead. "
    "`EvalSession` and `DEFAULT_EVENT_TIMEOUT_MS` are in the first, the rest in the second.",
    stacklevel=2,
)
