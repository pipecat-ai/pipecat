#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Eval session, at its former module path.

.. deprecated:: 1.9.0
    Moved to :mod:`pipecat.evals.script_session`.
    Will be removed in 2.0.0.
"""

import warnings

# Everything the module used to define or import at its old path, from where it
# lives now; the star import covers the session itself and its constants.
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

with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "`pipecat.evals.harness` is deprecated since 1.9.0 and will be removed in 2.0.0. "
        "Use `pipecat.evals.script_session` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
