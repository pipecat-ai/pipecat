#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Eval session, at its former module path.

.. deprecated:: 1.9.0
    Moved to :mod:`pipecat.evals.eval_session`.
    Will be removed in 2.0.0.
"""

import warnings

from pipecat.evals.eval_session import *  # noqa: F401,F403

with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "`pipecat.evals.harness` is deprecated since 1.9.0 and will be removed in 2.0.0. "
        "Use `pipecat.evals.eval_session` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
