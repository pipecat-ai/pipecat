#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""HTTP client for Jev, TypeSafe's hosted classification model.

.. deprecated:: 1.13.0
    Moved to :mod:`pipecat.classifiers.typesafe.jev.client`.
    Will be removed in 2.0.0.
"""

from pipecat.classifiers.typesafe.jev.client import *  # noqa: F401,F403
from pipecat.utils.deprecation import warn_deprecated

warn_deprecated(
    "`pipecat.classifiers.jev.client` is deprecated since 1.13.0 and will be removed in "
    "2.0.0. Use `pipecat.classifiers.typesafe.jev.client` instead.",
    stacklevel=2,
)
