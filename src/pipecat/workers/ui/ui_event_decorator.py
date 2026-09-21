#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The ``@ui_event`` decorator, at its former module path.

.. deprecated:: 1.12.0
    Moved to :mod:`pipecat.workers.ui_event_decorator`. Will be removed in
    2.0.0.
"""

import warnings

from pipecat.workers.ui_event_decorator import _collect_ui_event_handlers, ui_event

__all__ = ["_collect_ui_event_handlers", "ui_event"]

with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "`pipecat.workers.ui.ui_event_decorator` is deprecated since 1.12.0 and will be "
        "removed in 2.0.0. Use `pipecat.workers.ui_event_decorator` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
