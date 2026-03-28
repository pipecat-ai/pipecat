#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base notifier interface for Pipecat."""

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "Package pipecat.sync is deprecated, use pipecat.utils.sync instead.",
        DeprecationWarning,
        stacklevel=2,
    )
