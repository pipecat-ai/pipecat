#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Utility functions for object identification and counting.

This module provides thread-safe utilities for generating unique identifiers
and maintaining per-class instance counts across the Pipecat framework.
"""

import collections
import itertools
import threading

_COUNTS = collections.defaultdict(itertools.count)
_COUNTS_LOCK = threading.Lock()
_ID = itertools.count()
_ID_LOCK = threading.Lock()


def obj_id() -> int:
    """Generate a unique id for an object.

    Returns:
        A unique integer identifier that increments globally across all objects.
    """
    with _ID_LOCK:
        return next(_ID)


def obj_count(obj) -> int:
    """Generate a unique count for an object based on its class.

    Args:
        obj: The object instance to count.

    Returns:
        A unique integer count that increments per class type.
    """
    with _COUNTS_LOCK:
        return next(_COUNTS[obj.__class__.__name__])
