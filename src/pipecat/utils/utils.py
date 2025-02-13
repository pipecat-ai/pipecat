#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import collections
import itertools
import threading

_COUNTS = collections.defaultdict(itertools.count)
_COUNTS_LOCK = threading.Lock()
_ID = itertools.count()
_ID_LOCK = threading.Lock()


def obj_id() -> int:
    """Generate a unique id for an object.

    >>> obj_id()
    0
    >>> obj_id()
    1
    >>> obj_id()
    2
    """
    with _ID_LOCK:
        return next(_ID)


def obj_count(obj) -> int:
    """Generate a unique id for an object.

    >>> obj_count(object())
    0
    >>> obj_count(object())
    1
    >>> new_type = type('NewType', (object,), {})
    >>> obj_count(new_type())
    0
    """
    with _COUNTS_LOCK:
        return next(_COUNTS[obj.__class__.__name__])
