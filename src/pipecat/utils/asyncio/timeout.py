#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Implementation of Python's `asyncio.wait_for()`.

This module uses `wait_for2` package to implement `asyncio.wait_for()` for
Python < 3.12.

In Python 3.12, `asyncio.wait_for()` is implemented in terms of
`asyncio.timeout()` which fixed a bunch of issues. However, this was never
backported (because of the lack of `async.timeout()`) and there are still many
remainig issues, specially in Python 3.10, in `async.wait_for()`.

See https://github.com/python/cpython/pull/98518
"""

import sys

if sys.version_info >= (3, 12):
    import asyncio

    wait_for = asyncio.wait_for
else:
    import wait_for2

    wait_for = wait_for2.wait_for
