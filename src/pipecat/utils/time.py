#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import datetime


def time_now_iso8601() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")
