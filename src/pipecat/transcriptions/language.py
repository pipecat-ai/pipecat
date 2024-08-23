#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import sys

from enum import Enum

if sys.version_info < (3, 11):
    class StrEnum(str, Enum):
        def __new__(cls, value):
            obj = str.__new__(cls, value)
            obj._value_ = value
            return obj
else:
    from enum import StrEnum


class Language(StrEnum):
    EN = "en"
    ES = "es"
