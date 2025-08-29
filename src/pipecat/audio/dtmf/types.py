# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""This module defines generic type for DTMS.

It defines the `KeypadEntry` enumeration, representing dual-tone multi-frequency
(DTMF) keypad entries for phone system integration. Each entry corresponds to a
key on the telephone keypad, facilitating the handling of input in
telecommunication applications.
"""

from enum import Enum


class KeypadEntry(str, Enum):
    """DTMF keypad entries for phone system integration.

    Parameters:
        ONE: Number key 1.
        TWO: Number key 2.
        THREE: Number key 3.
        FOUR: Number key 4.
        FIVE: Number key 5.
        SIX: Number key 6.
        SEVEN: Number key 7.
        EIGHT: Number key 8.
        NINE: Number key 9.
        ZERO: Number key 0.
        POUND: Pound/hash key (#).
        STAR: Star/asterisk key (*).
    """

    ONE = "1"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    ZERO = "0"

    POUND = "#"
    STAR = "*"
