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
    BG = "bg"  # Bulgarian
    CA = "ca"  # Catalan
    ZH = "zh"  # Chinese simplified
    ZH_TW = "zh-TW"  # Chinese traditional
    CS = "cs"  # Czech
    DA = "da"  # Danish
    NL = "nl"  # Dutch
    EN = "en"  # English
    EN_US = "en-US"  # English (USA)
    EN_AU = "en-AU"  # English (Australia)
    EN_GB = "en-GB"  # English (Great Britain)
    EN_NZ = "en-NZ"  # English (New Zealand)
    EN_IN = "en-IN"  # English (India)
    ET = "et"  # Estonian
    FI = "fi"  # Finnish
    NL_BE = "nl-BE"  # Flemmish
    FR = "fr"  # French
    FR_CA = "fr-CA"  # French (Canada)
    DE = "de"  # German
    DE_CH = "de-CH"  # German (Switzerland)
    EL = "el"  # Greek
    HI = "hi"  # Hindi
    HU = "hu"  # Hungarian
    ID = "id"  # Indonesian
    IT = "it"  # Italian
    JA = "ja"  # Japanese
    KO = "ko"  # Korean
    LV = "lv"  # Latvian
    LT = "lt"  # Lithuanian
    MS = "ms"  # Malay
    NO = "no"  # Norwegian
    PL = "pl"  # Polish
    PT = "pt"  # Portuguese
    PT_BR = "pt-BR"  # Portuguese (Brazil)
    RO = "ro"  # Romanian
    RU = "ru"  # Russian
    SK = "sk"  # Slovak
    ES = "es"  # Spanish
    SV = "sv"  # Swedish
    TH = "th"  # Thai
    TR = "tr"  # Turkish
    UK = "uk"  # Ukrainian
    VI = "vi"  # Vietnamese
