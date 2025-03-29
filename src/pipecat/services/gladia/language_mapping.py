#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Optional

from pipecat.transcriptions.language import Language


def language_to_gladia_language(language: Language) -> Optional[str]:
    """Convert a Language enum to Gladia's language code format.

    Args:
        language: The Language enum value to convert

    Returns:
        The Gladia language code string or None if not supported
    """
    BASE_LANGUAGES = {
        Language.AF: "af",
        Language.AM: "am",
        Language.AR: "ar",
        Language.AS: "as",
        Language.AZ: "az",
        Language.BG: "bg",
        Language.BN: "bn",
        Language.BS: "bs",
        Language.CA: "ca",
        Language.CS: "cs",
        Language.CY: "cy",
        Language.DA: "da",
        Language.DE: "de",
        Language.EL: "el",
        Language.EN: "en",
        Language.ES: "es",
        Language.ET: "et",
        Language.EU: "eu",
        Language.FA: "fa",
        Language.FI: "fi",
        Language.FR: "fr",
        Language.GA: "ga",
        Language.GL: "gl",
        Language.GU: "gu",
        Language.HE: "he",
        Language.HI: "hi",
        Language.HR: "hr",
        Language.HU: "hu",
        Language.HY: "hy",
        Language.ID: "id",
        Language.IS: "is",
        Language.IT: "it",
        Language.JA: "ja",
        Language.JV: "jv",
        Language.KA: "ka",
        Language.KK: "kk",
        Language.KM: "km",
        Language.KN: "kn",
        Language.KO: "ko",
        Language.LO: "lo",
        Language.LT: "lt",
        Language.LV: "lv",
        Language.MK: "mk",
        Language.ML: "ml",
        Language.MN: "mn",
        Language.MR: "mr",
        Language.MS: "ms",
        Language.MT: "mt",
        Language.MY: "my",
        Language.NE: "ne",
        Language.NL: "nl",
        Language.NO: "no",
        Language.OR: "or",
        Language.PA: "pa",
        Language.PL: "pl",
        Language.PS: "ps",
        Language.PT: "pt",
        Language.RO: "ro",
        Language.RU: "ru",
        Language.SI: "si",
        Language.SK: "sk",
        Language.SL: "sl",
        Language.SO: "so",
        Language.SQ: "sq",
        Language.SR: "sr",
        Language.SU: "su",
        Language.SV: "sv",
        Language.SW: "sw",
        Language.TA: "ta",
        Language.TE: "te",
        Language.TH: "th",
        Language.TR: "tr",
        Language.UK: "uk",
        Language.UR: "ur",
        Language.UZ: "uz",
        Language.VI: "vi",
        Language.ZH: "zh",
        Language.ZU: "zu",
    }

    result = BASE_LANGUAGES.get(language)

    # If not found in base languages, try to find the base language from a variant
    if not result:
        # Convert enum value to string and get the base language part (e.g. es-ES -> es)
        lang_str = str(language.value)
        base_code = lang_str.split("-")[0].lower()
        # Look up the base code in our supported languages
        result = base_code if base_code in BASE_LANGUAGES.values() else None

    return result
