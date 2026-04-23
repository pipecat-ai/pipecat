#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Helper functions for Soniox STT and TTS services."""

from pipecat.transcriptions.language import Language, resolve_language


def language_to_soniox_language(language: Language) -> str:
    """Convert a Pipecat Language to a Soniox language code.

    For a list of all supported languages for STT see:
    https://soniox.com/docs/stt/concepts/supported-languages
    and for supported languages for TTS see:
    https://speechdev.soniox.com/docs/tts/concepts/languages
    """
    LANGUAGE_MAP = {
        Language.AF: "af",
        Language.AR: "ar",
        Language.AZ: "az",
        Language.BE: "be",
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
        Language.GL: "gl",
        Language.GU: "gu",
        Language.HE: "he",
        Language.HI: "hi",
        Language.HR: "hr",
        Language.HU: "hu",
        Language.ID: "id",
        Language.IT: "it",
        Language.JA: "ja",
        Language.KA: "ka",
        Language.KK: "kk",
        Language.KN: "kn",
        Language.KO: "ko",
        Language.LT: "lt",
        Language.LV: "lv",
        Language.MK: "mk",
        Language.ML: "ml",
        Language.MR: "mr",
        Language.MS: "ms",
        Language.NL: "nl",
        Language.NO: "no",
        Language.PA: "pa",
        Language.PL: "pl",
        Language.PT: "pt",
        Language.RO: "ro",
        Language.RU: "ru",
        Language.SK: "sk",
        Language.SL: "sl",
        Language.SQ: "sq",
        Language.SR: "sr",
        Language.SV: "sv",
        Language.SW: "sw",
        Language.TA: "ta",
        Language.TE: "te",
        Language.TH: "th",
        Language.TL: "tl",
        Language.TR: "tr",
        Language.UK: "uk",
        Language.UR: "ur",
        Language.VI: "vi",
        Language.ZH: "zh",
    }
    return resolve_language(language, LANGUAGE_MAP, use_base_code=True)
