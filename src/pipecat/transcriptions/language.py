#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Language code enumerations for Pipecat.

This module provides comprehensive language code constants following ISO 639
and BCP 47 standards, supporting both language-only and language-region
combinations for various speech and text processing services.
"""

import sys
from enum import Enum

if sys.version_info < (3, 11):

    class StrEnum(str, Enum):
        """String enumeration base class for Python < 3.11 compatibility."""

        def __new__(cls, value):
            """Create a new instance of the StrEnum."""
            obj = str.__new__(cls, value)
            obj._value_ = value
            return obj
else:
    from enum import StrEnum


class Language(StrEnum):
    """Language codes for speech and text processing services.

    Provides comprehensive language code constants following ISO 639 and BCP 47
    standards. Includes both language-only codes (e.g., 'en') and language-region
    combinations (e.g., 'en-US') to support various speech synthesis, recognition,
    and translation services.
    """

    # Afrikaans
    AF = "af"
    AF_ZA = "af-ZA"

    # Amharic
    AM = "am"
    AM_ET = "am-ET"

    # Arabic
    AR = "ar"
    AR_AE = "ar-AE"
    AR_BH = "ar-BH"
    AR_DZ = "ar-DZ"
    AR_EG = "ar-EG"
    AR_IQ = "ar-IQ"
    AR_JO = "ar-JO"
    AR_KW = "ar-KW"
    AR_LB = "ar-LB"
    AR_LY = "ar-LY"
    AR_MA = "ar-MA"
    AR_OM = "ar-OM"
    AR_QA = "ar-QA"
    AR_SA = "ar-SA"
    AR_SY = "ar-SY"
    AR_TN = "ar-TN"
    AR_YE = "ar-YE"

    # Assamese
    AS = "as"
    AS_IN = "as-IN"

    # Asturian
    AST = "ast"

    # Azerbaijani
    AZ = "az"
    AZ_AZ = "az-AZ"

    # Bashkir
    BA = "ba"

    # Belarusian
    BE = "be"

    # Bulgarian
    BG = "bg"
    BG_BG = "bg-BG"

    # Bengali
    BN = "bn"
    BN_BD = "bn-BD"
    BN_IN = "bn-IN"

    # Tibetan
    BO = "bo"

    # Breton
    BR = "br"

    # Bosnian
    BS = "bs"
    BS_BA = "bs-BA"

    # Catalan
    CA = "ca"
    CA_ES = "ca-ES"

    # Cebuano
    CEB = "ceb"

    # Mandarin Chinese
    CMN = "cmn"
    CMN_CN = "cmn-CN"

    # Czech
    CS = "cs"
    CS_CZ = "cs-CZ"

    # Welsh
    CY = "cy"
    CY_GB = "cy-GB"

    # Danish
    DA = "da"
    DA_DK = "da-DK"

    # German
    DE = "de"
    DE_AT = "de-AT"
    DE_CH = "de-CH"
    DE_DE = "de-DE"

    # Greek
    EL = "el"
    EL_GR = "el-GR"

    # English
    EN = "en"
    EN_AU = "en-AU"
    EN_CA = "en-CA"
    EN_GB = "en-GB"
    EN_GH = "en-GH"
    EN_HK = "en-HK"
    EN_IE = "en-IE"
    EN_IN = "en-IN"
    EN_KE = "en-KE"
    EN_NG = "en-NG"
    EN_NZ = "en-NZ"
    EN_PH = "en-PH"
    EN_SG = "en-SG"
    EN_TZ = "en-TZ"
    EN_US = "en-US"
    EN_ZA = "en-ZA"

    # Esperanto
    EO = "eo"

    # Spanish
    ES = "es"
    ES_AR = "es-AR"
    ES_BO = "es-BO"
    ES_CL = "es-CL"
    ES_CO = "es-CO"
    ES_CR = "es-CR"
    ES_CU = "es-CU"
    ES_DO = "es-DO"
    ES_EC = "es-EC"
    ES_ES = "es-ES"
    ES_GQ = "es-GQ"
    ES_GT = "es-GT"
    ES_HN = "es-HN"
    ES_MX = "es-MX"
    ES_NI = "es-NI"
    ES_PA = "es-PA"
    ES_PE = "es-PE"
    ES_PR = "es-PR"
    ES_PY = "es-PY"
    ES_SV = "es-SV"
    ES_US = "es-US"
    ES_UY = "es-UY"
    ES_VE = "es-VE"

    # Estonian
    ET = "et"
    ET_EE = "et-EE"

    # Basque
    EU = "eu"
    EU_ES = "eu-ES"

    # Persian
    FA = "fa"
    FA_IR = "fa-IR"

    # Fulah
    FF = "ff"

    # Finnish
    FI = "fi"
    FI_FI = "fi-FI"

    # Filipino
    FIL = "fil"
    FIL_PH = "fil-PH"

    # Faroese
    FO = "fo"

    # French
    FR = "fr"
    FR_BE = "fr-BE"
    FR_CA = "fr-CA"
    FR_CH = "fr-CH"
    FR_FR = "fr-FR"

    # Irish
    GA = "ga"
    GA_IE = "ga-IE"

    # Gaelic
    GD = "gd"

    # Galician
    GL = "gl"
    GL_ES = "gl-ES"

    # Gujarati
    GU = "gu"
    GU_IN = "gu-IN"

    # Hausa
    HA = "ha"

    # Hawaiian
    HAW = "haw"

    # Hebrew
    HE = "he"
    HE_IL = "he-IL"

    # Hindi
    HI = "hi"
    HI_IN = "hi-IN"

    # Croatian
    HR = "hr"
    HR_HR = "hr-HR"

    # Haitian Creole
    HT = "ht"

    # Hungarian
    HU = "hu"
    HU_HU = "hu-HU"

    # Armenian
    HY = "hy"
    HY_AM = "hy-AM"

    # Indonesian
    ID = "id"
    ID_ID = "id-ID"

    # Igbo
    IG = "ig"

    # Icelandic
    IS = "is"
    IS_IS = "is-IS"

    # Italian
    IT = "it"
    IT_IT = "it-IT"
    IT_CH = "it-CH"

    # Inuktitut
    IU_CANS = "iu-Cans"
    IU_CANS_CA = "iu-Cans-CA"
    IU_LATN = "iu-Latn"
    IU_LATN_CA = "iu-Latn-CA"

    # Japanese
    JA = "ja"
    JA_JP = "ja-JP"

    # Javanese
    JV = "jv"
    JV_ID = "jv-ID"
    JW = "jw"  # Fal requires for Javanese

    # Georgian
    KA = "ka"
    KA_GE = "ka-GE"

    # Kabuverdianu
    KEA = "kea"

    # Kazakh
    KK = "kk"
    KK_KZ = "kk-KZ"

    # Khmer
    KM = "km"
    KM_KH = "km-KH"

    # Kannada
    KN = "kn"
    KN_IN = "kn-IN"

    # Korean
    KO = "ko"
    KO_KR = "ko-KR"

    # Kurdish
    KU = "ku"

    # Kyrgyz
    KY = "ky"
    KY_KG = "ky-KG"

    # Latin
    LA = "la"

    # Luxembourgish
    LB = "lb"

    # Lingala
    LN = "ln"

    # Lao
    LO = "lo"
    LO_LA = "lo-LA"

    # Lithuanian
    LT = "lt"
    LT_LT = "lt-LT"

    # Ganda
    LG = "lg"

    # Luo
    LUO = "luo"

    # Latvian
    LV = "lv"
    LV_LV = "lv-LV"

    # Malagasy
    MG = "mg"

    # Maori
    MI = "mi"

    # Macedonian
    MK = "mk"
    MK_MK = "mk-MK"

    # Malayalam
    ML = "ml"
    ML_IN = "ml-IN"

    # Mongolian
    MN = "mn"
    MN_MN = "mn-MN"

    # Marathi
    MR = "mr"
    MR_IN = "mr-IN"

    # Malay
    MS = "ms"
    MS_MY = "ms-MY"

    # Maltese
    MT = "mt"
    MT_MT = "mt-MT"

    # Burmese
    MY = "my"
    MY_MM = "my-MM"
    MY_MR = "mymr"

    # Norwegian
    NB = "nb"  # Norwegian Bokmål
    NB_NO = "nb-NO"
    NO = "no"
    NN = "nn"  # Norwegian Nynorsk

    # Nepali
    NE = "ne"
    NE_NP = "ne-NP"

    # Dutch
    NL = "nl"
    NL_BE = "nl-BE"
    NL_NL = "nl-NL"

    # Northern Sotho
    NSO = "nso"

    # Chichewa
    NY = "ny"

    # Occitan
    OC = "oc"

    # Odia
    OR = "or"
    OR_IN = "or-IN"

    # Punjabi
    PA = "pa"
    PA_IN = "pa-IN"

    # Polish
    PL = "pl"
    PL_PL = "pl-PL"

    # Pashto
    PS = "ps"
    PS_AF = "ps-AF"

    # Portuguese
    PT = "pt"
    PT_BR = "pt-BR"
    PT_PT = "pt-PT"

    # Romanian
    RO = "ro"
    RO_RO = "ro-RO"

    # Russian
    RU = "ru"
    RU_RU = "ru-RU"

    # Sanskrit
    SA = "sa"

    # Sindhi
    SD = "sd"

    # Sinhala
    SI = "si"
    SI_LK = "si-LK"

    # Slovak
    SK = "sk"
    SK_SK = "sk-SK"

    # Slovenian
    SL = "sl"
    SL_SI = "sl-SI"

    # Shona
    SN = "sn"

    # Somali
    SO = "so"
    SO_SO = "so-SO"

    # Albanian
    SQ = "sq"
    SQ_AL = "sq-AL"

    # Serbian
    SR = "sr"
    SR_RS = "sr-RS"
    SR_LATN = "sr-Latn"
    SR_LATN_RS = "sr-Latn-RS"

    # Sundanese
    SU = "su"
    SU_ID = "su-ID"

    # Swedish
    SV = "sv"
    SV_SE = "sv-SE"

    # Swahili
    SW = "sw"
    SW_KE = "sw-KE"
    SW_TZ = "sw-TZ"

    # Tamil
    TA = "ta"
    TA_IN = "ta-IN"
    TA_LK = "ta-LK"
    TA_MY = "ta-MY"
    TA_SG = "ta-SG"

    # Telugu
    TE = "te"
    TE_IN = "te-IN"

    # Tajik
    TG = "tg"

    # Thai
    TH = "th"
    TH_TH = "th-TH"

    # Turkmen
    TK = "tk"

    # Tagalog
    TL = "tl"

    # Turkish
    TR = "tr"
    TR_TR = "tr-TR"

    # Tatar
    TT = "tt"

    # Uyghur
    UG = "ug"

    # Ukrainian
    UK = "uk"
    UK_UA = "uk-UA"

    # Umbundu
    UMB = "umb"

    # Urdu
    UR = "ur"
    UR_IN = "ur-IN"
    UR_PK = "ur-PK"

    # Uzbek
    UZ = "uz"
    UZ_UZ = "uz-UZ"

    # Vietnamese
    VI = "vi"
    VI_VN = "vi-VN"

    # Wolof
    WO = "wo"

    # Wu Chinese
    WUU = "wuu"
    WUU_CN = "wuu-CN"

    # Yiddish
    YI = "yi"

    # Yoruba
    YO = "yo"

    # Yue Chinese (Cantonese)
    YUE = "yue"
    YUE_CN = "yue-CN"

    # Chinese
    ZH = "zh"
    ZH_CN = "zh-CN"
    ZH_CN_GUANGXI = "zh-CN-guangxi"
    ZH_CN_HENAN = "zh-CN-henan"
    ZH_CN_LIAONING = "zh-CN-liaoning"
    ZH_CN_SHAANXI = "zh-CN-shaanxi"
    ZH_CN_SHANDONG = "zh-CN-shandong"
    ZH_CN_SICHUAN = "zh-CN-sichuan"
    ZH_HK = "zh-HK"
    ZH_TW = "zh-TW"

    # Xhosa
    XH = "xh-ZA"

    # Zulu
    ZU = "zu"
    ZU_ZA = "zu-ZA"
