#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Sarvam AI text-to-speech service implementation.

This module provides TTS services using Sarvam AI's API with support for multiple
Indian languages:

**Model Variants:**

- **bulbul:v3** (default): Sarvam's current TTS model, with temperature control
    - Does NOT support: pitch, loudness
    - Supports: pace (0.5-2.0), temperature (0.01-1.0)
    - Default sample rate: 24000 Hz
    - Preprocessing is always enabled
    - Speakers: shubh (default), aditya, ritu, priya, neha, rahul, pooja, rohan,
      simran, kavya, amit, dev, ishita, shreya, ratan, varun, manan, sumit, roopa,
      kabir, aayan, ashutosh, advait

- **bulbul:v4-flash**: Low-latency model with its own speaker catalogue
    - Supports: pitch (-0.5 to 0.5), loudness (0.1 to 2.5), pace (0.5-2.0)
    - Does NOT support: temperature (the API pins it to 0.6)
    - Default sample rate: 24000 Hz
    - Preprocessing is always enabled
    - Speakers: 224 wire names encoding voice, language, and style, such as
      shubh_en_narration_gentle (default) and ritu_hi_edtech. The bulbul:v2 and
      bulbul:v3 catalogues are not accepted — see :class:`SarvamTTSSpeakerV4Flash`
    - Access is gated: a subscription without it gets HTTP 422

- **bulbul:v2** (deprecated): Sarvam's previous TTS model. Sarvam's API answers a
  request for it with "Model 'bulbul:v2' has been deprecated. Please use
  'bulbul:v3' instead.", so it cannot synthesize.

See https://docs.sarvam.ai/api-reference-docs/text-to-speech/stream for full API details.
"""

import asyncio
import base64
import json
import warnings
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar

import aiohttp
from loguru import logger
from pydantic import BaseModel, Field
from websockets.protocol import State

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.sarvam._sdk import sdk_headers
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import InterruptibleTTSService, TextAggregationMode, TTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.deprecation import deprecated
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given, is_given


class SarvamTTSModel(StrEnum):
    """Available Sarvam TTS models.

    Parameters:
        BULBUL_V2: Previous TTS model, with pitch/loudness control. Sarvam's API
            rejects it, so it cannot synthesize.

            .. deprecated:: 1.9.0
                Use :attr:`SarvamTTSModel.BULBUL_V3` instead.
                Will be removed in 2.0.0.

        BULBUL_V3_BETA: Pre-GA tag of the v3 model.
        BULBUL_V3: Current TTS model, with temperature control.
            - Does NOT support pitch/loudness
            - Pace range: 0.5-2.0
            - Supports temperature parameter
            - Default sample rate: 24000 Hz
            - Preprocessing is always enabled
        BULBUL_V4_FLASH: Low-latency model with its own speaker catalogue.
            - Supports pitch (-0.5 to 0.5), loudness (0.1 to 2.5), pace (0.5-2.0)
            - Does NOT support temperature
            - Default sample rate: 24000 Hz
            - Preprocessing is always enabled
    """

    BULBUL_V2 = "bulbul:v2"
    BULBUL_V3_BETA = "bulbul:v3-beta"
    BULBUL_V3 = "bulbul:v3"
    BULBUL_V4_FLASH = "bulbul:v4-flash"


class SarvamTTSSpeakerV2(StrEnum):
    """Available speakers for the deprecated bulbul:v2 model.

    Female voices: anushka, manisha, vidya, arya
    Male voices: abhilash, karun, hitesh
    """

    ANUSHKA = "anushka"
    ABHILASH = "abhilash"
    MANISHA = "manisha"
    VIDYA = "vidya"
    ARYA = "arya"
    KARUN = "karun"
    HITESH = "hitesh"


class SarvamTTSSpeakerV3(StrEnum):
    """Available speakers for the bulbul:v3 model.

    Includes a wider variety of voices with different characteristics.
    """

    ADITYA = "aditya"
    RITU = "ritu"
    PRIYA = "priya"
    NEHA = "neha"
    RAHUL = "rahul"
    POOJA = "pooja"
    ROHAN = "rohan"
    SIMRAN = "simran"
    KAVYA = "kavya"
    AMIT = "amit"
    DEV = "dev"
    ISHITA = "ishita"
    SHREYA = "shreya"
    RATAN = "ratan"
    VARUN = "varun"
    MANAN = "manan"
    SUMIT = "sumit"
    ROOPA = "roopa"
    KABIR = "kabir"
    AAYAN = "aayan"
    SHUBH = "shubh"
    ASHUTOSH = "ashutosh"
    ADVAIT = "advait"
    AMELIA = "amelia"
    SOPHIA = "sophia"


class SarvamTTSSpeakerV4Flash(StrEnum):
    """Available speakers for the bulbul:v4-flash model.

    Speaker names encode the voice, language, and style (for example
    ``ritu_hi_edtech``). Pick one whose language matches the configured
    ``language``; the bulbul:v2 and bulbul:v3 catalogues are not accepted.
    """

    # Assamese
    KANGKANA_AS_CONVERSATIONAL = "kangkana_as_conversational"
    MOUCHUMI_AS_CONVERSATIONAL = "mouchumi_as_conversational"

    # Bengali
    BAPPA_BN_CONVERSATION = "bappa_bn_conversation"
    ROOPA_BN_CONVERSATIONAL = "roopa_bn_conversational"
    BIMAL_BN_SUSPENSE = "bimal_bn_suspense"

    # English
    ADITI_EN_STORIES = "aditi_en_stories"
    APARNA_EN_COMPANION = "aparna_en_companion"
    APARNA_EN_EDTECH = "aparna_en_edtech"
    ASHWIN_EN_SPORTS = "ashwin_en_sports"
    ASHWIN_EN_SPORTS_ENERGETIC = "ashwin_en_sports_energetic"
    CHANDRIKA_EN_STORIES = "chandrika_en_stories"
    DEV_EN_RECOVERY = "dev_en_recovery"
    DEV_EN_CONVERSATIONAL = "dev_en_conversational"
    DEVEN_EN_CONVERSATION = "deven_en_conversation"
    ISHITA_EN_CUSTOMER = "ishita_en_customer"
    ISHITA_EN_MEDICAL = "ishita_en_medical"
    ISHITA_EN_NUMBERS = "ishita_en_numbers"
    ISHITA_EN_SOCIAL = "ishita_en_social"
    ISHITA_EN_STORIES = "ishita_en_stories"
    KALPIT_EN_EDTECH = "kalpit_en_edtech"
    NACHIKET_EN_ADS = "nachiket_en_ads"
    NEHA_EN_CUSTOMER = "neha_en_customer"
    NEHA_EN_LATENIGHT = "neha_en_latenight"
    NUPUR_EN_KIDS = "nupur_en_kids"
    OJAS_EN_SOCIAL = "ojas_en_social"
    RITU_EN_EDTECH = "ritu_en_edtech"
    RITU_EN_LATENIGHT = "ritu_en_latenight"
    RITU_EN_MEDICAL = "ritu_en_medical"
    RITU_EN_REELS = "ritu_en_reels"
    ROHAN_EN_RECOVERY = "rohan_en_recovery"
    ROOPA_EN_CONVERSATIONAL = "roopa_en_conversational"
    RUSTOM_EN_SUSPENSE = "rustom_en_suspense"
    SANCHITA_EN_COMPANION = "sanchita_en_companion"
    SANCHITA_EN_INSURANCE = "sanchita_en_insurance"
    SANCHITA_EN_RECOVERY = "sanchita_en_recovery"
    SANCHITA_EN_MARKET = "sanchita_en_market"
    SANCHITA_EN_SOCIAL = "sanchita_en_social"
    SHABANA_EN_EDTECH = "shabana_en_edtech"
    SHALINI_EN_COMPANION = "shalini_en_companion"
    SHALINI_EN_CUSTOMER = "shalini_en_customer"
    SHUBH_EN_NARRATION = "shubh_en_narration"
    SHUBH_EN_NUMBERS = "shubh_en_numbers"
    SHUBH_EN_ADS = "shubh_en_ads"
    SHUBH_EN_RECOVERY = "shubh_en_recovery"
    SHUBH_EN_AUDIOBOOK = "shubh_en_audiobook"
    SHUBH_EN_NARRATION_GENTLE = "shubh_en_narration_gentle"
    SHUBH_EN_SPORTS = "shubh_en_sports"
    SIMRAN_EN_NARRATION = "simran_en_narration"
    SIMRAN_EN_AUTOMOBILE = "simran_en_automobile"
    SIMRAN_EN_CONVERSATION = "simran_en_conversation"
    SIMRAN_EN_CUSTOMER = "simran_en_customer"
    SIMRAN_EN_EDTECH = "simran_en_edtech"
    SIMRAN_EN_EDTECH_BOT = "simran_en_edtech_bot"
    SIMRAN_EN_SALES = "simran_en_sales"
    SIMRAN_EN_RECOVERY = "simran_en_recovery"
    SIMRAN_EN_ADS = "simran_en_ads"
    SIMRAN_EN_THERAPIST = "simran_en_therapist"
    SUNNY_EN_SOCIAL = "sunny_en_social"
    VARUN_EN_ADS = "varun_en_ads"
    VARUN_EN_SUSPENSE = "varun_en_suspense"
    ZARINA_EN_CONVERSATION = "zarina_en_conversation"
    AMELIA_EN_CONVERSATIONAL = "amelia_en_conversational"
    SOPHIA_EN_CONVERSATIONAL = "sophia_en_conversational"
    GIRISH_EN_DOCUMENTARY = "girish_en_documentary"
    GIRISH_EN_DEVOTIONAL = "girish_en_devotional"
    PAYAL_EN_EDTECH = "payal_en_edtech"
    SARANG_EN_NARRATION = "sarang_en_narration"

    # English–Hindi mix
    ISHITA_ENHI_COMPANION = "ishita_enhi_companion"
    ISHITA_ENHI_CUSTOMER = "ishita_enhi_customer"
    ISHITA_ENHI_CUSTOMER_EXPRESSIVE = "ishita_enhi_customer_expressive"
    SANCHITA_ENHI_COMPANION = "sanchita_enhi_companion"
    SHALINI_ENHI_COMPANION = "shalini_enhi_companion"
    SHALINI_ENHI_CUSTOMER = "shalini_enhi_customer"
    SHUBH_ENHI_COMPANION = "shubh_enhi_companion"
    SHUBH_ENHI_ADS = "shubh_enhi_ads"
    SHUBH_ENHI_BANKING = "shubh_enhi_banking"
    SIMRAN_ENHI_COMPANION = "simran_enhi_companion"
    SIMRAN_ENHI_CUSTOMER = "simran_enhi_customer"
    SIMRAN_ENHI_BANKING_EXPRESSIVE = "simran_enhi_banking_expressive"
    SUNNY_ENHI_CUSTOMER = "sunny_enhi_customer"

    # Gujarati
    BHAVIK_GU_CONVERSATION = "bhavik_gu_conversation"
    POOJA_GU_CONVERSATIONAL = "pooja_gu_conversational"
    POOJA_GU_CUSTOMER = "pooja_gu_customer"

    # Hindi
    AAYAN_HI_CONVERSATIONAL = "aayan_hi_conversational"
    AMIT_HI_CONVERSATIONAL = "amit_hi_conversational"
    ASHUTOSH_HI_CONVERSATIONAL = "ashutosh_hi_conversational"
    KABIR_HI_CONVERSATIONAL = "kabir_hi_conversational"
    KAVYA_HI_CONVERSATIONAL = "kavya_hi_conversational"
    MANAN_HI_CONVERSATIONAL = "manan_hi_conversational"
    RAHUL_HI_CONVERSATIONAL = "rahul_hi_conversational"
    SUMIT_HI_CONVERSATIONAL = "sumit_hi_conversational"
    ADITYA_HI_CONVERSATIONAL = "aditya_hi_conversational"
    ADITYA_HI_SALES = "aditya_hi_sales"
    ANAND_HI_DOCUMENTARY = "anand_hi_documentary"
    ANAND_HI_NEWS = "anand_hi_news"
    APARNA_HI_CUSTOMER = "aparna_hi_customer"
    APARNA_HI_KYC = "aparna_hi_kyc"
    ASHOK_HI_CHARACTER = "ashok_hi_character"
    ASHOK_HI_NEWS = "ashok_hi_news"
    CHHAVI_HI_KIDS = "chhavi_hi_kids"
    ISHITA_HI_ADS = "ishita_hi_ads"
    ISHITA_HI_EDTECH = "ishita_hi_edtech"
    ISHITA_HI_BANKING = "ishita_hi_banking"
    ISHITA_HI_ADS_INFORMAL = "ishita_hi_ads_informal"
    ISHITA_HI_DEVOTIONAL = "ishita_hi_devotional"
    ISHITA_HI_NUMBERS = "ishita_hi_numbers"
    ISHITA_HI_SOCIAL = "ishita_hi_social"
    KUNAL_HI_KIDS = "kunal_hi_kids"
    MAHESH_HI_DOCUMENTARY = "mahesh_hi_documentary"
    MANI_HI_DEVOTIONAL = "mani_hi_devotional"
    MANI_HI_CONVERSATIONAL = "mani_hi_conversational"
    MOHIT_HI_CONVERSATIONAL = "mohit_hi_conversational"
    NACHIKET_HI_DEVOTIONAL = "nachiket_hi_devotional"
    PRIYA_HI_RECOVERY = "priya_hi_recovery"
    RATAN_HI_LATENIGHT = "ratan_hi_latenight"
    RATAN_HI_CUSTOMER_EXPRESSIVE = "ratan_hi_customer_expressive"
    RATAN_HI_DOCUMENTARY = "ratan_hi_documentary"
    RATAN_HI_DEVOTIONAL = "ratan_hi_devotional"
    RATAN_HI_RECOVERY = "ratan_hi_recovery"
    RATAN_HI_SOCIAL = "ratan_hi_social"
    RATAN_HI_SPORTS = "ratan_hi_sports"
    RATAN_HI_LATENIGHT_WARM = "ratan_hi_latenight_warm"
    REHAN_HI_SOCIAL = "rehan_hi_social"
    RITU_HI_CUSTOMER_UTILITY = "ritu_hi_customer_utility"
    RITU_HI_KIDS = "ritu_hi_kids"
    RITU_HI_CONVERSATION = "ritu_hi_conversation"
    RITU_HI_CUSTOMER = "ritu_hi_customer"
    RITU_HI_EDTECH = "ritu_hi_edtech"
    RITU_HI_ADS_FORMAL = "ritu_hi_ads_formal"
    RITU_HI_BANKING = "ritu_hi_banking"
    RITU_HI_ADS_INFORMAL = "ritu_hi_ads_informal"
    RITU_HI_INSURANCE = "ritu_hi_insurance"
    RITU_HI_EDTECH_BOT = "ritu_hi_edtech_bot"
    RITU_HI_MEDICAL = "ritu_hi_medical"
    RITU_HI_SALES = "ritu_hi_sales"
    RITU_HI_REELS = "ritu_hi_reels"
    RITU_HI_SOCIAL = "ritu_hi_social"
    RITU_HI_CUSTOMER_WARM = "ritu_hi_customer_warm"
    RITU_HI_SOCIAL_LIVELY = "ritu_hi_social_lively"
    ROOPA_HI_COMPANION = "roopa_hi_companion"
    ROOPA_HI_NARRATION = "roopa_hi_narration"
    ROOPA_HI_RECOVERY = "roopa_hi_recovery"
    ROOPA_HI_MARKET = "roopa_hi_market"
    ROOPA_HI_CONVERSATIONAL = "roopa_hi_conversational"
    SANCHITA_HI_ASSISTANT = "sanchita_hi_assistant"
    SANCHITA_HI_EDTECH = "sanchita_hi_edtech"
    SANCHITA_HI_BANKING = "sanchita_hi_banking"
    SANCHITA_HI_FEEDBACK = "sanchita_hi_feedback"
    SANCHITA_HI_ADS_FORMAL = "sanchita_hi_ads_formal"
    SANCHITA_HI_ADS_INFORMAL = "sanchita_hi_ads_informal"
    SANCHITA_HI_INTERVIEW = "sanchita_hi_interview"
    SANCHITA_HI_ROMANTIC = "sanchita_hi_romantic"
    SANCHITA_HI_MARKET = "sanchita_hi_market"
    SANCHITA_HI_SOCIAL = "sanchita_hi_social"
    SANCHITA_HI_KYC = "sanchita_hi_kyc"
    SARIKA_HI_CONVERSATION = "sarika_hi_conversation"
    SHALINI_HI_COMPANION = "shalini_hi_companion"
    SHALINI_HI_SOCIAL = "shalini_hi_social"
    SHREYA_HI_CONVERSATIONAL = "shreya_hi_conversational"
    SHREYA_HI_NEWS = "shreya_hi_news"
    SHRUTI_HI_EDTECH = "shruti_hi_edtech"
    SHUBH_HI_CUSTOMER = "shubh_hi_customer"
    SHUBH_HI_ECOMM = "shubh_hi_ecomm"
    SHUBH_HI_STORIES_MIXED = "shubh_hi_stories_mixed"
    SHUBH_HI_DEVOTIONAL = "shubh_hi_devotional"
    SHUBH_HI_ADS = "shubh_hi_ads"
    SHUBH_HI_RECOVERY = "shubh_hi_recovery"
    SHUBH_HI_STORIES_DRAMATIC = "shubh_hi_stories_dramatic"
    SIMRAN_HI_ASSISTANT = "simran_hi_assistant"
    SIMRAN_HI_NARRATION = "simran_hi_narration"
    SIMRAN_HI_AUTOMOBILE = "simran_hi_automobile"
    SIMRAN_HI_CONVERSATION = "simran_hi_conversation"
    SIMRAN_HI_NEWS_BREAKING = "simran_hi_news_breaking"
    SIMRAN_HI_SOCIAL_ENERGETIC = "simran_hi_social_energetic"
    SIMRAN_HI_SOCIAL_EXCITED = "simran_hi_social_excited"
    SIMRAN_HI_LATENIGHT = "simran_hi_latenight"
    SIMRAN_HI_NEWS = "simran_hi_news"
    SIMRAN_HI_RECOVERY = "simran_hi_recovery"
    SIMRAN_HI_SALES = "simran_hi_sales"
    SUCHITRA_HI_ECOMM = "suchitra_hi_ecomm"
    SUHANI_HI_SOCIAL = "suhani_hi_social"
    SUNNY_HI_ADS = "sunny_hi_ads"
    SUNNY_HI_REELS = "sunny_hi_reels"
    TARUN_HI_CONVERSATIONAL = "tarun_hi_conversational"
    TARUN_HI_SALES = "tarun_hi_sales"
    CHAITRA_HI_CUSTOMER = "chaitra_hi_customer"
    SHILPA_HI_NARRATION = "shilpa_hi_narration"
    TANYA_HI_NARRATION = "tanya_hi_narration"
    AARTI_HI_CUSTOMER = "aarti_hi_customer"
    ADVAIT_HI_CHARACTER = "advait_hi_character"
    ARYAMAN_HI_ADS = "aryaman_hi_ads"
    CHIRAG_HI_SOCIAL = "chirag_hi_social"
    GIRISH_HI_DEVOTIONAL = "girish_hi_devotional"
    MUKUL_HI_ADS = "mukul_hi_ads"
    MUKUL_HI_SUSPENSE = "mukul_hi_suspense"
    SUMAN_HI_COMPANION = "suman_hi_companion"
    VAIBHAV_HI_SOCIAL = "vaibhav_hi_social"
    VANDANA_HI_ECOMM = "vandana_hi_ecomm"
    VIPUL_HI_SOCIAL = "vipul_hi_social"

    # Kannada
    CHAITRA_KN_CONVERSATION = "chaitra_kn_conversation"
    CHAITRA_KN_NARRATION = "chaitra_kn_narration"
    CHETAN_KN_CONVERSATION = "chetan_kn_conversation"
    SUCHITRA_KN_NARRATION = "suchitra_kn_narration"

    # Marathi
    ISHITA_MR_CONVERSATIONAL = "ishita_mr_conversational"
    MRUNAL_MR_NARRATION = "mrunal_mr_narration"
    NEHA_MR_NARRATION = "neha_mr_narration"
    NILESH_MR_CONVERSATION = "nilesh_mr_conversation"
    RITU_MR_INSURANCE = "ritu_mr_insurance"
    RITU_MR_NARRATION = "ritu_mr_narration"
    RUPALI_MR_STORIES = "rupali_mr_stories"
    SOHAM_MR_NARRATION = "soham_mr_narration"
    MUKUL_MR_STORIES = "mukul_mr_stories"

    # Punjabi
    ANAND_PA_CONVERSATION = "anand_pa_conversation"
    ANAND_PA_CUSTOMER = "anand_pa_customer"
    HARPREET_PA_NARRATION = "harpreet_pa_narration"
    JASPAL_PA_BANKING = "jaspal_pa_banking"

    # Tamil
    GOKUL_TA_NARRATION = "gokul_ta_narration"
    VETRI_TA_ADS = "vetri_ta_ads"
    VETRI_TA_SUSPENSE = "vetri_ta_suspense"
    VIJAY_TA_NARRATION = "vijay_ta_narration"

    # Telugu
    KAVITHA_TE_CONVERSATION = "kavitha_te_conversation"
    KAVITHA_TE_NARRATION = "kavitha_te_narration"
    POOJA_TE_CONVERSATION = "pooja_te_conversation"
    TARUN_TE_NARRATION = "tarun_te_narration"


@dataclass(frozen=True)
class TTSModelConfig:
    """Immutable configuration for a Sarvam TTS model.

    Parameters:
        name: The model name this configuration describes.
        supports_pitch: Whether the model accepts pitch parameter.
        supports_loudness: Whether the model accepts loudness parameter.
        supports_temperature: Whether the model accepts temperature parameter.
        default_sample_rate: Default audio sample rate in Hz.
        default_speaker: Default speaker voice ID.
        pace_range: Valid range for pace parameter (min, max).
        pitch_range: Valid range for pitch parameter (min, max).
        loudness_range: Valid range for loudness parameter (min, max).
        preprocessing_always_enabled: Whether preprocessing is always enabled.
        speakers: Tuple of available speaker names for this model.
    """

    name: str
    supports_pitch: bool
    supports_loudness: bool
    supports_temperature: bool
    default_sample_rate: int
    default_speaker: str
    pace_range: tuple[float, float]
    pitch_range: tuple[float, float]
    loudness_range: tuple[float, float]
    preprocessing_always_enabled: bool
    speakers: tuple[str, ...]


#: Sample rates the streaming surfaces accept. The non-streaming
#: ``/text-to-speech`` endpoint additionally accepts 32000, 44100, and 48000.
WEBSOCKET_SAMPLE_RATES: tuple[int, ...] = (8000, 16000, 22050, 24000)

TTS_MODEL_CONFIGS: dict[str, TTSModelConfig] = {
    "bulbul:v2": TTSModelConfig(
        name="bulbul:v2",
        supports_pitch=True,
        supports_loudness=True,
        supports_temperature=False,
        default_sample_rate=22050,
        default_speaker="anushka",
        pace_range=(0.3, 3.0),
        pitch_range=(-0.75, 0.75),
        loudness_range=(0.3, 3.0),
        preprocessing_always_enabled=False,
        speakers=tuple(s.value for s in SarvamTTSSpeakerV2),
    ),
    "bulbul:v3-beta": TTSModelConfig(
        name="bulbul:v3-beta",
        supports_pitch=False,
        supports_loudness=False,
        supports_temperature=True,
        default_sample_rate=24000,
        default_speaker="shubh",
        pace_range=(0.5, 2.0),
        pitch_range=(-0.5, 0.5),
        loudness_range=(0.1, 2.5),
        preprocessing_always_enabled=True,
        speakers=tuple(s.value for s in SarvamTTSSpeakerV3),
    ),
    "bulbul:v3": TTSModelConfig(
        name="bulbul:v3",
        supports_pitch=False,
        supports_loudness=False,
        supports_temperature=True,
        default_sample_rate=24000,
        default_speaker="shubh",
        pace_range=(0.5, 2.0),
        pitch_range=(-0.5, 0.5),
        loudness_range=(0.1, 2.5),
        preprocessing_always_enabled=True,
        speakers=tuple(s.value for s in SarvamTTSSpeakerV3),
    ),
    "bulbul:v4-flash": TTSModelConfig(
        name="bulbul:v4-flash",
        supports_pitch=True,
        supports_loudness=True,
        supports_temperature=False,
        default_sample_rate=24000,
        default_speaker="shubh_en_narration_gentle",
        pace_range=(0.5, 2.0),
        pitch_range=(-0.5, 0.5),
        loudness_range=(0.1, 2.5),
        preprocessing_always_enabled=True,
        speakers=tuple(s.value for s in SarvamTTSSpeakerV4Flash),
    ),
}


def resolve_model_config(model: str | None) -> TTSModelConfig:
    """Look up the configuration for a Sarvam TTS model.

    Args:
        model: The model name (e.g. ``"bulbul:v3"`` or ``"bulbul:v4-flash"``).

    Returns:
        The configuration describing the model's capabilities and defaults.

    Raises:
        ValueError: If the model is not one Sarvam serves.
    """
    if model is None or model not in TTS_MODEL_CONFIGS:
        allowed = ", ".join(sorted(TTS_MODEL_CONFIGS.keys()))
        raise ValueError(f"Unsupported model '{model}'. Allowed values: {allowed}.")
    return TTS_MODEL_CONFIGS[model]


def _clamp_to_range(value: float, value_range: tuple[float, float], name: str, model: str) -> float:
    """Clamp a voice parameter to the range its model accepts."""
    low, high = value_range
    if value < low or value > high:
        logger.warning(f"{name} {value} is outside the {model} range ({low}-{high}). Clamping.")
        return max(low, min(high, value))
    return value


def _format_error(message: str, code: Any = None, request_id: str | None = None) -> str:
    """Build an error string that keeps Sarvam's own message and identifiers intact."""
    context = []
    if code is not None:
        context.append(f"code={code}")
    if request_id:
        context.append(f"request_id={request_id}")
    suffix = f" ({', '.join(context)})" if context else ""
    return f"Sarvam TTS error{suffix}: {message}"


def _format_http_error(status: int, body: str) -> str:
    """Unwrap Sarvam's HTTP error envelope, falling back to the raw body.

    Errors arrive either wrapped in an ``error`` object carrying a code and a
    request id, or as a bare ``message``.
    """
    try:
        payload = json.loads(body)
        error = payload.get("error", payload)
        message = error["message"]
    except (ValueError, AttributeError, KeyError, TypeError):
        return f"Sarvam TTS error (HTTP {status}): {body}"
    return _format_error(
        message, code=error.get("code", f"HTTP {status}"), request_id=error.get("request_id")
    )


def get_speakers_for_model(model: str) -> list[str]:
    """Get the list of available speakers for a given model.

    Args:
        model: The model name (e.g., "bulbul:v3").

    Returns:
        List of speaker names available for the model.
    """
    if model in TTS_MODEL_CONFIGS:
        return list(TTS_MODEL_CONFIGS[model].speakers)
    # Default to v3 speakers for unknown models
    return list(TTS_MODEL_CONFIGS["bulbul:v3"].speakers)


def language_to_sarvam_language(language: Language) -> str:
    """Convert Pipecat Language enum to Sarvam AI language codes.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding service language code. If ``language`` is not in
        the verified mapping, falls back to the full language code string and
        logs a warning (via ``resolve_language(..., use_base_code=False)``).
    """
    LANGUAGE_MAP = {
        Language.AS: "as-IN",  # Assamese
        Language.AS_IN: "as-IN",
        Language.BN: "bn-IN",  # Bengali
        Language.BN_IN: "bn-IN",
        Language.BRX: "brx-IN",  # Bodo
        Language.BRX_IN: "brx-IN",
        Language.DOI: "doi-IN",  # Dogri
        Language.DOI_IN: "doi-IN",
        Language.EN: "en-IN",  # English (India)
        Language.EN_IN: "en-IN",
        Language.GU: "gu-IN",  # Gujarati
        Language.GU_IN: "gu-IN",
        Language.HI: "hi-IN",  # Hindi
        Language.HI_IN: "hi-IN",
        Language.KN: "kn-IN",  # Kannada
        Language.KN_IN: "kn-IN",
        Language.KOK: "kok-IN",  # Konkani
        Language.KOK_IN: "kok-IN",
        Language.KS: "ks-IN",  # Kashmiri
        Language.KS_IN: "ks-IN",
        Language.MAI: "mai-IN",  # Maithili
        Language.MAI_IN: "mai-IN",
        Language.ML: "ml-IN",  # Malayalam
        Language.ML_IN: "ml-IN",
        Language.MNI: "mni-IN",  # Manipuri
        Language.MNI_IN: "mni-IN",
        Language.MR: "mr-IN",  # Marathi
        Language.MR_IN: "mr-IN",
        Language.NE: "ne-IN",  # Nepali
        Language.OR: "od-IN",  # Odia
        Language.OR_IN: "od-IN",
        Language.PA: "pa-IN",  # Punjabi
        Language.PA_IN: "pa-IN",
        Language.SA: "sa-IN",  # Sanskrit
        Language.SAT: "sat-IN",  # Santali
        Language.SAT_IN: "sat-IN",
        Language.SD: "sd-IN",  # Sindhi
        Language.SD_IN: "sd-IN",
        Language.TA: "ta-IN",  # Tamil
        Language.TA_IN: "ta-IN",
        Language.TE: "te-IN",  # Telugu
        Language.TE_IN: "te-IN",
        Language.UR: "ur-IN",  # Urdu
        Language.UR_IN: "ur-IN",
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=False)


@dataclass
class SarvamHttpTTSSettings(TTSSettings):
    """Settings for SarvamHttpTTSService.

    Parameters:
        enable_preprocessing: Whether to enable text preprocessing. Defaults to False.
            **Note:** Always enabled for bulbul:v3 (cannot be disabled).
        pace: Speech pace multiplier, 0.5 to 2.0 on bulbul:v3. Defaults to 1.0.
        pitch: Voice pitch adjustment. Defaults to 0.0.
            - bulbul:v4-flash: Range -0.5 to 0.5
            - bulbul:v2 (deprecated): Range -0.75 to 0.75
            **Note:** Ignored by bulbul:v3.
        loudness: Volume multiplier. Defaults to 1.0.
            - bulbul:v4-flash: Range 0.1 to 2.5
            - bulbul:v2 (deprecated): Range 0.3 to 3.0
            **Note:** Ignored by bulbul:v3.
        temperature: Controls output randomness for bulbul:v3 (0.01 to 1.0).
            Lower values = more deterministic, higher = more random. Defaults to 0.6.
            **Note:** Ignored by bulbul:v4-flash, which the API pins to 0.6.
    """

    enable_preprocessing: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    pace: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    pitch: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    loudness: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    temperature: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


@dataclass
class SarvamTTSSettings(SarvamHttpTTSSettings):
    """Settings for SarvamTTSService.

    Extends :class:`SarvamHttpTTSService.Settings` with WebSocket-specific buffering parameters.

    Parameters:
        min_buffer_size: Minimum characters to buffer before generating audio.
            Lower values reduce latency but may affect quality. Defaults to 50.
        max_chunk_length: Maximum characters processed in a single chunk.
            Controls memory usage and processing efficiency. Defaults to 150.
    """

    _aliases: ClassVar[dict[str, str]] = {"target_language_code": "language"}

    min_buffer_size: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    max_chunk_length: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


def apply_model_constraints(settings: SarvamHttpTTSSettings, config: TTSModelConfig):
    """Reconcile voice settings with what the selected model accepts.

    Parameters outside the model's range are clamped, and parameters the model
    ignores are dropped so they are never sent on the wire.

    Args:
        settings: The settings to adjust in place.
        config: The selected model's configuration.
    """
    model = config.name
    pace = assert_given(settings.pace)
    if pace is not None:
        settings.pace = _clamp_to_range(pace, config.pace_range, "Pace", model)

    if config.preprocessing_always_enabled:
        settings.enable_preprocessing = True

    pitch = assert_given(settings.pitch)
    if not config.supports_pitch:
        if pitch not in (None, 0.0):
            logger.warning(f"pitch parameter is ignored for {model}")
        settings.pitch = None
    elif pitch is not None:
        settings.pitch = _clamp_to_range(pitch, config.pitch_range, "Pitch", model)

    loudness = assert_given(settings.loudness)
    if not config.supports_loudness:
        if loudness not in (None, 1.0):
            logger.warning(f"loudness parameter is ignored for {model}")
        settings.loudness = None
    elif loudness is not None:
        settings.loudness = _clamp_to_range(loudness, config.loudness_range, "Loudness", model)

    if not config.supports_temperature:
        if assert_given(settings.temperature) not in (None, 0.6):
            logger.warning(f"temperature parameter is ignored for {model}")
        settings.temperature = None


class SarvamHttpTTSService(TTSService):
    """Text-to-Speech service using Sarvam AI's API.

    Converts text to speech using Sarvam AI's TTS models with support for multiple
    Indian languages. Provides control over voice characteristics.

    **Models:**

    - **bulbul:v3** (default):
        - Does NOT support: pitch, loudness (will be ignored)
        - Supports: pace (0.5 to 2.0), temperature (0.01 to 1.0)
        - Default sample rate: 24000 Hz
        - Preprocessing is always enabled
        - Speakers: shubh, aditya, ritu, priya, neha, rahul, pooja, rohan, simran,
          kavya, amit, dev, ishita, shreya, ratan, varun, manan, sumit, roopa,
          kabir, aayan, ashutosh, advait

    - **bulbul:v4-flash**:
        - Supports: pitch (-0.5 to 0.5), loudness (0.1 to 2.5), pace (0.5 to 2.0)
        - Does NOT support: temperature (will be ignored)
        - Default sample rate: 24000 Hz
        - Preprocessing is always enabled
        - Speakers: its own catalogue of 224 wire names such as
          shubh_en_narration_gentle (default) and ritu_hi_edtech; the v3 names
          are rejected. See :class:`SarvamTTSSpeakerV4Flash`.

    The previous model, bulbul:v2, is deprecated: Sarvam's API rejects it.

    Example::

        # Using bulbul:v3 (default) with temperature control
        tts = SarvamHttpTTSService(
            api_key="your-api-key",
            aiohttp_session=session,
            settings=SarvamHttpTTSService.Settings(
                voice="shubh",
                model="bulbul:v3",
                language=Language.HI,
                pace=1.2,  # Range: 0.5-2.0 for v3
                temperature=0.8,
            ),
        )

        # Using bulbul:v4-flash
        tts_v4 = SarvamHttpTTSService(
            api_key="your-api-key",
            aiohttp_session=session,
            settings=SarvamHttpTTSService.Settings(
                voice="ritu_hi_edtech",  # Use a v4-flash speaker
                model="bulbul:v4-flash",
                language=Language.HI,
                pitch=0.2,  # Range: -0.5 to 0.5 for v4-flash
                loudness=1.4,  # Range: 0.1 to 2.5 for v4-flash
            ),
        )
    """

    Settings = SarvamHttpTTSSettings
    _settings: Settings

    @deprecated(
        "`SarvamHttpTTSService.InputParams` is deprecated since 0.0.105 and will be removed in "
        "2.0.0. Use `SarvamHttpTTSService.Settings` instead."
    )
    class InputParams(BaseModel):
        """Input parameters for Sarvam TTS configuration.

        .. deprecated:: 0.0.105
            Use ``SarvamHttpTTSService.Settings`` directly via the ``settings`` parameter instead.
            Will be removed in 2.0.0.

        Parameters:
            language: Language for synthesis. Defaults to English (India).
            pitch: Voice pitch adjustment (-0.75 to 0.75). Defaults to 0.0.
                **Note:** Only supported for bulbul:v2. Ignored for v3 models.
            pace: Speech pace multiplier. Defaults to 1.0.
                - bulbul:v3: Range 0.5 to 2.0
                - bulbul:v2: Range 0.3 to 3.0
            loudness: Volume multiplier (0.3 to 3.0). Defaults to 1.0.
                **Note:** Only supported for bulbul:v2. Ignored for v3 models.
            enable_preprocessing: Whether to enable text preprocessing. Defaults to False.
                **Note:** Always enabled for bulbul:v3 (cannot be disabled).
            temperature: Controls output randomness for bulbul:v3 (0.01 to 1.0).
                Lower values = more deterministic, higher = more random. Defaults to 0.6.
                **Note:** Only supported for bulbul:v3. Ignored for v2.
        """

        language: Language | None = Language.EN
        pitch: float | None = Field(
            default=0.0,
            ge=-0.75,
            le=0.75,
            description="Voice pitch adjustment. Only for bulbul:v2.",
        )
        pace: float | None = Field(
            default=1.0,
            ge=0.3,
            le=3.0,
            description="Speech pace. v2: 0.3-3.0, v3: 0.5-2.0.",
        )
        loudness: float | None = Field(
            default=1.0,
            ge=0.3,
            le=3.0,
            description="Volume multiplier. Only for bulbul:v2.",
        )
        enable_preprocessing: bool | None = Field(
            default=False,
            description="Enable text preprocessing. Always enabled for the v3 model.",
        )
        temperature: float | None = Field(
            default=0.6,
            ge=0.01,
            le=1.0,
            description="Output randomness for bulbul:v3 only. Range: 0.01-1.0.",
        )

    def __init__(
        self,
        *,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        voice_id: str | None = None,
        model: str | None = None,
        base_url: str = "https://api.sarvam.ai",
        sample_rate: int | None = None,
        params: InputParams | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Sarvam TTS service.

        Args:
            api_key: Sarvam AI API subscription key.
            aiohttp_session: Shared aiohttp session for making requests.
            voice_id: Speaker voice ID. If None, uses model-appropriate default.

                .. deprecated:: 0.0.105
                    Use ``settings=SarvamHttpTTSService.Settings(voice=...)`` instead.
                    Will be removed in 2.0.0.

            model: TTS model to use. Defaults to "bulbul:v3", Sarvam's current
                model; "bulbul:v4-flash" is also available.

                .. deprecated:: 0.0.105
                    Use ``settings=SarvamHttpTTSService.Settings(model=...)`` instead.
                    Will be removed in 2.0.0.

            base_url: Sarvam AI API base URL. Defaults to "https://api.sarvam.ai".
            sample_rate: Audio sample rate in Hz (8000, 16000, 22050, 24000, and,
                for bulbul:v3, 32000, 44100, 48000). If None, uses model-specific
                default.
            params: Additional voice and preprocessing parameters. If None, uses defaults.

                .. deprecated:: 0.0.105
                    Use ``settings=SarvamHttpTTSService.Settings(...)`` instead.
                    Will be removed in 2.0.0.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional arguments passed to parent TTSService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="bulbul:v3",
            voice="shubh",
            language="en-IN",
            enable_preprocessing=False,
            pace=1.0,
            pitch=None,
            loudness=None,
            temperature=None,
        )

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model
        if voice_id is not None:
            self._warn_init_param_moved_to_settings("voice_id", "voice")
            default_settings.voice = voice_id

        # 3. Apply params overrides — only if settings not provided
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings:
                if params.language is not None:
                    default_settings.language = params.language
                if params.enable_preprocessing is not None:
                    default_settings.enable_preprocessing = params.enable_preprocessing
                if params.pace is not None:
                    default_settings.pace = params.pace
                if params.pitch is not None:
                    default_settings.pitch = params.pitch
                if params.loudness is not None:
                    default_settings.loudness = params.loudness
                if params.temperature is not None:
                    default_settings.temperature = params.temperature

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        resolved_model = assert_given(default_settings.model)
        self._config = resolve_model_config(resolved_model)

        if resolved_model == SarvamTTSModel.BULBUL_V2:
            warnings.warn(
                "`SarvamTTSModel.BULBUL_V2` is deprecated since 1.9.0 and will be removed in "
                "2.0.0. Use `SarvamTTSModel.BULBUL_V3` instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Set default sample rate based on model if not specified
        if sample_rate is None:
            sample_rate = self._config.default_sample_rate

        # Set default voice based on model if not specified via any mechanism
        if voice_id is None and (settings is None or not is_given(settings.voice)):
            default_settings.voice = self._config.default_speaker

        apply_model_constraints(default_settings, self._config)

        super().__init__(
            sample_rate=sample_rate,
            push_stop_frames=True,
            push_start_frame=True,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url
        self._session = aiohttp_session

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Sarvam service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to Sarvam AI language format.

        Args:
            language: The language to convert.

        Returns:
            The Sarvam AI-specific language code, or None if not supported.
        """
        return language_to_sarvam_language(language)

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using Sarvam AI's API.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        try:
            # Build payload with common parameters
            payload = {
                "text": text,
                "target_language_code": self._settings.language,
                "speaker": self._settings.voice,
                "speech_sample_rate": self.sample_rate,
                "enable_preprocessing": self._settings.enable_preprocessing,
                "model": self._settings.model,
                "pace": self._settings.pace if self._settings.pace is not None else 1.0,
            }

            # Add model-specific parameters based on config
            if self._config.supports_pitch:
                payload["pitch"] = self._settings.pitch if self._settings.pitch is not None else 0.0
            if self._config.supports_loudness:
                payload["loudness"] = (
                    self._settings.loudness if self._settings.loudness is not None else 1.0
                )
            if self._config.supports_temperature:
                payload["temperature"] = (
                    self._settings.temperature if self._settings.temperature is not None else 0.6
                )

            headers = {
                "api-subscription-key": self._api_key,
                "Content-Type": "application/json",
                **sdk_headers(),
            }

            url = f"{self._base_url}/text-to-speech"

            async with self._session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    yield ErrorFrame(error=_format_http_error(response.status, error_text))
                    return

                response_data = await response.json()

            await self.start_tts_usage_metrics(text)

            # Decode base64 audio data
            if "audios" not in response_data or not response_data["audios"]:
                yield ErrorFrame(
                    error=_format_error(
                        "No audio data received", request_id=response_data.get("request_id")
                    )
                )
                return

            # Get the first audio (there should be only one for single text input)
            base64_audio = response_data["audios"][0]
            audio_data = base64.b64decode(base64_audio)

            # Strip WAV header (first 44 bytes) if present
            if len(audio_data) > 44 and audio_data.startswith(b"RIFF"):
                logger.debug("Stripping WAV header from Sarvam audio data")
                audio_data = audio_data[44:]

            frame = TTSAudioRawFrame(
                audio=audio_data,
                sample_rate=self.sample_rate,
                num_channels=1,
                context_id=context_id,
            )

            yield frame

        except Exception as e:
            yield ErrorFrame(error=f"Error generating TTS: {e}", exception=e)
        finally:
            await self.stop_ttfb_metrics()


class SarvamTTSService(InterruptibleTTSService):
    """WebSocket-based text-to-speech service using Sarvam AI.

    Provides streaming TTS with real-time audio generation for multiple Indian languages.
    Uses WebSocket for low-latency streaming audio synthesis.

    **Models:**

    - **bulbul:v3** (default):
        - Does NOT support: pitch, loudness (will be ignored)
        - Supports: pace (0.5 to 2.0), temperature (0.01 to 1.0)
        - Default sample rate: 24000 Hz
        - Preprocessing is always enabled
        - Speakers: shubh, aditya, ritu, priya, neha, rahul, pooja, rohan, simran,
          kavya, amit, dev, ishita, shreya, ratan, varun, manan, sumit, roopa,
          kabir, aayan, ashutosh, advait

    - **bulbul:v4-flash**:
        - Supports: pitch (-0.5 to 0.5), loudness (0.1 to 2.5), pace (0.5 to 2.0)
        - Does NOT support: temperature (will be ignored)
        - Default sample rate: 24000 Hz
        - Preprocessing is always enabled
        - Speakers: its own catalogue of 224 wire names such as
          shubh_en_narration_gentle (default) and ritu_hi_edtech; the v3 names
          are rejected. See :class:`SarvamTTSSpeakerV4Flash`.

    The previous model, bulbul:v2, is deprecated: Sarvam's API rejects it.

    Sample rates are limited to 8000, 16000, 22050, and 24000 Hz on the WebSocket.

    **WebSocket Protocol:**
    The service uses a WebSocket connection for real-time streaming. Messages include:
    - config: Initial configuration with voice settings
    - text: Text chunks for synthesis
    - flush: Signal to process remaining buffered text
    - ping: Keepalive signal

    Example::

        # Using bulbul:v3 (default) with temperature control
        tts = SarvamTTSService(
            api_key="your-api-key",
            settings=SarvamTTSService.Settings(
                voice="shubh",
                model="bulbul:v3",
                language=Language.HI,
                pace=1.2,  # Range: 0.5-2.0 for v3
                temperature=0.8,
            ),
        )

        # Using bulbul:v4-flash
        tts_v4 = SarvamTTSService(
            api_key="your-api-key",
            settings=SarvamTTSService.Settings(
                voice="ritu_hi_edtech",  # Use a v4-flash speaker
                model="bulbul:v4-flash",
                language=Language.HI,
                pitch=0.2,  # Range: -0.5 to 0.5 for v4-flash
                loudness=1.4,  # Range: 0.1 to 2.5 for v4-flash
            ),
        )

    See https://docs.sarvam.ai/api-reference-docs/text-to-speech/stream for API details.
    """

    Settings = SarvamTTSSettings
    _settings: Settings

    @deprecated(
        "`SarvamTTSService.InputParams` is deprecated since 0.0.105 and will be removed in 2.0.0. "
        "Use `SarvamTTSService.Settings` instead."
    )
    class InputParams(BaseModel):
        """Configuration parameters for Sarvam TTS WebSocket service.

        .. deprecated:: 0.0.105
            Use ``SarvamTTSService.Settings`` directly via the ``settings`` parameter instead.
            Will be removed in 2.0.0.

        Parameters:
            pitch: Voice pitch adjustment (-0.75 to 0.75). Defaults to 0.0.
                **Note:** Only supported for bulbul:v2. Ignored for v3 models.
            pace: Speech pace multiplier. Defaults to 1.0.
                - bulbul:v3: Range 0.5 to 2.0
                - bulbul:v2: Range 0.3 to 3.0
            loudness: Volume multiplier (0.3 to 3.0). Defaults to 1.0.
                **Note:** Only supported for bulbul:v2. Ignored for v3 models.
            enable_preprocessing: Enable text preprocessing. Defaults to False.
                **Note:** Always enabled for bulbul:v3.
            min_buffer_size: Minimum characters to buffer before generating audio.
                Lower values reduce latency but may affect quality. Defaults to 50.
            max_chunk_length: Maximum characters processed in a single chunk.
                Controls memory usage and processing efficiency. Defaults to 150.
            output_audio_codec: Audio codec format. Options: linear16, mulaw, alaw,
                opus, flac, aac, wav, mp3. Defaults to "linear16".
            output_audio_bitrate: Audio bitrate (32k, 64k, 96k, 128k, 192k).
                Defaults to "128k".
            language: Target language for synthesis. Supports Indian languages.
            temperature: Controls output randomness for bulbul:v3 (0.01 to 1.0).
                Lower = more deterministic, higher = more random. Defaults to 0.6.
                **Note:** Only supported for bulbul:v3. Ignored for v2.

        **Speakers by Model:**

        bulbul:v3:
            - shubh (default), aditya, ritu, priya, neha, rahul, pooja, rohan,
              simran, kavya, amit, dev, ishita, shreya, ratan, varun, manan,
              sumit, roopa, kabir, aayan, ashutosh, advait

        bulbul:v2:
            - Female: anushka (default), manisha, vidya, arya
            - Male: abhilash, karun, hitesh
        """

        pitch: float | None = Field(
            default=0.0,
            ge=-0.75,
            le=0.75,
            description="Voice pitch adjustment. Only for bulbul:v2.",
        )
        pace: float | None = Field(
            default=1.0,
            ge=0.3,
            le=3.0,
            description="Speech pace. v2: 0.3-3.0, v3: 0.5-2.0.",
        )
        loudness: float | None = Field(
            default=1.0,
            ge=0.3,
            le=3.0,
            description="Volume multiplier. Only for bulbul:v2.",
        )
        enable_preprocessing: bool | None = Field(
            default=False,
            description="Enable text preprocessing. Always enabled for v3 models.",
        )
        min_buffer_size: int | None = Field(
            default=50,
            description="Minimum characters to buffer before TTS processing.",
        )
        max_chunk_length: int | None = Field(
            default=150,
            description="Maximum length for sentence splitting.",
        )
        output_audio_codec: str | None = Field(
            default="linear16",
            description="Audio codec: linear16, mulaw, alaw, opus, flac, aac, wav, mp3.",
        )
        output_audio_bitrate: str | None = Field(
            default="128k",
            description="Audio bitrate: 32k, 64k, 96k, 128k, 192k.",
        )
        language: Language | None = Language.EN
        temperature: float | None = Field(
            default=0.6,
            ge=0.01,
            le=1.0,
            description="Output randomness for bulbul:v3 only. Range: 0.01-1.0.",
        )

    def __init__(
        self,
        *,
        api_key: str,
        model: str | None = None,
        voice_id: str | None = None,
        url: str = "wss://api.sarvam.ai/text-to-speech/ws",
        aggregate_sentences: bool | None = None,
        text_aggregation_mode: TextAggregationMode | None = None,
        sample_rate: int | None = None,
        params: InputParams | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Sarvam TTS service with voice and transport configuration.

        Args:
            api_key: Sarvam API key for authenticating TTS requests.
            model: TTS model to use. Defaults to "bulbul:v3", Sarvam's current
                model; "bulbul:v4-flash" is also available.

                .. deprecated:: 0.0.105
                    Use ``settings=SarvamTTSService.Settings(model=...)`` instead.
                    Will be removed in 2.0.0.

            voice_id: Speaker voice ID. If None, uses model-appropriate default.

                .. deprecated:: 0.0.105
                    Use ``settings=SarvamTTSService.Settings(voice=...)`` instead.
                    Will be removed in 2.0.0.

            url: WebSocket URL for the TTS backend (default production URL).
            aggregate_sentences: Deprecated. Use text_aggregation_mode instead.

                .. deprecated:: 0.0.104
                    Use ``text_aggregation_mode`` instead.
                    Will be removed in 2.0.0.

            text_aggregation_mode: How to aggregate text before synthesis.
            sample_rate: Output audio sample rate in Hz (8000, 16000, 22050, 24000).
                If None, uses model-specific default.
            params: Optional input parameters to override defaults.

                .. deprecated:: 0.0.105
                    Use ``settings=SarvamTTSService.Settings(...)`` instead.
                    Will be removed in 2.0.0.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Arguments forwarded to InterruptibleTTSService.

        See https://docs.sarvam.ai/api-reference-docs/text-to-speech/stream
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="bulbul:v3",
            voice="shubh",
            language="en-IN",
            enable_preprocessing=False,
            min_buffer_size=50,
            max_chunk_length=150,
            pace=1.0,
            pitch=None,
            loudness=None,
            temperature=None,
        )

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model
        if voice_id is not None:
            self._warn_init_param_moved_to_settings("voice_id", "voice")
            default_settings.voice = voice_id

        # Init-only audio format fields (not runtime-updatable)
        output_audio_codec = "linear16"
        output_audio_bitrate = "128k"

        # 3. Apply params overrides — only if settings not provided
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings:
                if params.language is not None:
                    default_settings.language = params.language
                if params.enable_preprocessing is not None:
                    default_settings.enable_preprocessing = params.enable_preprocessing
                if params.min_buffer_size is not None:
                    default_settings.min_buffer_size = params.min_buffer_size
                if params.max_chunk_length is not None:
                    default_settings.max_chunk_length = params.max_chunk_length
                if params.output_audio_codec is not None:
                    output_audio_codec = params.output_audio_codec
                if params.output_audio_bitrate is not None:
                    output_audio_bitrate = params.output_audio_bitrate
                if params.pace is not None:
                    default_settings.pace = params.pace
                if params.pitch is not None:
                    default_settings.pitch = params.pitch
                if params.loudness is not None:
                    default_settings.loudness = params.loudness
                if params.temperature is not None:
                    default_settings.temperature = params.temperature

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        resolved_model = assert_given(default_settings.model)
        self._config = resolve_model_config(resolved_model)

        if resolved_model == SarvamTTSModel.BULBUL_V2:
            warnings.warn(
                "`SarvamTTSModel.BULBUL_V2` is deprecated since 1.9.0 and will be removed in "
                "2.0.0. Use `SarvamTTSModel.BULBUL_V3` instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Set default sample rate based on model if not specified
        if sample_rate is None:
            sample_rate = self._config.default_sample_rate

        # Set default voice based on model if not specified via any mechanism
        if voice_id is None and (settings is None or not is_given(settings.voice)):
            default_settings.voice = self._config.default_speaker

        if sample_rate not in WEBSOCKET_SAMPLE_RATES:
            allowed = ", ".join(str(rate) for rate in WEBSOCKET_SAMPLE_RATES)
            raise ValueError(
                f"Unsupported sample rate {sample_rate}. The Sarvam TTS WebSocket "
                f"accepts: {allowed}."
            )

        apply_model_constraints(default_settings, self._config)

        # Initialize parent class
        super().__init__(
            aggregate_sentences=aggregate_sentences,
            text_aggregation_mode=text_aggregation_mode,
            push_text_frames=True,
            pause_frame_processing=True,
            push_stop_frames=True,
            push_start_frame=True,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        # Init-only audio format fields (not runtime-updatable)
        self._speech_sample_rate = str(sample_rate)
        self._output_audio_codec = output_audio_codec
        self._output_audio_bitrate = output_audio_bitrate

        # WebSocket endpoint URL with model query parameter. We explicitly request
        # the completion event so we can emit TTSStoppedFrame as soon as synthesis
        # finishes, rather than waiting for the stop_frame_timeout_s idle timer.
        self._websocket_url = f"{url}?model={resolved_model}&send_completion_event=true"
        self._api_key = api_key

        self._receive_task = None
        self._keepalive_task = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Sarvam service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to Sarvam AI language format.

        Args:
            language: The language to convert.

        Returns:
            The Sarvam AI-specific language code, or None if not supported.
        """
        return language_to_sarvam_language(language)

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        # WebSocket API expects sample rate as string
        self._speech_sample_rate = str(self.sample_rate)
        await self._connect()

    async def flush_audio(self, context_id: str | None = None):
        """Flush any pending audio synthesis by sending flush command."""
        try:
            if self._websocket:
                msg = {"type": "flush"}
                await self._websocket.send(json.dumps(msg))
        except Exception as e:
            await self.push_error(error_msg=f"Error sending flush to Sarvam: {e}", exception=e)

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta and resend config if voice changed."""
        changed = await super()._update_settings(delta)

        if changed:
            await self._send_config()

        return changed

    async def _connect(self):
        """Connect to Sarvam WebSocket and start background tasks."""
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

        if self._websocket and not self._keepalive_task:
            self._keepalive_task = self.create_task(
                self._keepalive_task_handler(),
            )

    async def _disconnect(self):
        """Disconnect from Sarvam WebSocket and clean up tasks."""
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        if self._keepalive_task:
            await self.cancel_task(self._keepalive_task)
            self._keepalive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Establish WebSocket connection to Sarvam API."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            ws_additional_headers = {
                "api-subscription-key": self._api_key,
            }

            self._websocket = await self._websocket_connect(
                self._websocket_url,
                additional_headers=ws_additional_headers,
                user_agent_header=sdk_headers()["User-Agent"],
            )
            logger.debug("Connected to Sarvam TTS Websocket")
            await self._send_config()

            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(
                error_msg=f"Error connecting to Sarvam TTS Websocket: {e}", exception=e
            )
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _send_config(self):
        """Send initial configuration message."""
        if not self._websocket:
            raise Exception("WebSocket not connected")
        # Build config dict for the API
        config_data = {
            "target_language_code": self._settings.language,
            "speaker": self._settings.voice,
            "speech_sample_rate": self._speech_sample_rate,
            "enable_preprocessing": self._settings.enable_preprocessing,
            "min_buffer_size": self._settings.min_buffer_size,
            "max_chunk_length": self._settings.max_chunk_length,
            "output_audio_codec": self._output_audio_codec,
            "output_audio_bitrate": self._output_audio_bitrate,
            "pace": self._settings.pace,
            "model": self._settings.model,
        }
        if self._settings.pitch is not None:
            config_data["pitch"] = self._settings.pitch
        if self._settings.loudness is not None:
            config_data["loudness"] = self._settings.loudness
        if self._settings.temperature is not None:
            config_data["temperature"] = self._settings.temperature
        logger.debug(f"Config being sent is {config_data}")
        config_message = {"type": "config", "data": config_data}

        try:
            await self._websocket.send(json.dumps(config_message))
            logger.debug("Configuration sent successfully")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            raise

    async def _disconnect_websocket(self):
        """Close WebSocket connection and clean up state."""
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from Sarvam")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing websocket: {e}", exception=e)
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _receive_messages(self):
        """Receive and process messages from Sarvam WebSocket."""
        async for message in self._get_websocket():
            if isinstance(message, str):
                msg = json.loads(message)
                context_id = self.get_active_audio_context_id()
                if msg.get("type") == "audio":
                    request_id = msg.get("data", {}).get("request_id", "N/A")
                    logger.trace(f"TTS request_id={request_id}, context_id={context_id}")

                    # Check for interruption before processing audio
                    await self.stop_ttfb_metrics()
                    audio = base64.b64decode(msg["data"]["audio"])
                    frame = TTSAudioRawFrame(audio, self.sample_rate, 1, context_id=context_id)
                    await self.append_to_audio_context(context_id, frame)
                elif (
                    msg.get("type") == "event" and msg.get("data", {}).get("event_type") == "final"
                ):
                    # Synthesis for the active context is complete. Emit the
                    # TTSStoppedFrame immediately so BotStoppedSpeakingFrame tracks
                    # the end of audio, instead of waiting on stop_frame_timeout_s.
                    logger.trace(f"TTS final event for context_id={context_id}")
                    if context_id and self.audio_context_available(context_id):
                        await self.append_to_audio_context(
                            context_id, TTSStoppedFrame(context_id=context_id)
                        )
                        await self.remove_audio_context(context_id)
                elif msg.get("type") == "error":
                    data = msg.get("data", {})
                    error_msg = data.get("message", "unknown error")
                    if data.get("details"):
                        error_msg = f"{error_msg} details={data['details']}"
                    error = _format_error(
                        error_msg, code=data.get("code"), request_id=data.get("request_id")
                    )
                    await self.push_error(error_msg=error)

                    # If it's a timeout error, the connection might need to be reset
                    if "too long" in error_msg.lower() or "timeout" in error_msg.lower():
                        logger.warning("Connection timeout detected, service may need restart")
                    await self.append_to_audio_context(context_id, ErrorFrame(error=error))

    async def _keepalive_task_handler(self):
        """Handle keepalive messages to maintain WebSocket connection."""
        KEEPALIVE_SLEEP = 20
        while True:
            await asyncio.sleep(KEEPALIVE_SLEEP)
            await self._send_keepalive()

    async def _send_keepalive(self):
        """Send keepalive message to maintain connection."""
        if self._websocket and self._websocket.state == State.OPEN:
            msg = {"type": "ping"}
            await self._websocket.send(json.dumps(msg))

    async def _send_text(self, text: str):
        """Send text to Sarvam WebSocket for synthesis."""
        if self._websocket and self._websocket.state == State.OPEN:
            msg = {"type": "text", "data": {"text": text}}
            await self._websocket.send(json.dumps(msg))
        else:
            logger.warning("WebSocket not ready, cannot send text")

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech audio frames from input text using Sarvam TTS.

        Sends text over WebSocket for synthesis and yields corresponding audio or status frames.

        Args:
            text: The text input to synthesize.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame objects including TTSStartedFrame, TTSAudioRawFrame(s, context_id=context_id), or TTSStoppedFrame.
        """
        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            try:
                await self._send_text(text)
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                yield ErrorFrame(error=f"Unknown error occurred: {e}")
                yield TTSStoppedFrame(context_id=context_id)
                await self._disconnect()
                await self._connect()
                return
            yield None
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
