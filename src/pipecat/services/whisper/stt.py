#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""This module implements Whisper transcription with a locally-downloaded model."""

import asyncio
from enum import Enum
from typing import AsyncGenerator, Optional

import numpy as np
from loguru import logger
from typing_extensions import TYPE_CHECKING, override

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

if TYPE_CHECKING:
    try:
        from faster_whisper import WhisperModel
    except ModuleNotFoundError as e:
        logger.error(f"Exception: {e}")
        logger.error("In order to use Whisper, you need to `pip install pipecat-ai[whisper]`.")
        raise Exception(f"Missing module: {e}")

    try:
        import mlx_whisper
    except ModuleNotFoundError as e:
        logger.error(f"Exception: {e}")
        logger.error("In order to use Whisper, you need to `pip install pipecat-ai[mlx-whisper]`.")
        raise Exception(f"Missing module: {e}")


class Model(Enum):
    """Class of basic Whisper model selection options.

    Available models:
        Multilingual models:
            TINY: Smallest multilingual model
            BASE: Basic multilingual model
            MEDIUM: Good balance for multilingual
            LARGE: Best quality multilingual
            DISTIL_LARGE_V2: Fast multilingual

        English-only models:
            DISTIL_MEDIUM_EN: Fast English-only
    """

    # Multilingual models
    TINY = "tiny"
    BASE = "base"
    MEDIUM = "medium"
    LARGE = "large-v3"
    DISTIL_LARGE_V2 = "Systran/faster-distil-whisper-large-v2"

    # English-only models
    DISTIL_MEDIUM_EN = "Systran/faster-distil-whisper-medium.en"


class MLXModel(Enum):
    """Class of MLX Whisper model selection options.

    Available models:
        Multilingual models:
            TINY: Smallest multilingual model
            MEDIUM: Good balance for multilingual
            LARGE_V3: Best quality multilingual
            LARGE_V3_TURBO: Finetuned, pruned Whisper large-v3, much faster, slightly lower quality
            DISTIL_LARGE_V3: Fast multilingual
            LARGE_V3_TURBO_Q4: LARGE_V3_TURBO, quantized to Q4
    """

    # Multilingual models
    TINY = "mlx-community/whisper-tiny"
    MEDIUM = "mlx-community/whisper-medium-mlx"
    LARGE_V3 = "mlx-community/whisper-large-v3-mlx"
    LARGE_V3_TURBO = "mlx-community/whisper-large-v3-turbo"
    DISTIL_LARGE_V3 = "mlx-community/distil-whisper-large-v3"
    LARGE_V3_TURBO_Q4 = "mlx-community/whisper-large-v3-turbo-q4"


def language_to_whisper_language(language: Language) -> Optional[str]:
    """Maps pipecat Language enum to Whisper language codes.

    Args:
        language: A Language enum value representing the input language.

    Returns:
        str or None: The corresponding Whisper language code, or None if not supported.

    Note:
        Only includes languages officially supported by Whisper.
    """
    language_map = {
        # Arabic
        Language.AR: "ar",
        Language.AR_AE: "ar",
        Language.AR_BH: "ar",
        Language.AR_DZ: "ar",
        Language.AR_EG: "ar",
        Language.AR_IQ: "ar",
        Language.AR_JO: "ar",
        Language.AR_KW: "ar",
        Language.AR_LB: "ar",
        Language.AR_LY: "ar",
        Language.AR_MA: "ar",
        Language.AR_OM: "ar",
        Language.AR_QA: "ar",
        Language.AR_SA: "ar",
        Language.AR_SY: "ar",
        Language.AR_TN: "ar",
        Language.AR_YE: "ar",
        # Bengali
        Language.BN: "bn",
        Language.BN_BD: "bn",
        Language.BN_IN: "bn",
        # Czech
        Language.CS: "cs",
        Language.CS_CZ: "cs",
        # Danish
        Language.DA: "da",
        Language.DA_DK: "da",
        # German
        Language.DE: "de",
        Language.DE_AT: "de",
        Language.DE_CH: "de",
        Language.DE_DE: "de",
        # Greek
        Language.EL: "el",
        Language.EL_GR: "el",
        # English
        Language.EN: "en",
        Language.EN_AU: "en",
        Language.EN_CA: "en",
        Language.EN_GB: "en",
        Language.EN_HK: "en",
        Language.EN_IE: "en",
        Language.EN_IN: "en",
        Language.EN_KE: "en",
        Language.EN_NG: "en",
        Language.EN_NZ: "en",
        Language.EN_PH: "en",
        Language.EN_SG: "en",
        Language.EN_TZ: "en",
        Language.EN_US: "en",
        Language.EN_ZA: "en",
        # Spanish
        Language.ES: "es",
        Language.ES_AR: "es",
        Language.ES_BO: "es",
        Language.ES_CL: "es",
        Language.ES_CO: "es",
        Language.ES_CR: "es",
        Language.ES_CU: "es",
        Language.ES_DO: "es",
        Language.ES_EC: "es",
        Language.ES_ES: "es",
        Language.ES_GQ: "es",
        Language.ES_GT: "es",
        Language.ES_HN: "es",
        Language.ES_MX: "es",
        Language.ES_NI: "es",
        Language.ES_PA: "es",
        Language.ES_PE: "es",
        Language.ES_PR: "es",
        Language.ES_PY: "es",
        Language.ES_SV: "es",
        Language.ES_US: "es",
        Language.ES_UY: "es",
        Language.ES_VE: "es",
        # Persian
        Language.FA: "fa",
        Language.FA_IR: "fa",
        # Finnish
        Language.FI: "fi",
        Language.FI_FI: "fi",
        # French
        Language.FR: "fr",
        Language.FR_BE: "fr",
        Language.FR_CA: "fr",
        Language.FR_CH: "fr",
        Language.FR_FR: "fr",
        # Hindi
        Language.HI: "hi",
        Language.HI_IN: "hi",
        # Hungarian
        Language.HU: "hu",
        Language.HU_HU: "hu",
        # Indonesian
        Language.ID: "id",
        Language.ID_ID: "id",
        # Italian
        Language.IT: "it",
        Language.IT_IT: "it",
        # Japanese
        Language.JA: "ja",
        Language.JA_JP: "ja",
        # Korean
        Language.KO: "ko",
        Language.KO_KR: "ko",
        # Dutch
        Language.NL: "nl",
        Language.NL_BE: "nl",
        Language.NL_NL: "nl",
        # Polish
        Language.PL: "pl",
        Language.PL_PL: "pl",
        # Portuguese
        Language.PT: "pt",
        Language.PT_BR: "pt",
        Language.PT_PT: "pt",
        # Romanian
        Language.RO: "ro",
        Language.RO_RO: "ro",
        # Russian
        Language.RU: "ru",
        Language.RU_RU: "ru",
        # Slovak
        Language.SK: "sk",
        Language.SK_SK: "sk",
        # Swedish
        Language.SV: "sv",
        Language.SV_SE: "sv",
        # Thai
        Language.TH: "th",
        Language.TH_TH: "th",
        # Turkish
        Language.TR: "tr",
        Language.TR_TR: "tr",
        # Ukrainian
        Language.UK: "uk",
        Language.UK_UA: "uk",
        # Urdu
        Language.UR: "ur",
        Language.UR_IN: "ur",
        Language.UR_PK: "ur",
        # Vietnamese
        Language.VI: "vi",
        Language.VI_VN: "vi",
        # Chinese
        Language.ZH: "zh",
        Language.ZH_CN: "zh",
        Language.ZH_HK: "zh",
        Language.ZH_TW: "zh",
    }
    return language_map.get(language)


class WhisperSTTService(SegmentedSTTService):
    """Class to transcribe audio with a locally-downloaded Whisper model.

    This service uses Faster Whisper to perform speech-to-text transcription on audio
    segments. It supports multiple languages and various model sizes.

    Args:
        model: The Whisper model to use for transcription. Can be a Model enum or string.
        device: The device to run inference on ('cpu', 'cuda', or 'auto').
        compute_type: The compute type for inference ('default', 'int8', 'int8_float16', etc.).
        no_speech_prob: Probability threshold for filtering out non-speech segments.
        language: The default language for transcription.
        **kwargs: Additional arguments passed to SegmentedSTTService.

    Attributes:
        _device: The device used for inference.
        _compute_type: The compute type for inference.
        _no_speech_prob: Threshold for non-speech filtering.
        _model: The loaded Whisper model instance.
        _settings: Dictionary containing service settings.
    """

    def __init__(
        self,
        *,
        model: str | Model = Model.DISTIL_MEDIUM_EN,
        device: str = "auto",
        compute_type: str = "default",
        no_speech_prob: float = 0.4,
        language: Language = Language.EN,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._device: str = device
        self._compute_type = compute_type
        self.set_model_name(model if isinstance(model, str) else model.value)
        self._no_speech_prob = no_speech_prob
        self._model: Optional[WhisperModel] = None

        self._settings = {
            "language": language,
            "device": self._device,
            "compute_type": self._compute_type,
            "no_speech_prob": self._no_speech_prob,
        }

        self._load()

    def can_generate_metrics(self) -> bool:
        """Indicates whether this service can generate metrics.

        Returns:
            bool: True, as this service supports metric generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert from pipecat Language to Whisper language code.

        Args:
            language: The Language enum value to convert.

        Returns:
            str or None: The corresponding Whisper language code, or None if not supported.
        """
        return language_to_whisper_language(language)

    async def set_language(self, language: Language):
        """Set the language for transcription.

        Args:
            language: The Language enum value to use for transcription.
        """
        logger.info(f"Switching STT language to: [{language}]")
        self._settings["language"] = language

    def _load(self):
        """Loads the Whisper model.

        Note:
            If this is the first time this model is being run,
            it will take time to download from the Hugging Face model hub.
        """
        try:
            from faster_whisper import WhisperModel

            logger.debug("Loading Whisper model...")
            self._model = WhisperModel(
                self.model_name, device=self._device, compute_type=self._compute_type
            )
            logger.debug("Loaded Whisper model")
        except ModuleNotFoundError as e:
            logger.error(f"Exception: {e}")
            logger.error("In order to use Whisper, you need to `pip install pipecat-ai[whisper]`.")
            self._model = None

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribes given audio using Whisper.

        Args:
            audio: Raw audio bytes in 16-bit PCM format.

        Yields:
            Frame: Either a TranscriptionFrame containing the transcribed text
                  or an ErrorFrame if transcription fails.

        Note:
            The audio is expected to be 16-bit signed PCM data.
            The service will normalize it to float32 in the range [-1, 1].
        """
        if not self._model:
            logger.error(f"{self} error: Whisper model not available")
            yield ErrorFrame("Whisper model not available")
            return

        await self.start_processing_metrics()
        await self.start_ttfb_metrics()

        # Divide by 32768 because we have signed 16-bit data.
        audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        whisper_lang = self.language_to_service_language(self._settings["language"])
        segments, _ = await asyncio.to_thread(
            self._model.transcribe, audio_float, language=whisper_lang
        )
        text: str = ""
        for segment in segments:
            if segment.no_speech_prob < self._no_speech_prob:
                text += f"{segment.text} "

        await self.stop_ttfb_metrics()
        await self.stop_processing_metrics()

        if text:
            await self._handle_transcription(text, True, self._settings["language"])
            logger.debug(f"Transcription: [{text}]")
            yield TranscriptionFrame(text, "", time_now_iso8601(), self._settings["language"])


class WhisperSTTServiceMLX(WhisperSTTService):
    """Subclass of `WhisperSTTService` with MLX Whisper model support.

    This service uses MLX Whisper to perform speech-to-text transcription on audio
    segments. It's optimized for Apple Silicon and supports multiple languages and quantizations.

    Args:
        model: The MLX Whisper model to use for transcription. Can be an MLXModel enum or string.
        no_speech_prob: Probability threshold for filtering out non-speech segments.
        language: The default language for transcription.
        temperature: Temperature for sampling. Can be a float or tuple of floats.
        **kwargs: Additional arguments passed to SegmentedSTTService.

    Attributes:
        _no_speech_threshold: Threshold for non-speech filtering.
        _temperature: Temperature for sampling.
        _settings: Dictionary containing service settings.
    """

    def __init__(
        self,
        *,
        model: str | MLXModel = MLXModel.TINY,
        no_speech_prob: float = 0.6,
        language: Language = Language.EN,
        temperature: float = 0.0,
        **kwargs,
    ):
        # Skip WhisperSTTService.__init__ and call its parent directly
        SegmentedSTTService.__init__(self, **kwargs)

        self.set_model_name(model if isinstance(model, str) else model.value)
        self._no_speech_prob = no_speech_prob
        self._temperature = temperature

        self._settings = {
            "language": language,
            "no_speech_prob": self._no_speech_prob,
            "temperature": self._temperature,
            "engine": "mlx",
        }

        # No need to call _load() as MLX Whisper loads models on demand

    @override
    def _load(self):
        """MLX Whisper loads models on demand, so this is a no-op."""
        pass

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing."""
        pass

    @override
    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribes given audio using MLX Whisper.

        Args:
            audio: Raw audio bytes in 16-bit PCM format.

        Yields:
            Frame: Either a TranscriptionFrame containing the transcribed text
                  or an ErrorFrame if transcription fails.

        Note:
            The audio is expected to be 16-bit signed PCM data.
            MLX Whisper will handle the conversion internally.
        """
        try:
            import mlx_whisper

            await self.start_processing_metrics()
            await self.start_ttfb_metrics()

            # Divide by 32768 because we have signed 16-bit data.
            audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

            whisper_lang = self.language_to_service_language(self._settings["language"])
            chunk = await asyncio.to_thread(
                mlx_whisper.transcribe,
                audio_float,
                path_or_hf_repo=self.model_name,
                temperature=self._temperature,
                language=whisper_lang,
            )
            text: str = ""
            for segment in chunk.get("segments", []):
                # Drop likely hallucinations
                if segment.get("compression_ratio", None) == 0.5555555555555556:
                    continue

                if segment.get("no_speech_prob", 0.0) < self._no_speech_prob:
                    text += f"{segment.get('text', '')} "

            if len(text.strip()) == 0:
                text = None

            await self.stop_ttfb_metrics()
            await self.stop_processing_metrics()

            if text:
                await self._handle_transcription(text, True, self._settings["language"])
                logger.debug(f"Transcription: [{text}]")
                yield TranscriptionFrame(text, "", time_now_iso8601(), self._settings["language"])

        except Exception as e:
            logger.exception(f"MLX Whisper transcription error: {e}")
            yield ErrorFrame(f"MLX Whisper transcription error: {str(e)}")
