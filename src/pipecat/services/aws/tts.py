#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AWS Polly text-to-speech service implementation.

This module provides integration with Amazon Polly for text-to-speech synthesis,
supporting multiple languages, voices, and SSML features.
"""

import asyncio
import os
from typing import AsyncGenerator, List, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    import aioboto3
    from botocore.exceptions import BotoCoreError, ClientError
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use AWS services, you need to `pip install pipecat-ai[aws]`.")
    raise Exception(f"Missing module: {e}")


def language_to_aws_language(language: Language) -> Optional[str]:
    """Convert a Language enum to AWS Polly language code.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding AWS Polly language code, or None if not supported.
    """
    language_map = {
        # Arabic
        Language.AR: "arb",
        Language.AR_AE: "ar-AE",
        # Catalan
        Language.CA: "ca-ES",
        # Chinese
        Language.ZH: "cmn-CN",  # Mandarin
        Language.YUE: "yue-CN",  # Cantonese
        Language.YUE_CN: "yue-CN",
        # Czech
        Language.CS: "cs-CZ",
        # Danish
        Language.DA: "da-DK",
        # Dutch
        Language.NL: "nl-NL",
        Language.NL_BE: "nl-BE",
        # English
        Language.EN: "en-US",  # Default to US English
        Language.EN_AU: "en-AU",
        Language.EN_GB: "en-GB",
        Language.EN_IN: "en-IN",
        Language.EN_NZ: "en-NZ",
        Language.EN_US: "en-US",
        Language.EN_ZA: "en-ZA",
        # Finnish
        Language.FI: "fi-FI",
        # French
        Language.FR: "fr-FR",
        Language.FR_BE: "fr-BE",
        Language.FR_CA: "fr-CA",
        # German
        Language.DE: "de-DE",
        Language.DE_AT: "de-AT",
        Language.DE_CH: "de-CH",
        # Hindi
        Language.HI: "hi-IN",
        # Icelandic
        Language.IS: "is-IS",
        # Italian
        Language.IT: "it-IT",
        # Japanese
        Language.JA: "ja-JP",
        # Korean
        Language.KO: "ko-KR",
        # Norwegian
        Language.NO: "nb-NO",
        Language.NB: "nb-NO",
        Language.NB_NO: "nb-NO",
        # Polish
        Language.PL: "pl-PL",
        # Portuguese
        Language.PT: "pt-PT",
        Language.PT_BR: "pt-BR",
        Language.PT_PT: "pt-PT",
        # Romanian
        Language.RO: "ro-RO",
        # Russian
        Language.RU: "ru-RU",
        # Spanish
        Language.ES: "es-ES",
        Language.ES_MX: "es-MX",
        Language.ES_US: "es-US",
        # Swedish
        Language.SV: "sv-SE",
        # Turkish
        Language.TR: "tr-TR",
        # Welsh
        Language.CY: "cy-GB",
        Language.CY_GB: "cy-GB",
    }

    return language_map.get(language)


class AWSPollyTTSService(TTSService):
    """AWS Polly text-to-speech service.

    Provides text-to-speech synthesis using Amazon Polly with support for
    multiple languages, voices, SSML features, and voice customization
    options including prosody controls.
    """

    class InputParams(BaseModel):
        """Input parameters for AWS Polly TTS configuration.

        Parameters:
            engine: TTS engine to use ('standard', 'neural', etc.).
            language: Language for synthesis. Defaults to English.
            pitch: Voice pitch adjustment (for standard engine only).
            rate: Speech rate adjustment.
            volume: Voice volume adjustment.
            lexicon_names: List of pronunciation lexicons to apply.
        """

        engine: Optional[str] = None
        language: Optional[Language] = Language.EN
        pitch: Optional[str] = None
        rate: Optional[str] = None
        volume: Optional[str] = None
        lexicon_names: Optional[List[str]] = None

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        region: Optional[str] = None,
        voice_id: str = "Joanna",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initializes the AWS Polly TTS service.

        Args:
            api_key: AWS secret access key. If None, uses AWS_SECRET_ACCESS_KEY environment variable.
            aws_access_key_id: AWS access key ID. If None, uses AWS_ACCESS_KEY_ID environment variable.
            aws_session_token: AWS session token for temporary credentials.
            region: AWS region for Polly service. Defaults to 'us-east-1'.
            voice_id: Voice ID to use for synthesis. Defaults to 'Joanna'.
            sample_rate: Audio sample rate. If None, uses service default.
            params: Additional input parameters for voice customization.
            **kwargs: Additional arguments passed to parent TTSService class.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        params = params or AWSPollyTTSService.InputParams()

        # Get credentials from environment variables if not provided
        self._aws_params = {
            "aws_access_key_id": aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID"),
            "aws_secret_access_key": api_key or os.getenv("AWS_SECRET_ACCESS_KEY"),
            "aws_session_token": aws_session_token or os.getenv("AWS_SESSION_TOKEN"),
            "region_name": region or os.getenv("AWS_REGION", "us-east-1"),
        }

        # Validate that we have the required credentials
        if (
            not self._aws_params["aws_access_key_id"]
            or not self._aws_params["aws_secret_access_key"]
        ):
            raise ValueError(
                "AWS credentials not found. Please provide them either through constructor parameters "
                "or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
            )

        self._aws_session = aioboto3.Session()
        self._settings = {
            "engine": params.engine,
            "language": self.language_to_service_language(params.language)
            if params.language
            else "en-US",
            "pitch": params.pitch,
            "rate": params.rate,
            "volume": params.volume,
            "lexicon_names": params.lexicon_names,
        }

        self._resampler = create_stream_resampler()

        self.set_voice(voice_id)

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as AWS Polly service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to AWS Polly language format.

        Args:
            language: The language to convert.

        Returns:
            The AWS Polly-specific language code, or None if not supported.
        """
        return language_to_aws_language(language)

    def _construct_ssml(self, text: str) -> str:
        ssml = "<speak>"

        language = self._settings["language"]
        ssml += f"<lang xml:lang='{language}'>"

        prosody_attrs = []
        # Prosody tags are only supported for standard and neural engines
        if self._settings["engine"] == "standard":
            if self._settings["pitch"]:
                prosody_attrs.append(f"pitch='{self._settings['pitch']}'")

        if self._settings["rate"]:
            prosody_attrs.append(f"rate='{self._settings['rate']}'")
        if self._settings["volume"]:
            prosody_attrs.append(f"volume='{self._settings['volume']}'")

        if prosody_attrs:
            ssml += f"<prosody {' '.join(prosody_attrs)}>"

        ssml += text

        if prosody_attrs:
            ssml += "</prosody>"

        ssml += "</lang>"

        ssml += "</speak>"

        logger.trace(f"{self} SSML: {ssml}")

        return ssml

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using AWS Polly.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            await self.start_ttfb_metrics()

            # Construct the parameters dictionary
            ssml = self._construct_ssml(text)

            params = {
                "Text": ssml,
                "TextType": "ssml",
                "OutputFormat": "pcm",
                "VoiceId": self._voice_id,
                "Engine": self._settings["engine"],
                # AWS only supports 8000 and 16000 for PCM. We select 16000.
                "SampleRate": "16000",
                "LexiconNames": self._settings["lexicon_names"],
            }

            # Filter out None values
            filtered_params = {k: v for k, v in params.items() if v is not None}

            async with self._aws_session.client("polly", **self._aws_params) as polly:
                response = await polly.synthesize_speech(**filtered_params)
                if "AudioStream" in response:
                    # Get the streaming body and read it
                    stream = response["AudioStream"]
                    audio_data = await stream.read()
                else:
                    logger.error(f"{self} No audio stream in response")
                    audio_data = None

                audio_data = await self._resampler.resample(audio_data, 16000, self.sample_rate)

                await self.start_tts_usage_metrics(text)

                yield TTSStartedFrame()

                CHUNK_SIZE = self.chunk_size

                for i in range(0, len(audio_data), CHUNK_SIZE):
                    chunk = audio_data[i : i + CHUNK_SIZE]
                    if len(chunk) > 0:
                        await self.stop_ttfb_metrics()
                        frame = TTSAudioRawFrame(chunk, self.sample_rate, 1)
                        yield frame

                yield TTSStoppedFrame()
        except (BotoCoreError, ClientError) as error:
            logger.exception(f"{self} error generating TTS: {error}")
            error_message = f"AWS Polly TTS error: {str(error)}"
            yield ErrorFrame(error=error_message)

        finally:
            yield TTSStoppedFrame()


class PollyTTSService(AWSPollyTTSService):
    """Deprecated alias for AWSPollyTTSService.

    .. deprecated:: 0.0.67
        `PollyTTSService` is deprecated, use `AWSPollyTTSService` instead.

    """

    def __init__(self, **kwargs):
        """Initialize the deprecated PollyTTSService.

        Args:
            **kwargs: All arguments passed to AWSPollyTTSService.
        """
        super().__init__(**kwargs)

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "'PollyTTSService' is deprecated, use 'AWSPollyTTSService' instead.",
                DeprecationWarning,
            )
