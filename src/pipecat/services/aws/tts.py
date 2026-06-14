#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AWS Polly text-to-speech service implementation.

This module provides integration with Amazon Polly for text-to-speech synthesis,
supporting multiple languages, voices, SSML features, and per-word timestamps
via Polly SpeechMarks.
"""

import asyncio
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

from loguru import logger
from pydantic import BaseModel

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    AggregationType,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.aws.utils import resolve_credentials
from pipecat.services.settings import NOT_GIVEN, TTSSettings, _NotGiven
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    import aiobotocore.session
    from botocore.exceptions import BotoCoreError, ClientError
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use AWS services, you need to `uv add "pipecat-ai[aws]"`.')
    raise ImportError(f"Missing module: {e}") from e


def language_to_aws_language(language: Language) -> str:
    """Convert a Language enum to AWS Polly language code.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding service language code. If ``language`` is not in
        the verified mapping, falls back to the full language code string and
        logs a warning (via ``resolve_language(..., use_base_code=False)``).
    """
    LANGUAGE_MAP = {
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

    return resolve_language(language, LANGUAGE_MAP, use_base_code=False)


@dataclass
class AWSPollyTTSSettings(TTSSettings):
    """Settings for AWSPollyTTSService.

    Parameters:
        engine: TTS engine to use ('standard', 'neural', etc.).
        pitch: Voice pitch adjustment (for standard engine only).
        rate: Speech rate adjustment.
        volume: Voice volume adjustment.
        lexicon_names: List of pronunciation lexicons to apply.
    """

    engine: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    pitch: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    rate: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    volume: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    lexicon_names: list[str] | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class AWSPollyTTSService(TTSService):
    """AWS Polly text-to-speech service.

    Provides text-to-speech synthesis using Amazon Polly with support for
    multiple languages, voices, SSML features, and voice customization
    options including prosody controls.

    When ``word_timestamps`` is enabled (the default), the service makes a
    second Polly call requesting word SpeechMarks and emits one timestamped
    ``TTSTextFrame`` per word, the same behaviour timestamp-capable services
    such as ElevenLabs provide.
    """

    Settings = AWSPollyTTSSettings
    _settings: Settings

    class InputParams(BaseModel):
        """Input parameters for AWS Polly TTS configuration.

        .. deprecated:: 0.0.105
            Use ``AWSPollyTTSService.Settings`` directly via the ``settings`` parameter instead.

        Parameters:
            engine: TTS engine to use ('standard', 'neural', etc.).
            language: Language for synthesis. Defaults to English.
            pitch: Voice pitch adjustment (for standard engine only).
            rate: Speech rate adjustment.
            volume: Voice volume adjustment.
            lexicon_names: List of pronunciation lexicons to apply.
        """

        engine: str | None = None
        language: Language | None = Language.EN
        pitch: str | None = None
        rate: str | None = None
        volume: str | None = None
        lexicon_names: list[str] | None = None

    def __init__(
        self,
        *,
        api_key: str | None = None,
        aws_access_key_id: str | None = None,
        aws_session_token: str | None = None,
        region: str | None = None,
        voice_id: str | None = None,
        sample_rate: int | None = None,
        word_timestamps: bool = True,
        params: InputParams | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initializes the AWS Polly TTS service.

        Args:
            api_key: AWS secret access key. If None, falls back to environment
                variables and the default botocore credential chain (instance
                profiles, IRSA, ECS task roles, SSO, etc.).
            aws_access_key_id: AWS access key ID. Same fallback behaviour as
                ``api_key``.
            aws_session_token: AWS session token for temporary credentials.
            region: AWS region for Polly service. Defaults to 'us-east-1'.
            voice_id: Voice ID to use for synthesis. Defaults to 'Joanna'.

                .. deprecated:: 0.0.105
                    Use ``settings=AWSPollyTTSService.Settings(voice=...)`` instead.

            sample_rate: Audio sample rate. If None, uses service default.
            word_timestamps: Whether to emit per-word timestamped TTSTextFrames
                using Polly SpeechMarks. Defaults to True. When True the service
                makes a second (concurrent) Polly call to fetch word timing.
                Set to False to skip that extra call (e.g. to reduce request
                cost), in which case the base class emits a single whole-sentence
                TTSTextFrame with no timing.
            params: Additional input parameters for voice customization.

                .. deprecated:: 0.0.105
                    Use ``settings=AWSPollyTTSService.Settings(...)`` instead.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional arguments passed to parent TTSService class.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model=None,
            voice="Joanna",
            language="en-US",
            engine=None,
            pitch=None,
            rate=None,
            volume=None,
            lexicon_names=None,
        )

        # 2. Apply direct init arg overrides (deprecated)
        if voice_id is not None:
            self._warn_init_param_moved_to_settings("voice_id", "voice")
            default_settings.voice = voice_id

        # 3. Apply params overrides — only if settings not provided
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings:
                default_settings.engine = params.engine
                default_settings.language = params.language if params.language else "en-US"
                default_settings.pitch = params.pitch
                default_settings.rate = params.rate
                default_settings.volume = params.volume
                default_settings.lexicon_names = params.lexicon_names

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            # push_text_frames=False activates the base word-timestamp path
            # (per-word TTSTextFrames). When word timestamps are disabled we let
            # the base push a single whole-sentence TTSTextFrame instead.
            push_text_frames=not word_timestamps,
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )

        # Resolve credentials using the shared chain (explicit → env → botocore).
        self._aws_params = resolve_credentials(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=api_key,
            aws_session_token=aws_session_token,
            region=region,
        ).to_boto_kwargs()

        self._aws_session = aiobotocore.session.get_session()

        self._resampler = create_stream_resampler()

        self._word_timestamps = word_timestamps
        # Cumulative time offset (seconds) so each sentence's word timestamps
        # land after the previous sentence on the shared turn timeline. Reset at
        # the start of the service and on interruption / end of turn.
        self._cumulative_time = 0.0

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as AWS Polly service supports metrics generation.
        """
        return True

    async def start(self, frame: StartFrame):
        """Start the service.

        Args:
            frame: The start frame.
        """
        await super().start(frame)
        self._cumulative_time = 0.0

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame downstream and reset timing state on interruption or stop.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        await super().push_frame(frame, direction)
        if isinstance(frame, (InterruptionFrame, TTSStoppedFrame)):
            # Reset the cumulative offset at end of turn / on barge-in so the
            # next turn's word timestamps start from zero.
            self._cumulative_time = 0.0

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to AWS Polly language format.

        Args:
            language: The language to convert.

        Returns:
            The AWS Polly-specific language code, or None if not supported.
        """
        return language_to_aws_language(language)

    def _construct_ssml(self, text: str) -> str:
        ssml = "<speak>"

        language = self._settings.language
        ssml += f"<lang xml:lang='{language}'>"

        prosody_attrs = []
        # Prosody tags are only supported for standard and neural engines
        if self._settings.engine == "standard":
            if self._settings.pitch:
                prosody_attrs.append(f"pitch='{self._settings.pitch}'")

        if self._settings.rate:
            prosody_attrs.append(f"rate='{self._settings.rate}'")
        if self._settings.volume:
            prosody_attrs.append(f"volume='{self._settings.volume}'")

        if prosody_attrs:
            ssml += f"<prosody {' '.join(prosody_attrs)}>"

        ssml += text

        if prosody_attrs:
            ssml += "</prosody>"

        ssml += "</lang>"

        ssml += "</speak>"

        logger.trace(f"{self} SSML: {ssml}")

        return ssml

    async def _fetch_word_marks(self, polly, base_params: dict) -> list[tuple[str, float]]:
        """Fetch per-word SpeechMarks from Polly for the same utterance.

        Makes a second ``synthesize_speech`` call with ``OutputFormat="json"``
        and ``SpeechMarkTypes=["word"]``. Polly's ``OutputFormat`` is
        single-valued, so audio and marks cannot come from one call. The marks
        arrive as newline-delimited JSON in the ``AudioStream`` body.

        This never raises: on any failure it returns an empty list so audio is
        unaffected and the caller can fall back to whole-text output.

        Args:
            polly: An open aiobotocore Polly client.
            base_params: The shared synthesis parameters (Text, VoiceId, etc.).

        Returns:
            A list of ``(word, start_seconds_within_this_utterance)`` tuples.
        """
        marks_params = {**base_params, "OutputFormat": "json", "SpeechMarkTypes": ["word"]}
        # SampleRate does not apply to JSON marks.
        marks_params.pop("SampleRate", None)
        try:
            response = await polly.synthesize_speech(**marks_params)
            if "AudioStream" not in response:
                return []
            raw = await response["AudioStream"].read()
            word_times: list[tuple[str, float]] = []
            for line in raw.splitlines():
                if not line.strip():
                    continue
                mark = json.loads(line)
                if mark.get("type") == "word":
                    # Polly reports the start time in milliseconds.
                    word_times.append((mark["value"], mark["time"] / 1000.0))
            return word_times
        except (BotoCoreError, ClientError, ValueError, KeyError) as error:
            logger.warning(f"{self}: Polly speech marks fetch failed: {error}")
            return []

    async def _push_fallback_text(self, text: str, context_id: str):
        """Push the full text when no word marks were returned.

        Without timestamps no per-word ``TTSTextFrame`` is emitted, so push the
        whole text directly to keep the assistant conversation context in sync
        with what was spoken (committed even if the turn is later interrupted).

        Args:
            text: The text that was synthesized.
            context_id: The context ID for this TTS request.
        """
        text_clean = text.rstrip()
        if not text_clean:
            return
        logger.debug(f"{self}: No speech marks received, pushing fallback text: [{text_clean}]")
        fallback = TTSTextFrame(
            text_clean, aggregated_by=AggregationType.SENTENCE, context_id=context_id
        )
        ctx = self._tts_contexts.get(context_id)
        fallback.append_to_context = ctx.append_to_context if ctx else True
        await self.push_frame(fallback)

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using AWS Polly.

        When word timestamps are enabled, a concurrent second Polly call fetches
        word SpeechMarks; the parsed timings are fed to the base class via
        ``add_word_timestamps`` so it can emit one timestamped ``TTSTextFrame``
        per word, aligned to the audio.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        ssml = self._construct_ssml(text)

        # Parameters shared by the audio call and the marks call. Both must use
        # identical Text/Voice/Engine so the word timings line up with the audio.
        base_params = {
            "Text": ssml,
            "TextType": "ssml",
            "VoiceId": self._settings.voice,
            "Engine": self._settings.engine,
            # AWS only supports 8000 and 16000 for PCM. We select 16000.
            "SampleRate": "16000",
            "LexiconNames": self._settings.lexicon_names,
        }
        # Filter out None values
        base_params = {k: v for k, v in base_params.items() if v is not None}

        marks_future = None
        try:
            async with self._aws_session.create_client(  # pyright: ignore[reportGeneralTypeIssues]
                "polly",
                **self._aws_params,  # pyright: ignore[reportArgumentType]
            ) as polly:
                # Kick off the marks call so it overlaps audio synthesis and
                # adds no latency on the critical path (TTFB).
                if self._word_timestamps:
                    marks_future = asyncio.ensure_future(self._fetch_word_marks(polly, base_params))

                audio_params = {**base_params, "OutputFormat": "pcm"}
                response = await polly.synthesize_speech(**audio_params)  # pyright: ignore[reportGeneralTypeIssues]
                if "AudioStream" not in response:
                    logger.error(f"{self} No audio stream in response")
                    return

                pcm = await response["AudioStream"].read()
                # Marks carry only start times, so derive the utterance duration
                # from the PCM length (16-bit mono at 16000 Hz).
                utterance_seconds = len(pcm) / (16000 * 2)

                audio = await self._resampler.resample(pcm, 16000, self.sample_rate)

                await self.start_tts_usage_metrics(text)

                # Yield audio first so the first chunk stops TTFB and anchors the
                # word-timestamp baseline.
                for i in range(0, len(audio), self.chunk_size):
                    chunk = audio[i : i + self.chunk_size]
                    if len(chunk) > 0:
                        await self.stop_ttfb_metrics()
                        yield TTSAudioRawFrame(chunk, self.sample_rate, 1, context_id=context_id)

                # Feed the word marks (already in flight) and advance the offset.
                if marks_future is not None:
                    word_marks = await marks_future
                    marks_future = None
                    if word_marks:
                        word_times = [
                            (word, self._cumulative_time + start) for word, start in word_marks
                        ]
                        await self.add_word_timestamps(word_times, context_id)
                    else:
                        await self._push_fallback_text(text, context_id)
                    # Offset the next sentence by this utterance's audio duration.
                    self._cumulative_time += utterance_seconds

        except (BotoCoreError, ClientError) as error:
            yield ErrorFrame(error=f"AWS Polly TTS error: {str(error)}")
        finally:
            # Make sure a still-pending marks call doesn't dangle if the audio
            # call failed or the generator was closed early (e.g. interruption).
            if marks_future is not None:
                marks_future.cancel()
