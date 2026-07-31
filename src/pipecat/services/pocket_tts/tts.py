#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pocket TTS service implementation using kyutai-labs' pocket-tts."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass
from typing import Any

from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame
from pipecat.services.settings import TTSSettings, assert_given
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    import torch
    from pocket_tts import TTSModel
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use Pocket TTS, you need to `uv add "pipecat-ai[pocket-tts]"`.')
    raise ImportError(f"Missing module: {e}") from e


def language_to_pocket_tts_language(language: Language) -> str:
    """Convert a Language enum to a pocket-tts model language name.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding pocket-tts model language name.
    """
    LANGUAGE_MAP = {
        Language.DE: "german_24l",
        Language.EN: "english",
        Language.ES: "spanish_24l",
        Language.FR: "french_24l",
        Language.IT: "italian_24l",
        Language.PT: "portuguese_24l",
    }

    result = resolve_language(language, LANGUAGE_MAP, use_base_code=True)
    if result not in LANGUAGE_MAP.values():
        # The fallback for unmapped regional variants is a bare base code
        # (e.g. "fr" for FR_CA), which is not a valid model name; map it back
        # onto a model name when one exists for the base language.
        base_map = {str(k): v for k, v in LANGUAGE_MAP.items()}
        result = base_map.get(result, result)
    return result


@dataclass
class PocketTTSSettings(TTSSettings):
    """Settings for PocketTTSService."""

    pass


class PocketTTSService(TTSService):
    """Pocket TTS service implementation.

    Provides local, CPU-only text-to-speech synthesis using kyutai-labs'
    pocket-tts streaming model. Model weights and voice prompts are downloaded
    from Hugging Face on first use, which may block for a while.

    The voice may be a predefined voice name (e.g. ``"alba"``), a local
    ``.wav`` file to clone, an exported ``.safetensors`` voice state, or an
    ``hf://`` path. The voice can be changed at runtime; the language cannot,
    since model weights are loaded per language.
    """

    Settings = PocketTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        temp: float | None = None,
        quantize: bool = False,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Pocket TTS service.

        Args:
            temp: Sampling temperature for audio generation. Defaults to the
                pocket-tts model default.
            quantize: Quantize model weights for faster CPU inference
                (requires the ``torchao`` package).
            settings: Runtime-updatable settings. Defaults to the ``"alba"``
                voice and English.
            **kwargs: Additional arguments passed to the parent `TTSService`.
        """
        # Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model=None,
            voice="alba",
            language=Language.EN,
        )

        # Apply settings delta
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )

        # The base __init__ has already converted a Language enum to the
        # pocket-tts language name via language_to_service_language().
        language = assert_given(self._settings.language)
        if language is None:
            raise ValueError("Pocket TTS language must be specified")

        load_kwargs: dict[str, Any] = {"language": language, "quantize": quantize}
        if temp is not None:
            load_kwargs["temp"] = temp

        logger.debug(f"Loading Pocket TTS '{language}' model")
        self._model = TTSModel.load_model(**load_kwargs)
        logger.debug(f"Loaded Pocket TTS '{language}' model")

        voice = assert_given(self._settings.voice)
        if voice is None:
            raise ValueError("Pocket TTS voice must be specified")

        # Voice state derived from the voice prompt, cached across utterances.
        # generate_audio_stream() is called with copy_state=True so this cache
        # is never mutated by generation; None forces re-derivation.
        self._voice_state: Any = self._model.get_state_for_audio_prompt(voice)

    def can_generate_metrics(self) -> bool:
        """Indicate that this service supports TTFB and usage metrics."""
        return True

    def language_to_service_language(self, language: Language) -> str:
        """Convert a Language enum to a pocket-tts model language name.

        Args:
            language: The language to convert.

        Returns:
            The pocket-tts model language name.
        """
        return language_to_pocket_tts_language(language)

    async def _update_settings(self, delta: Settings) -> dict[str, Any]:
        """Apply a settings delta.

        Voice changes take effect on the next utterance (the voice state is
        re-derived lazily). Language and model changes are stored but not
        applied, since they would require reloading model weights.
        """
        changed = await super()._update_settings(delta)
        if not changed:
            return changed
        if "voice" in changed:
            self._voice_state = None
        unhandled = {k: v for k, v in changed.items() if k != "voice"}
        if unhandled:
            self._warn_unhandled_updated_settings(unhandled)
        return changed

    async def _get_voice_state(self) -> Any:
        """Return the cached voice state, deriving it if invalidated."""
        if self._voice_state is None:
            voice = assert_given(self._settings.voice)
            if voice is None:
                raise ValueError("Pocket TTS voice must be specified")
            logger.debug(f"{self}: deriving voice state for [{voice}]")
            self._voice_state = await asyncio.to_thread(
                self._model.get_state_for_audio_prompt, voice
            )
        return self._voice_state

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using pocket-tts.

        Args:
            text: The text to convert to speech.
            context_id: Unique identifier for this TTS context.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """

        def sync_next(it):
            try:
                return next(it)
            except StopIteration:
                return None

        async def audio_iterator(voice_state) -> AsyncIterator[bytes]:
            # copy_state=True (passed explicitly, matching the library
            # default) keeps the cached voice state unmodified across
            # generations.
            stream = self._model.generate_audio_stream(voice_state, text, copy_state=True)
            while True:
                chunk = await asyncio.to_thread(sync_next, stream)
                if chunk is None:
                    return
                # chunk is a 1-D float32 tensor in [-1, 1] at model.sample_rate.
                yield (chunk.clamp(-1.0, 1.0) * 32767).to(torch.int16).numpy().tobytes()

        try:
            await self.start_tts_usage_metrics(text)

            voice_state = await self._get_voice_state()

            async for frame in self._stream_audio_frames_from_iterator(
                audio_iterator(voice_state),
                in_sample_rate=self._model.sample_rate,
                context_id=context_id,
            ):
                await self.stop_ttfb_metrics()
                yield frame
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
        finally:
            logger.debug(f"{self}: Finished TTS [{text}]")
            await self.stop_ttfb_metrics()
