#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI text-to-speech service implementation.

This module provides integration with OpenAI's text-to-speech API for
generating high-quality synthetic speech from text input.
"""

import base64
import json
from typing import AsyncGenerator, Dict, Literal, Optional

import aiohttp
from loguru import logger
from openai import AsyncOpenAI, BadRequestError
from pydantic import BaseModel

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

ValidVoice = Literal[
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "onyx",
    "nova",
    "sage",
    "shimmer",
    "verse",
]

VALID_VOICES: Dict[str, ValidVoice] = {
    "alloy": "alloy",
    "ash": "ash",
    "ballad": "ballad",
    "coral": "coral",
    "echo": "echo",
    "fable": "fable",
    "onyx": "onyx",
    "nova": "nova",
    "sage": "sage",
    "shimmer": "shimmer",
    "verse": "verse",
}


def parse_sse_chunk(chunk: str) -> Optional[dict]:
    """Parse a raw SSE chunk to OpenAI stream audio event.

    Args:
        chunk: The raw SSE chunk to parse.

    Returns:
        A dictionary containing the parsed data or None if the chunk is empty.
    """
    data_lines = []

    for line in chunk.splitlines():
        if not line:
            continue
        if ":" in line:
            _, value = line.split(":", 1)
            value = value.lstrip()  # remove one leading space if present
        else:
            _, value = line, ""

        data_lines.append(value)

    # Join all data lines and parse as JSON
    data_str = "\n".join(data_lines)

    if data_str.strip() == "[DONE]":
        return

    parsed_data = None
    if data_str:
        try:
            parsed_data = json.loads(data_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in SSE chunk: {e}")

    return parsed_data


class OpenAITTSService(TTSService):
    """OpenAI Text-to-Speech service that generates audio from text.

    This service uses the OpenAI TTS API to generate PCM-encoded audio at 24kHz.
    Supports multiple voice models and configurable parameters for high-quality
    speech synthesis with streaming audio output.
    """

    OPENAI_BASE_AUDIO_URL = "https://api.openai.com/v1/audio/speech"
    OPENAI_SAMPLE_RATE = 24000  # OpenAI TTS always outputs at 24kHz

    class InputParams(BaseModel):
        """Input parameters for OpenAI TTS configuration.

        Parameters:
            instructions: Instructions to guide voice synthesis behavior.
            speed: Voice speed control (0.25 to 4.0, default 1.0).
        """

        instructions: Optional[str] = None
        speed: Optional[float] = None
        stream_format: Optional[Literal["sse", "audio"]] = "audio"
        stream_chunk_size: Optional[int] = 960

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        voice: str = "alloy",
        model: str = "gpt-4o-mini-tts",
        sample_rate: Optional[int] = None,
        instructions: Optional[str] = None,
        speed: Optional[float] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize OpenAI TTS service.

        Args:
            api_key: OpenAI API key for authentication. If None, uses environment variable.
            base_url: Custom base URL for OpenAI API. If None, uses default.
            voice: Voice ID to use for synthesis. Defaults to "alloy".
            model: TTS model to use. Defaults to "gpt-4o-mini-tts".
            sample_rate: Output audio sample rate in Hz. If None, uses OpenAI's default 24kHz.
            instructions: Optional instructions to guide voice synthesis behavior.
            speed: Voice speed control (0.25 to 4.0, default 1.0).
            params: Optional synthesis controls (acting instructions, speed, ...).
            **kwargs: Additional keyword arguments passed to TTSService.

                .. deprecated:: 0.0.91
                        The `instructions` and `speed` parameters are deprecated, use `InputParams` instead.
        """
        if sample_rate and sample_rate != self.OPENAI_SAMPLE_RATE:
            logger.warning(
                f"OpenAI TTS only supports {self.OPENAI_SAMPLE_RATE}Hz sample rate. "
                f"Current rate of {sample_rate}Hz may cause issues."
            )
        super().__init__(sample_rate=sample_rate, **kwargs)

        self.set_model_name(model)
        self.set_voice(voice)
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._api_key = api_key
        self._base_url = base_url if base_url else self.OPENAI_BASE_AUDIO_URL
        if instructions or speed:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "The `instructions` and `speed` parameters are deprecated, use `InputParams` instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

        self._settings = {
            "instructions": params.instructions if params else instructions,
            "speed": params.speed if params else speed,
            "stream_format": params.stream_format if params else "audio",
            "stream_chunk_size": params.stream_chunk_size if params else 960,
        }

        if self._settings["stream_format"] == "sse" and model in ["tts-1", "tts-1-hd"]:
            logger.warning(
                "OpenAI SSE streaming is not supported for models tts-1 and tts-1-hd. Using 'audio' format instead."
            )
            self._settings["stream_format"] = "audio"

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as OpenAI TTS service supports metrics generation.
        """
        return True

    @property
    def includes_inter_frame_spaces(self) -> bool:
        """Indicates that OpenAI TTSTextFrames include necessary inter-frame spaces.

        Returns:
            True, indicating that OpenAI's text frames include necessary inter-frame spaces.
        """
        return True

    async def set_model(self, model: str):
        """Set the TTS model to use.

        Args:
            model: The model name to use for text-to-speech synthesis.
        """
        logger.info(f"Switching TTS model to: [{model}]")
        self.set_model_name(model)

    async def start(self, frame: StartFrame):
        """Start the OpenAI TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        if self.sample_rate != self.OPENAI_SAMPLE_RATE:
            logger.warning(
                f"OpenAI TTS requires {self.OPENAI_SAMPLE_RATE}Hz sample rate. "
                f"Current rate of {self.sample_rate}Hz may cause issues."
            )

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using OpenAI's TTS API.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech data.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")
        try:
            await self.start_ttfb_metrics()

            # Setup API parameters
            create_params = {
                "input": text,
                "model": self.model_name,
                "voice": VALID_VOICES[self._voice_id],
                "response_format": "pcm",
            }

            if self._settings["instructions"]:
                create_params["instructions"] = self._settings["instructions"]

            if self._settings["speed"]:
                create_params["speed"] = self._settings["speed"]

            if self._settings["stream_format"] == "audio":
                async with self._client.audio.speech.with_streaming_response.create(
                    **create_params
                ) as r:
                    if r.status_code != 200:
                        error = await r.text()
                        logger.error(
                            f"{self} error getting audio (status: {r.status_code}, error: {error})"
                        )
                        yield ErrorFrame(
                            f"Error getting audio (status: {r.status_code}, error: {error})"
                        )
                        return

                    await self.start_tts_usage_metrics(text)

                    CHUNK_SIZE = self.chunk_size

                    yield TTSStartedFrame()
                    async for chunk in r.iter_bytes(CHUNK_SIZE):
                        if len(chunk) > 0:
                            await self.stop_ttfb_metrics()
                            frame = TTSAudioRawFrame(chunk, self.sample_rate, 1)
                            yield frame
                    yield TTSStoppedFrame()
            else:
                create_params["stream_format"] = "sse"
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self._base_url,
                        json=create_params,
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                        },
                    ) as r:
                        if r.status != 200:
                            error = await r.text()
                            logger.error(
                                f"{self} error getting audio (status: {r.status}, error: {error})"
                            )
                            yield ErrorFrame(
                                f"Error getting audio (status: {r.status}, error: {error})"
                            )
                            return

                        await self.start_tts_usage_metrics(text)

                        yield TTSStartedFrame()

                        buf = ""

                        async for chunk in r.content.iter_chunked(
                            self._settings["stream_chunk_size"]
                        ):
                            if not chunk:
                                break

                            buf += chunk.decode(errors="replace")

                            while "\n\n" in buf or "\r\n\r\n" in buf:
                                if "\r\n\r\n" in buf:
                                    raw, buf = buf.split("\r\n\r\n", 1)
                                else:
                                    raw, buf = buf.split("\n\n", 1)

                                event = parse_sse_chunk(raw)

                                if not event:
                                    continue

                                type = event.get("type")
                                if type == "speech.audio.delta":
                                    audio = event.get("audio")
                                    try:
                                        audio_bytes = base64.b64decode(audio)
                                        if len(audio_bytes) > 0:
                                            await self.stop_ttfb_metrics()
                                            yield TTSAudioRawFrame(audio_bytes, self.sample_rate, 1)
                                    except Exception as e:
                                        logger.error(f"Error decoding audio: {e}")
                                elif type == "speech.audio.done":
                                    break
                                else:
                                    logger.warning(f"Unknown event type: {type}")

                        yield TTSStoppedFrame()
        except aiohttp.ClientError as e:
            logger.exception(f"{self} HTTP error generating TTS: {e}")
        except BadRequestError as e:
            logger.exception(f"{self} error generating TTS: {e}")
