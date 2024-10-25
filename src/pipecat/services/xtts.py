#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Any, AsyncGenerator, Dict

import aiohttp

from pipecat.audio.utils import resample_audio
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.ai_services import TTSService
from pipecat.transcriptions.language import Language

from loguru import logger


# The server below can connect to XTTS through a local running docker
#
# Docker command: $ docker run --gpus=all -e COQUI_TOS_AGREED=1 --rm -p 8000:80 ghcr.io/coqui-ai/xtts-streaming-server:latest-cuda121
#
# You can find more information on the official repo:
# https://github.com/coqui-ai/xtts-streaming-server


class XTTSService(TTSService):
    def __init__(
        self,
        *,
        voice_id: str,
        language: Language,
        base_url: str,
        aiohttp_session: aiohttp.ClientSession,
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._settings = {
            "language": self.language_to_service_language(language),
            "base_url": base_url,
        }
        self.set_voice(voice_id)
        self._studio_speakers: Dict[str, Any] | None = None
        self._aiohttp_session = aiohttp_session

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        match language:
            case Language.CS:
                return "cs"
            case Language.DE:
                return "de"
            case (
                Language.EN
                | Language.EN_US
                | Language.EN_AU
                | Language.EN_GB
                | Language.EN_NZ
                | Language.EN_IN
            ):
                return "en"
            case Language.ES:
                return "es"
            case Language.FR:
                return "fr"
            case Language.HI:
                return "hi"
            case Language.HU:
                return "hu"
            case Language.IT:
                return "it"
            case Language.JA:
                return "ja"
            case Language.KO:
                return "ko"
            case Language.NL:
                return "nl"
            case Language.PL:
                return "pl"
            case Language.PT | Language.PT_BR:
                return "pt"
            case Language.RU:
                return "ru"
            case Language.TR:
                return "tr"
            case Language.ZH:
                return "zh-cn"
        return None

    async def start(self, frame: StartFrame):
        await super().start(frame)
        async with self._aiohttp_session.get(self._settings["base_url"] + "/studio_speakers") as r:
            if r.status != 200:
                text = await r.text()
                logger.error(
                    f"{self} error getting studio speakers (status: {r.status}, error: {text})"
                )
                await self.push_error(
                    ErrorFrame(
                        f"Error error getting studio speakers (status: {r.status}, error: {text})"
                    )
                )
                return
            self._studio_speakers = await r.json()

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        if not self._studio_speakers:
            logger.error(f"{self} no studio speakers available")
            return

        embeddings = self._studio_speakers[self._voice_id]

        url = self._settings["base_url"] + "/tts_stream"

        payload = {
            "text": text.replace(".", "").replace("*", ""),
            "language": self._settings["language"],
            "speaker_embedding": embeddings["speaker_embedding"],
            "gpt_cond_latent": embeddings["gpt_cond_latent"],
            "add_wav_header": False,
            "stream_chunk_size": 20,
        }

        await self.start_ttfb_metrics()

        async with self._aiohttp_session.post(url, json=payload) as r:
            if r.status != 200:
                text = await r.text()
                logger.error(f"{self} error getting audio (status: {r.status}, error: {text})")
                yield ErrorFrame(f"Error getting audio (status: {r.status}, error: {text})")
                return

            await self.start_tts_usage_metrics(text)

            yield TTSStartedFrame()

            buffer = bytearray()
            async for chunk in r.content.iter_chunked(1024):
                if len(chunk) > 0:
                    await self.stop_ttfb_metrics()
                    # Append new chunk to the buffer.
                    buffer.extend(chunk)

                    # Check if buffer has enough data for processing.
                    while (
                        len(buffer) >= 48000
                    ):  # Assuming at least 0.5 seconds of audio data at 24000 Hz
                        # Process the buffer up to a safe size for resampling.
                        process_data = buffer[:48000]
                        # Remove processed data from buffer.
                        buffer = buffer[48000:]

                        # XTTS uses 24000 so we need to resample to our desired rate.
                        resampled_audio = resample_audio(
                            bytes(process_data), 24000, self._sample_rate
                        )
                        # Create the frame with the resampled audio
                        frame = TTSAudioRawFrame(resampled_audio, self._sample_rate, 1)
                        yield frame

            # Process any remaining data in the buffer.
            if len(buffer) > 0:
                resampled_audio = resample_audio(bytes(buffer), 24000, self._sample_rate)
                frame = TTSAudioRawFrame(resampled_audio, self._sample_rate, 1)
                yield frame

            yield TTSStoppedFrame()
