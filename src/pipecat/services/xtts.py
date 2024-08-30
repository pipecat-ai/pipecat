#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import aiohttp

from typing import Any, AsyncGenerator, Dict

from pipecat.frames.frames import (
    AudioRawFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSStartedFrame,
    TTSStoppedFrame)
from pipecat.services.ai_services import TTSService

from loguru import logger

import numpy as np

try:
    import resampy
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use XTTS, you need to `pip install pipecat-ai[xtts]`.")
    raise Exception(f"Missing module: {e}")


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
            language: str,
            base_url: str,
            aiohttp_session: aiohttp.ClientSession,
            **kwargs):
        super().__init__(**kwargs)

        self._voice_id = voice_id
        self._language = language
        self._base_url = base_url
        self._studio_speakers: Dict[str, Any] | None = None
        self._aiohttp_session = aiohttp_session

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)
        async with self._aiohttp_session.get(self._base_url + "/studio_speakers") as r:
            if r.status != 200:
                text = await r.text()
                logger.error(
                    f"{self} error getting studio speakers (status: {r.status}, error: {text})")
                await self.push_error(
                    ErrorFrame(f"Error error getting studio speakers (status: {r.status}, error: {text})"))
                return
            self._studio_speakers = await r.json()

    async def set_voice(self, voice: str):
        logger.debug(f"Switching TTS voice to: [{voice}]")
        self._voice_id = voice

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        if not self._studio_speakers:
            logger.error(f"{self} no studio speakers available")
            return

        embeddings = self._studio_speakers[self._voice_id]

        url = self._base_url + "/tts_stream"

        payload = {
            "text": text.replace('.', '').replace('*', ''),
            "language": self._language,
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

            await self.push_frame(TTSStartedFrame())

            buffer = bytearray()
            async for chunk in r.content.iter_chunked(1024):
                if len(chunk) > 0:
                    await self.stop_ttfb_metrics()
                    # Append new chunk to the buffer
                    buffer.extend(chunk)

                    # Check if buffer has enough data for processing
                    while len(buffer) >= 48000:  # Assuming at least 0.5 seconds of audio data at 24000 Hz
                        # Process the buffer up to a safe size for resampling
                        process_data = buffer[:48000]
                        # Remove processed data from buffer
                        buffer = buffer[48000:]

                        # Convert the byte data to numpy array for resampling
                        audio_np = np.frombuffer(process_data, dtype=np.int16)
                        # Resample the audio from 24000 Hz to 16000 Hz
                        resampled_audio = resampy.resample(audio_np, 24000, 16000)
                        # Convert the numpy array back to bytes
                        resampled_audio_bytes = resampled_audio.astype(np.int16).tobytes()
                        # Create the frame with the resampled audio
                        frame = AudioRawFrame(resampled_audio_bytes, 16000, 1)
                        yield frame

            # Process any remaining data in the buffer
            if len(buffer) > 0:
                audio_np = np.frombuffer(buffer, dtype=np.int16)
                resampled_audio = resampy.resample(audio_np, 24000, 16000)
                resampled_audio_bytes = resampled_audio.astype(np.int16).tobytes()
                frame = AudioRawFrame(resampled_audio_bytes, 16000, 1)
                yield frame

            await self.push_frame(TTSStoppedFrame())
