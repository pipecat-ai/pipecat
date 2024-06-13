#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import io
import struct

from typing import AsyncGenerator

from pipecat.frames.frames import AudioRawFrame, Frame
from pipecat.services.ai_services import TTSService

from loguru import logger

try:
    from pyht.client import TTSOptions
    from pyht.async_client import AsyncClient
    from pyht.protos.api_pb2 import Format
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use PlayHT, you need to `pip install pipecat-ai[playht]`. Also, set `PLAY_HT_USER_ID` and `PLAY_HT_API_KEY` environment variables.")
    raise Exception(f"Missing module: {e}")


class PlayHTTTSService(TTSService):

    def __init__(self, *, api_key: str, user_id: str, voice_url: str, **kwargs):
        super().__init__(**kwargs)

        self._user_id = user_id
        self._speech_key = api_key

        self._client = AsyncClient(
            user_id=self._user_id,
            api_key=self._speech_key,
        )
        self._options = TTSOptions(
            voice=voice_url,
            sample_rate=16000,
            quality="higher",
            format=Format.FORMAT_WAV)

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        try:
            b = bytearray()
            in_header = True

            await self.start_ttfb_metrics()

            playht_gen = self._client.tts(
                text,
                voice_engine="PlayHT2.0-turbo",
                options=self._options)

            async for chunk in playht_gen:
                # skip the RIFF header.
                if in_header:
                    b.extend(chunk)
                    if len(b) <= 36:
                        continue
                    else:
                        fh = io.BytesIO(b)
                        fh.seek(36)
                        (data, size) = struct.unpack('<4sI', fh.read(8))
                        while data != b'data':
                            fh.read(size)
                            (data, size) = struct.unpack('<4sI', fh.read(8))
                        in_header = False
                else:
                    if len(chunk):
                        await self.stop_ttfb_metrics()
                        frame = AudioRawFrame(chunk, 16000, 1)
                        yield frame
        except Exception as e:
            logger.error(f"{self} error generating TTS: {e}")
