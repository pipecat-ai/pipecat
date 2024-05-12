#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import io
import struct

from pipecat.frames.frames import AudioRawFrame
from pipecat.services.ai_services import TTSService

from loguru import logger

try:
    from pyht import Client
    from pyht.client import TTSOptions
    from pyht.protos.api_pb2 import Format
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use PlayHT, you need to `pip install pipecat[playht]`. Also, set `PLAY_HT_USER_ID` and `PLAY_HT_API_KEY` environment variables.")
    raise Exception(f"Missing module: {e}")


class PlayHTAIService(TTSService):

    def __init__(self, *, api_key, user_id, voice_url):
        super().__init__()

        self._user_id = user_id
        self._speech_key = api_key

        self._client = Client(
            user_id=self._user_id,
            api_key=self._speech_key,
        )
        self._options = TTSOptions(
            voice=voice_url,
            sample_rate=16000,
            quality="higher",
            format=Format.FORMAT_WAV)

    def __del__(self):
        self._client.close()

    async def run_tts(self, text: str):
        b = bytearray()
        in_header = True
        for chunk in self._client.tts(text, self._options):
            # skip the RIFF header.
            if in_header:
                b.extend(chunk)
                if len(b) <= 36:
                    continue
                else:
                    fh = io.BytesIO(b)
                    fh.seek(36)
                    (data, size) = struct.unpack('<4sI', fh.read(8))
                    logger.debug(
                        f"first attempt: data: {data}, size: {hex(size)}, position: {fh.tell()}")
                    while data != b'data':
                        fh.read(size)
                        (data, size) = struct.unpack('<4sI', fh.read(8))
                        logger.debug(
                            f"subsequent data: {data}, size: {hex(size)}, position: {fh.tell()}, data != data: {data != b'data'}")
                    logger.debug("position: ", fh.tell())
                    in_header = False
            else:
                if len(chunk):
                    frame = AudioRawFrame(chunk, 16000, 1)
                    await self.push_frame(frame)
