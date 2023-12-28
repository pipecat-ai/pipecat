import io
import os
import struct
from pyht import Client
from dotenv import load_dotenv
from pyht.client import TTSOptions
from pyht.protos.api_pb2 import Format

from services.ai_service import AIService

class PlayHTAIService(AIService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.speech_key = os.getenv("PLAY_HT_KEY") or ''
        self.user_id = os.getenv("PLAY_HT_USER_ID") or ''

        self.client = Client(
            user_id=self.user_id,
            api_key=self.speech_key,
        )
        self.options = TTSOptions(
            voice="s3://voice-cloning-zero-shot/820da3d2-3a3b-42e7-844d-e68db835a206/sarah/manifest.json",
            sample_rate=16000,
            quality="higher",
            format=Format.FORMAT_WAV
        )

    def close(self):
        super().close()
        self.client.close()

    def run_tts(self, sentence):
        b = bytearray()
        in_header = True
        for chunk in self.client.tts(sentence, self.options):
            # skip the RIFF header.
            if in_header:
                b.extend(chunk)
                if len(b) <= 36:
                    continue
                else:
                    fh = io.BytesIO(b)
                    fh.seek(36)
                    (data, size) = struct.unpack('<4sI', fh.read(8))
                    self.logger.info(f"first attempt: data: {data}, size: {hex(size)}, position: {fh.tell()}")
                    while data != b'data':
                        fh.read(size)
                        (data, size) = struct.unpack('<4sI', fh.read(8))
                        self.logger.info(f"subsequent data: {data}, size: {hex(size)}, position: {fh.tell()}, data != data: {data != b'data'}")
                    self.logger.info("position: ", fh.tell())
                    in_header = False
            else:
                if len(chunk):
                    yield chunk

