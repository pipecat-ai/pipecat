import io
import struct

from dailyai.services.ai_services import TTSService

try:
    from pyht import Client
    from pyht.client import TTSOptions
    from pyht.protos.api_pb2 import Format
except ModuleNotFoundError as e:
    print(f"Exception: {e}")
    print(
        "In order to use PlayHT, you need to `pip install dailyai[playht]`. Also, set `PLAY_HT_USER_ID` and `PLAY_HT_API_KEY` environment variables.")
    raise Exception(f"Missing module: {e}")


class PlayHTAIService(TTSService):

    def __init__(
        self,
        *,
        api_key,
        user_id,
        voice_url
    ):
        super().__init__()

        self.speech_key = api_key
        self.user_id = user_id

        self.client = Client(
            user_id=self.user_id,
            api_key=self.speech_key,
        )
        self.options = TTSOptions(
            voice=voice_url,
            sample_rate=16000,
            quality="higher",
            format=Format.FORMAT_WAV)

    def __del__(self):
        self.client.close()

    async def run_tts(self, sentence):
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
                    self.logger.info(
                        f"first attempt: data: {data}, size: {hex(size)}, position: {fh.tell()}")
                    while data != b'data':
                        fh.read(size)
                        (data, size) = struct.unpack('<4sI', fh.read(8))
                        self.logger.info(
                            f"subsequent data: {data}, size: {hex(size)}, position: {fh.tell()}, data != data: {data != b'data'}")
                    self.logger.info("position: ", fh.tell())
                    in_header = False
            else:
                if len(chunk):
                    yield chunk
