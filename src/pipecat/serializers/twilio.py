import base64
import json

from pipecat.frames.frames import AudioRawFrame, Frame
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.utils.audio import ulaw_8000_to_pcm_16000, pcm_16000_to_ulaw_8000


class TwilioFrameSerializer(FrameSerializer):
    SERIALIZABLE_TYPES = {
        AudioRawFrame: "audio",
    }

    def __init__(self):
        self.sid = None


    def serialize(self, frame: AudioRawFrame) -> dict:
        data = frame.audio

        serialized_data = pcm_16000_to_ulaw_8000(data)
        payload = base64.b64encode(serialized_data).decode('utf-8')
        answer_dict = {"event": "media",
                       "streamSid": self.sid,
                       "media": {"payload": payload}}

        return answer_dict

    def deserialize(self, message: bytes) -> AudioRawFrame | None:
        data = json.loads(message)
        if not self.sid:
            self.sid = data['streamSid'] if data.get("streamSid") else None

        if data['event'] != 'media':
            return None
        else:
            payload_base64 = data['media']['payload']
            payload = base64.b64decode(payload_base64)

            deserialized_data = ulaw_8000_to_pcm_16000(payload)
            audio_frame = AudioRawFrame(audio=deserialized_data, num_channels=1, sample_rate=16000)
            return audio_frame
