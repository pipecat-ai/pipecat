import json
from pipecat.serializers.twilio import TwilioFrameSerializer

class SmartfloFrameSerializer(TwilioFrameSerializer):
    """Serializer for Tata Smartflo Bi-Directional Audio Streaming WebSocket protocol.
    
    Smartflo's protocol is based on the Twilio Media Streams format with these differences:
    - Outbound media messages require an incrementing `chunk` field
    - Audio payloads should be multiples of 160 bytes to avoid audio gaps
    - Start event includes additional metadata (from, to, direction, bitRate, bitDepth)
    
    Reference: https://docs.smartflo.tatatelebusiness.com/docs/bi-directional-audio-streaming-integration-document
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._chunk = 0

    def _build_media_message(self, b64_payload: str) -> str:
        self._chunk += 1
        return json.dumps({
            "event": "media",
            "streamSid": self.stream_sid,
            "media": {
                "payload": b64_payload,
                "chunk": self._chunk
            }
        })