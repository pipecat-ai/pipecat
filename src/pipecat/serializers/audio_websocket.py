from pipecat.serializers.base_serializer import FrameSerializer, FrameSerializerType
from pipecat.frames.frames import Frame, InputAudioRawFrame, StartFrame

class AudioWebsocketSerializer(FrameSerializer):
    def __init__(self):
        self.sample_rate = 16000
        self.num_channels = 1

    @property
    def type(self) -> FrameSerializerType:
        return FrameSerializerType.BINARY
    
    async def setup(self, frame: StartFrame):
        self.sample_rate = frame.audio_in_sample_rate

    async def serialize(self, frame: Frame) -> bytes:
        return frame.audio if isinstance(frame, InputAudioRawFrame) else None

    async def deserialize(self, data: bytes) -> Frame:
        return InputAudioRawFrame(audio=data, sample_rate=self.sample_rate, num_channels=self.num_channels)