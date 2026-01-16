from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    InputAudioRawFrame,
    StartFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer


class RawPCM16Serializer(FrameSerializer):
    """Serializer for raw PCM16 audio data from browser WebSockets.
    
    Converts raw binary PCM16 audio bytes to InputAudioRawFrame objects
    for processing by the Pipecat pipeline.
    """
    
    def __init__(self, sample_rate: int = 16000, num_channels: int = 1):
        """Initialize the raw audio serializer.
        
        Args:
            sample_rate: Audio sample rate in Hz (default: 16000).
            num_channels: Number of audio channels (default: 1 for mono).
        """
        self._sample_rate = sample_rate
        self._num_channels = num_channels
    
    async def setup(self, frame: StartFrame):
        """Initialize with pipeline configuration.
        
        Args:
            frame: StartFrame containing pipeline configuration.
        """
        # Use pipeline sample rate if available
        if frame.audio_in_sample_rate:
            self._sample_rate = frame.audio_in_sample_rate
    
    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serialize a frame for output to the WebSocket.
        
        For audio output, returns raw PCM16 bytes that the browser can play.
        
        Args:
            frame: The frame to serialize.
            
        Returns:
            Raw audio bytes for AudioRawFrame, None for other frames.
        """
        if isinstance(frame, AudioRawFrame):
            # Return raw audio bytes - browser will handle PCM16 directly
            return bytes(frame.audio)
        return None
    
    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserialize raw WebSocket data to a Pipecat frame."""
        if isinstance(data, bytes) and len(data) > 0:
            return InputAudioRawFrame(
                audio=data,
                sample_rate=self._sample_rate,
                num_channels=self._num_channels,
            )
        return None
