from pydantic import BaseModel
from typing import Dict, List, Optional, Literal


class InputAudioTranscription(BaseModel):
    model: Optional[str] = "whisper-1"


class TurnDetection(BaseModel):
    type: Optional[Literal["server_vad"]] = "server_vad"
    threshold: Optional[float] = 0.5
    prefix_padding_ms: Optional[int] = 300
    silence_duration_ms: Optional[int] = 800


class SessionProperties(BaseModel):
    modalities: Optional[List[Literal["text", "audio"]]] = ["text", "audio"]
    instructions: Optional[str] = None
    voice: Optional[str] = "alloy"
    input_audio_format: Optional[Literal["pcm16", "g711_ulaw", "g711_alaw"]] = "pcm16"
    output_audio_format: Optional[Literal["pcm16", "g711_ulaw", "g711_alaw"]] = "pcm16"
    input_audio_transcription: Optional[InputAudioTranscription] = InputAudioTranscription()
    turn_detection: Optional[TurnDetection] = TurnDetection()
    tools: Optional[List[Dict]] = []
    tool_choice: Optional[Literal["auto", "none", "required"]] = "auto"
    temperature: Optional[float] = 0.8
    max_response_output_tokens: Optional[int] = 4096


class SessionUpdateEvent(BaseModel):
    event_id: str
    type: Literal["session.update"]
    session: SessionProperties
