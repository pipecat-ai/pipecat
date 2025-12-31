from pydantic import BaseModel

class TranscriptTurn(BaseModel):
    role: str
    content: str
    timestamp: float
    latency: float

class FreezeEvent(BaseModel):
    start_time: float
    end_time: float
    duration: float