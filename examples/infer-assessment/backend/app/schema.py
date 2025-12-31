from sqlmodel import SQLModel, Field, Column, JSON
from typing import List, Dict, Optional
from datetime import datetime
from uuid import UUID, uuid4

from app.models import TranscriptTurn, FreezeEvent
import json
from sqlmodel import TypeDecorator

class PydanticJSONType(TypeDecorator):
    impl = JSON
    
    def __init__(self, pydantic_model):
        super().__init__()
        self.pydantic_model = pydantic_model

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        # Convert list of Pydantic models to list of dicts
        return [item.model_dump() if hasattr(item, "model_dump") else item for item in value]

    def process_result_value(self, value, dialect):
        if value is None:
            return []
        # Convert list of dicts to list of Pydantic models
        return [self.pydantic_model.model_validate(item) for item in value]

class Session(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    transcript: List[TranscriptTurn] = Field(default=[], sa_column=Column(PydanticJSONType(TranscriptTurn)))
    freeze_events: List[FreezeEvent] = Field(default=[], sa_column=Column(PydanticJSONType(FreezeEvent)))
    latency_metrics: Dict[str, float] = Field(default={}, sa_column=Column(JSON))
    audio_url: Optional[str] = None
