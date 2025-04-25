from typing import Optional

from pydantic import BaseModel


class MetricsData(BaseModel):
    processor: str
    model: Optional[str] = None


class TTFBMetricsData(MetricsData):
    value: float


class ProcessingMetricsData(MetricsData):
    value: float


class LLMTokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_read_input_tokens: Optional[int] = None
    cache_creation_input_tokens: Optional[int] = None


class LLMUsageMetricsData(MetricsData):
    value: LLMTokenUsage


class TTSUsageMetricsData(MetricsData):
    value: int


class SmartTurnMetricsData(MetricsData):
    """Metrics data for smart turn predictions."""

    is_complete: bool
    probability: float
    inference_time_ms: float
    server_total_time_ms: float
    e2e_processing_time_ms: float
