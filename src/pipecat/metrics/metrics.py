from typing import Optional
from pydantic import BaseModel


class MetricsData(BaseModel):
    processor: str


class TTFBMetricsData(MetricsData):
    value: float
    model: Optional[str]


class ProcessingMetricsData(MetricsData):
    value: float
    model: Optional[str]


class LLMUsageMetricsData(MetricsData):
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CacheUsageMetricsData(LLMUsageMetricsData):
    cache_read_input_tokens: int
    cache_creation_input_tokens: int


class TTSUsageMetricsData(MetricsData):
    processor: str
    model: Optional[str]
    value: int
