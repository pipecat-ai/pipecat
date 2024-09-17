from typing import Optional
from pydantic import BaseModel


class MetricsParams(BaseModel):
    pass


class TTFBMetricsParams(MetricsParams):
    processor: str
    value: float
    model: Optional[str]


class ProcessingMetricsParams(MetricsParams):
    processor: str
    value: float
    model: Optional[str]


class LLMUsageMetricsParams(MetricsParams):
    processor: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CacheUsageMetricsParams(LLMUsageMetricsParams):
    cache_read_input_tokens: int
    cache_creation_input_tokens: int


class TTSUsageMetricsParams(MetricsParams):
    processor: str
    model: Optional[str]
    value: int
