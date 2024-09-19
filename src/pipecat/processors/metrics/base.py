import time
from loguru import logger
from pipecat.frames.frames import MetricsFrame

class FrameProcessorMetrics:
    def __init__(self, name: str):
        self._name = name
        self._start_ttfb_time = 0
        self._start_processing_time = 0
        self._should_report_ttfb = True

    async def start_ttfb_metrics(self, report_only_initial_ttfb):
        if self._should_report_ttfb:
            self._start_ttfb_time = time.time()
            self._should_report_ttfb = not report_only_initial_ttfb

    async def stop_ttfb_metrics(self):
        if self._start_ttfb_time == 0:
            return None

        value = time.time() - self._start_ttfb_time
        logger.debug(f"{self._name} TTFB: {value}")
        ttfb = {
            "processor": self._name,
            "value": value
        }
        self._start_ttfb_time = 0
        return MetricsFrame(ttfb=[ttfb])

    async def start_processing_metrics(self):
        self._start_processing_time = time.time()

    async def stop_processing_metrics(self):
        if self._start_processing_time == 0:
            return None

        value = time.time() - self._start_processing_time
        logger.debug(f"{self._name} processing time: {value}")
        processing = {
            "processor": self._name,
            "value": value
        }
        self._start_processing_time = 0
        return MetricsFrame(processing=[processing])

    async def start_llm_usage_metrics(self, tokens: dict):
        logger.debug(
            f"{self._name} prompt tokens: {tokens['prompt_tokens']}, completion tokens: {tokens['completion_tokens']}")
        return MetricsFrame(tokens=[tokens])

    async def start_tts_usage_metrics(self, text: str):
        characters = {
            "processor": self._name,
            "value": len(text),
        }
        logger.debug(f"{self._name} usage characters: {characters['value']}")
        return MetricsFrame(characters=[characters])