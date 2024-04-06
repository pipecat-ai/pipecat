from abc import abstractmethod
import asyncio
import logging
import time

from dailyai.pipeline.frame_processor import FrameProcessor
from dailyai.pipeline.pipeline import Pipeline


class AbstractTransport:
    def __init__(self, **kwargs):
        self.send_queue = asyncio.Queue()
        self.receive_queue = asyncio.Queue()
        self.completed_queue = asyncio.Queue()

        duration_minutes = kwargs.get("duration_minutes") or 10
        self._expiration = time.time() + duration_minutes * 60

        self._mic_enabled = kwargs.get("mic_enabled") or False
        self._mic_sample_rate = kwargs.get("mic_sample_rate") or 16000
        self._camera_enabled = kwargs.get("camera_enabled") or False
        self._camera_width = kwargs.get("camera_width") or 1024
        self._camera_height = kwargs.get("camera_height") or 768
        self._camera_bitrate = kwargs.get("camera_bitrate") or 250000
        self._camera_framerate = kwargs.get("camera_framerate") or 10
        self._speaker_enabled = kwargs.get("speaker_enabled") or False
        self._speaker_sample_rate = kwargs.get("speaker_sample_rate") or 16000

        self._logger: logging.Logger = logging.getLogger("dailyai.transport")

    @abstractmethod
    async def run(self, pipeline: Pipeline, override_pipeline_source_queue=True):
        pass

    @abstractmethod
    async def run_interruptible_pipeline(
        self,
        pipeline: Pipeline,
        pre_processor: FrameProcessor | None = None,
        post_processor: FrameProcessor | None = None,
    ):
        pass
