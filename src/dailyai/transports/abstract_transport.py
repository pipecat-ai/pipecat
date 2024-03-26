from abc import abstractmethod

from dailyai.pipeline.pipeline import Pipeline


class AbstractTransport:
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    async def run(self, pipeline: Pipeline, override_pipeline_source_queue=True):
        pass
