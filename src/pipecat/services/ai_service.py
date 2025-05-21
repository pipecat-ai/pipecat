#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Any, AsyncGenerator, Dict, Mapping

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
)
from pipecat.metrics.metrics import MetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class AIService(FrameProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_name: str = ""
        self._settings: Dict[str, Any] = {}
        self._session_properties: Dict[str, Any] = {}

    @property
    def model_name(self) -> str:
        return self._model_name

    def set_model_name(self, model: str):
        self._model_name = model
        self.set_core_metrics_data(MetricsData(processor=self.name, model=self._model_name))

    async def start(self, frame: StartFrame):
        pass

    async def stop(self, frame: EndFrame):
        pass

    async def cancel(self, frame: CancelFrame):
        pass

    async def _update_settings(self, settings: Mapping[str, Any]):
        from pipecat.services.openai_realtime_beta.events import (
            SessionProperties,
        )

        for key, value in settings.items():
            logger.debug("Update request for:", key, value)

            if key in self._settings:
                logger.info(f"Updating LLM setting {key} to: [{value}]")
                self._settings[key] = value
            elif key in SessionProperties.model_fields:
                logger.debug("Attempting to update", key, value)

                try:
                    from pipecat.services.openai_realtime_beta.events import (
                        TurnDetection,
                    )

                    if isinstance(self._session_properties, SessionProperties):
                        current_properties = self._session_properties
                    else:
                        current_properties = SessionProperties(**self._session_properties)

                    if key == "turn_detection" and isinstance(value, dict):
                        turn_detection = TurnDetection(**value)
                        setattr(current_properties, key, turn_detection)
                    else:
                        setattr(current_properties, key, value)

                    validated_properties = SessionProperties.model_validate(
                        current_properties.model_dump()
                    )
                    logger.info(f"Updating LLM setting {key} to: [{value}]")
                    self._session_properties = validated_properties.model_dump()
                except Exception as e:
                    logger.warning(f"Unexpected error updating session property {key}: {e}")
            elif key == "model":
                logger.info(f"Updating LLM setting {key} to: [{value}]")
                self.set_model_name(value)
            else:
                logger.warning(f"Unknown setting for {self.name} service: {key}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.start(frame)
        elif isinstance(frame, CancelFrame):
            await self.cancel(frame)
        elif isinstance(frame, EndFrame):
            await self.stop(frame)

    async def process_generator(self, generator: AsyncGenerator[Frame | None, None]):
        async for f in generator:
            if f:
                if isinstance(f, ErrorFrame):
                    await self.push_error(f)
                else:
                    await self.push_frame(f)
