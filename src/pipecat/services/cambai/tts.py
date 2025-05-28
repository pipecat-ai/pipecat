#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import base64
import json
import uuid
import warnings
from typing import AsyncGenerator, List, Optional, Union

import aiohttp

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

# See .env.example for Camb.AI configuration needed

class CambAIClient:
    def __init__(self, api_key: str, base_url: str):
        self._api_key = api_key
        self._base_url = base_url
        self._session = None

    async def _get_session(self):
        if self._session is None or self._session.closed:
            # Set timeout for all requests
            timeout = aiohttp.ClientTimeout(total=60, connect=10)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

class CambAITTSService(TTSService):
    class InputParams(BaseModel):
        language: Optional[int] = 1
        gender: Optional[int] = 1
        age: Optional[int] = 0

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: int,
        base_url: str = "https://client.camb.ai/apis",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        params = params or CambAITTSService.InputParams()
        self._api_key = api_key
        self._voice_id = str(voice_id)
        self._base_url = base_url
        self._settings = {
            "language": params.language,
            "gender": params.gender,
            "age": params.age,
            "output_format": {"sample_rate": sample_rate or 24000}
        }
        self._client = CambAIClient(api_key, base_url)

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._settings["output_format"]["sample_rate"] = self.sample_rate

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._client.close()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._client.close()

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")
        try:
            await self.start_ttfb_metrics()
            payload = {
                "text": text,
                "voice_id": self._voice_id,
                "sample_rate": self.sample_rate,
                "language": self._settings["language"],
                "gender": self._settings["gender"],
                "age": self._settings["age"],
            }
            yield TTSStartedFrame()
            session = await self._client._get_session()
            headers = {
                "Accept": "application/json",
                "x-api-key": self._api_key,
                "Content-Type": "application/json"
            }
            url = f"{self._base_url}/tts"

            # Step 1: Get Task_ID
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Camb AI API error: {error_text}")
                    await self.push_error(ErrorFrame(f"Camb AI API error: {error_text}"))
                    return
                response_data = await response.json()
                task_id = response_data.get("task_id")
                if not task_id:
                    logger.error("Camb AI API error: No task_id in response")
                    await self.push_error(ErrorFrame("Camb AI API error: No task_id in response"))
                    return
                logger.debug(f"Task ID: {task_id}")

            #Step 2: Poll Status for Run_ID
            run_id = None
            max_attempts = 30
            attempt = 0

            while attempt < max_attempts:
                async with session.get(f"{self._base_url}/tts/{task_id}", headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Camb AI API error: {error_text}")
                        await self.push_error(ErrorFrame(f"Camb AI API error: {error_text}"))
                        return
                    status_data = await response.json()
                    status = status_data.get("status")
                    if status == "SUCCESS":
                        run_id = status_data.get("run_id")
                        break
                    elif status == "FAILED":
                        logger.error(f"Camb AI API error: TTS task failed")
                        await self.push_error(ErrorFrame(f"Camb AI API error: TTS task failed"))
                        return
                    await asyncio.sleep(1)
                    attempt += 1
            if not run_id:
                logger.error(f"Camb AI API error: Timed out waiting for TTS task to complete")
                await self.push_error(ErrorFrame(f"Camb AI API error: Timed out waiting for TTS task to complete"))
                return

            # Step 3: Get Audio
            async with session.get(f"{self._base_url}/tts-result/{run_id}", headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Camb AI API error: {error_text}")
                    await self.push_error(ErrorFrame(f"Camb AI API error: {error_text}"))
                    return
                audio_data = await response.read()

            await self.start_tts_usage_metrics(text)
            frame = TTSAudioRawFrame(
                audio=audio_data,
                sample_rate=self.sample_rate,
                num_channels=1,
            )

            yield frame
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            await self.push_error(ErrorFrame(f"Error generating TTS: {e}"))
        finally:
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()
