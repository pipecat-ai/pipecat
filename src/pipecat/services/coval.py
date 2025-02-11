#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import io
import os
import uuid
import wave
from datetime import datetime
from typing import Optional

import aiohttp
from loguru import logger

from pipecat.frames.frames import CancelFrame, EndFrame, Frame
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import AIService


class CovalObservabilityService(AIService):
    """Initialize a CovalObservabilityService instance.

    This class uses an AudioBufferProcessor to get the conversation audio and
    uploads it to Coval API for audio processing and metrics.

    Args:
        call_id (str): Your unique identifier for the call.
        api_key (str): Your Coval API key.
        api_url (str, optional): Coval API endpoint. Defaults to "https://api.coval.dev/audio".
        output_dir (str, optional): Directory to save temporary audio files. Defaults to "recordings".

    Attributes:
        call_id (str): Stores the unique call identifier.
        output_dir (str): Directory path for saving temporary audio files.
    """

    def __init__(
        self,
        *,
        aiohttp_session: aiohttp.ClientSession,
        audio_buffer_processor: AudioBufferProcessor,
        call_id: str,
        api_key: str,
        api_url: str = "https://api.coval.dev/audio",
        output_dir: str = "recordings",
        context: Optional[OpenAILLMContext] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._aiohttp_session = aiohttp_session
        self._audio_buffer_processor = audio_buffer_processor
        self._api_key = api_key
        self._api_url = api_url
        self._call_id = call_id
        self._output_dir = output_dir
        self._context = context

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._process_audio()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._process_audio()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

    def _get_output_filename(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self._output_dir}/{timestamp}-{uuid.uuid4().hex}.wav"

    async def _process_audio(self):
        audio_buffer_processor = self._audio_buffer_processor

        if not audio_buffer_processor.has_audio():
            return

        os.makedirs(self._output_dir, exist_ok=True)
        filename = self._get_output_filename()
        audio = audio_buffer_processor.merge_audio_buffers()

        # Create WAV file in memory
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wf:
                wf.setsampwidth(2)
                wf.setnchannels(audio_buffer_processor.num_channels)
                wf.setframerate(audio_buffer_processor.sample_rate)
                wf.writeframes(audio)
            audio_data = buffer.getvalue()

        # Prepare request payload
        payload = {
            "metrics": {
                "metric_type_wpm": {},
                "metric_type_freq300": {}
            },
            "audio": base64.b64encode(audio_data).decode('utf-8'),
            "transcript": self._context.messages if self._context else []
        }

        # Send request to Coval API
        try:
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self._api_key
            }
            
            response = await self._aiohttp_session.post(
                self._api_url,
                headers=headers,
                json=payload
            )
            
            if not response.ok:
                logger.error(f"Failed to upload to Coval: {await response.text()}")
            else:
                logger.debug(f"Successfully uploaded audio for call {self._call_id} to Coval")
                
        except Exception as e:
            logger.error(f"Failed to upload to Coval: {str(e)}")
