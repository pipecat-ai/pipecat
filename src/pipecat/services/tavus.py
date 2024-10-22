#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""This module implements Whisper transcription with a locally-downloaded model."""

import asyncio
from typing import Any
import base64
import requests

from enum import Enum
from typing import AsyncGenerator

import numpy as np

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame, TTSAudioRawFrame, TransportMessageUrgentFrame, TTSStartedFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.ai_services import AIService
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.services.daily import DailyTransportClient, DailyParams
from pipecat.utils.time import time_now_iso8601

from loguru import logger

class TavusVideoService(BaseOutputTransport):
    """Class to send base64 encoded audio to Tavus"""

    def __init__(
        self,
        *,
        conversation_id: str,
        client: DailyTransportClient,
        params: DailyParams,
        **kwargs,
    ) -> None:
        super().__init__(params, **kwargs)
        self._client = client
        self._conversation_id = conversation_id
        self._tavus_audio_buffer = b""


    def can_generate_metrics(self) -> bool:
        return True

    @classmethod
    def _initiate_conversation(
        cls,
        api_key: str,
        replica_id: str,
        custom_greeting: str = {}
    ) -> tuple[str, str]:
        url = "https://tavusapi.com/v2/conversations"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key
        }
        payload = {
            "replica_id": replica_id,
            "persona_id": "pipecat0",
            "custom_greeting": custom_greeting,
        }

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        response_json = response.json()
        logger.debug(f"TavusVideoService joined {response_json['conversation_url']}")
        return response_json["conversation_url"], response_json["conversation_id"]

    async def encode_audio_and_send(self, audio: bytes) -> None:
        """Encodes audio to base64 and sends it to Tavus"""
        await self.start_processing_metrics()
        await self.start_ttfb_metrics()
        self._tavus_audio_buffer = self._tavus_audio_buffer + audio
        if len(self._tavus_audio_buffer) > 72000:
            print(f"sending audio")
            audio_base64 = base64.b64encode(self._tavus_audio_buffer[:72000]).decode("utf-8")
            await self.send_audio_message(audio_base64)
            self._tavus_audio_buffer = self._tavus_audio_buffer[72000:]
        else:
            print(f"accumulating audio buffer: {len(self._tavus_audio_buffer)}")
        
        await self.stop_ttfb_metrics()
        await self.stop_processing_metrics()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSStartedFrame):
            self._current_idx_str = str(frame.id)
        elif isinstance(frame, TTSAudioRawFrame):
            await self.encode_audio_and_send(frame.audio)
        else:
            await self.push_frame(frame, direction)

    async def send_audio_message(self, audio_base64: str) -> None:
        message = TransportMessageUrgentFrame(
            message={
                "message_type": "conversation",
                "event_type": "conversation.echo",
                "conversation_id": self._conversation_id,
                "properties": {
                    "type": "audio",
                    "inference_id": self._current_idx_str,
                    "audio": audio_base64
                }
            }
        )
        await self._client.send_message(message)
