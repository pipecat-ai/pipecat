#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


"""This module implements Tavus as a sink transport layer"""

import aiohttp
import base64

from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TransportMessageUrgentFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    StartInterruptionFrame,
    EndFrame,
    CancelFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import AIService
from pipecat.audio.utils import resample_audio

from loguru import logger


class TavusVideoService(AIService):
    """Class to send base64 encoded audio to Tavus"""

    def __init__(
        self,
        *,
        api_key: str,
        replica_id: str,
        persona_id: str = "pipecat0",
        session: aiohttp.ClientSession,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._api_key = api_key
        self._replica_id = replica_id
        self._persona_id = persona_id
        self._session = session

        self._conversation_id: str

    async def initialize(self) -> str:
        url = "https://tavusapi.com/v2/conversations"
        headers = {"Content-Type": "application/json", "x-api-key": self._api_key}
        payload = {
            "replica_id": self._replica_id,
            "persona_id": self._persona_id,
        }
        async with self._session.post(url, headers=headers, json=payload) as r:
            r.raise_for_status()
            response_json = await r.json()

        logger.debug(f"TavusVideoService joined {response_json['conversation_url']}")
        self._conversation_id = response_json["conversation_id"]
        return response_json["conversation_url"]

    def can_generate_metrics(self) -> bool:
        return True

    async def get_persona_name(self) -> str:
        url = f"https://tavusapi.com/v2/personas/{self._persona_id}"
        headers = {"Content-Type": "application/json", "x-api-key": self._api_key}
        async with self._session.get(url, headers=headers) as r:
            r.raise_for_status()
            response_json = await r.json()

        logger.debug(f"TavusVideoService persona grabbed {response_json}")
        return response_json["persona_name"]

    async def _end_conversation(self) -> None:
        url = f"https://tavusapi.com/v2/conversations/{self._conversation_id}/end"
        headers = {"Content-Type": "application/json", "x-api-key": self._api_key}
        async with self._session.post(url, headers=headers) as r:
            r.raise_for_status()

    async def _encode_audio_and_send(
        self, audio: bytes, original_sample_rate: int, done: bool
    ) -> None:
        """Encodes audio to base64 and sends it to Tavus"""
        if not done:
            audio = resample_audio(audio, original_sample_rate, 16000)
        audio_base64 = base64.b64encode(audio).decode("utf-8")
        logger.trace(f"TavusVideoService sending {len(audio)} bytes")
        await self._send_audio_message(audio_base64, done=done)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSStartedFrame):
            await self.start_processing_metrics()
            await self.start_ttfb_metrics()
            self._current_idx_str = str(frame.id)
        elif isinstance(frame, TTSAudioRawFrame):
            await self._encode_audio_and_send(frame.audio, frame.sample_rate, done=False)
        elif isinstance(frame, TTSStoppedFrame):
            await self._encode_audio_and_send(b"\x00", 16000, done=True)
            await self.stop_ttfb_metrics()
            await self.stop_processing_metrics()
        elif isinstance(frame, StartInterruptionFrame):
            await self._send_interrupt_message()
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._end_conversation()
        else:
            await self.push_frame(frame, direction)

    async def _send_interrupt_message(self) -> None:
        transport_frame = TransportMessageUrgentFrame(
            message={
                "message_type": "conversation",
                "event_type": "conversation.interrupt",
                "conversation_id": self._conversation_id,
            }
        )
        await self.push_frame(transport_frame)

    async def _send_audio_message(self, audio_base64: str, done: bool) -> None:
        transport_frame = TransportMessageUrgentFrame(
            message={
                "message_type": "conversation",
                "event_type": "conversation.echo",
                "conversation_id": self._conversation_id,
                "properties": {
                    "modality": "audio",
                    "inference_id": self._current_idx_str,
                    "audio": audio_base64,
                    "done": done,
                },
            }
        )
        await self.push_frame(transport_frame)
