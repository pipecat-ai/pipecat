#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


"""This module implements Tavus as a sink transport layer"""

import base64
import aiohttp

from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TransportMessageFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from loguru import logger

MIN_AUDIO_BUFFER_SIZE = int(16000 * 2 * 0.5)  # 0.5 seconds


class TavusVideoService(FrameProcessor):
    """Class to send base64 encoded audio to Tavus"""

    def __init__(
        self,
        *,
        conversation_id: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._conversation_id = conversation_id
        self._tavus_audio_buffer = b""

    def can_generate_metrics(self) -> bool:
        return True

    @classmethod
    async def get_persona_name(
        cls,
        *,
        session: aiohttp.ClientSession,
        api_key: str,
        persona_id: str,
    ) -> str:
        url = f"https://tavusapi.com/v2/personas/{persona_id}"
        headers = {"Content-Type": "application/json", "x-api-key": api_key}
        async with session.get(url, headers=headers) as r:
            r.raise_for_status()
            response_json = await r.json()

        logger.debug(f"TavusVideoService persona grabbed {response_json}")
        return response_json["persona_name"]

    @classmethod
    async def initiate_conversation(
        cls,
        *,
        session: aiohttp.ClientSession,
        api_key: str,
        replica_id: str,
        persona_id: str = "pipecat0",
        custom_greeting: str = {},
    ) -> tuple[str, str]:
        url = "https://tavusapi.com/v2/conversations"
        headers = {"Content-Type": "application/json", "x-api-key": api_key}
        payload = {
            "replica_id": replica_id,
            "persona_id": persona_id,
            "custom_greeting": custom_greeting,
        }
        async with session.post(url, headers=headers, json=payload) as r:
            r.raise_for_status()
            response_json = await r.json()

        logger.debug(f"TavusVideoService joined {response_json['conversation_url']}")
        return response_json["conversation_url"], response_json["conversation_id"]

    async def _encode_audio_and_send(self, audio: bytes) -> None:
        """Encodes audio to base64 and sends it to Tavus"""
        self._tavus_audio_buffer += audio
        if len(self._tavus_audio_buffer) >= MIN_AUDIO_BUFFER_SIZE:
            await self._flush_audio_buffer()

    async def _flush_audio_buffer(self) -> None:
        """Flushes the audio buffer"""
        if len(self._tavus_audio_buffer) > 0:
            audio_base64 = base64.b64encode(self._tavus_audio_buffer).decode("utf-8")
            logger.debug(f"TavusVideoService sending {len(audio_base64)} bytes")
            await self._send_audio_message(audio_base64)
            self._tavus_audio_buffer = b""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSStartedFrame):
            await self.start_processing_metrics()
            await self.start_ttfb_metrics()
            self._current_idx_str = str(frame.id)
        elif isinstance(frame, TTSAudioRawFrame):
            await self._encode_audio_and_send(frame.audio)
        elif isinstance(frame, TTSStoppedFrame):
            await self._flush_audio_buffer()
            await self.stop_ttfb_metrics()
            await self.stop_processing_metrics()
        else:
            await self.push_frame(frame, direction)

    async def _send_audio_message(self, audio_base64: str) -> None:
        transport_frame = TransportMessageFrame(
            message={
                "message_type": "conversation",
                "event_type": "conversation.echo",
                "conversation_id": self._conversation_id,
                "properties": {
                    "type": "audio",
                    "inference_id": self._current_idx_str,
                    "audio": audio_base64,
                },
            }
        )
        await self.push_frame(transport_frame)
