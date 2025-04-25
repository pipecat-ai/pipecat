#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""This module implements Tavus as a sink transport layer"""

import asyncio
import base64
from typing import Optional

import aiohttp
from loguru import logger

from pipecat.audio.utils import create_default_resampler
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    StartFrame,
    StartInterruptionFrame,
    TransportMessageUrgentFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_service import AIService


class TavusVideoService(AIService):
    """Class to send base64 encoded audio to Tavus"""

    def __init__(
        self,
        *,
        api_key: str,
        replica_id: str,
        persona_id: str = "pipecat0",  # Use `pipecat0` so that your TTS voice is used in place of the Tavus persona
        session: aiohttp.ClientSession,
        sample_rate: int = 16000,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._api_key = api_key
        self._replica_id = replica_id
        self._persona_id = persona_id
        self._session = session
        self._sample_rate = sample_rate

        self._conversation_id: str

        self._resampler = create_default_resampler()

        self._audio_buffer = bytearray()
        self._queue = asyncio.Queue()
        self._send_task: Optional[asyncio.Task] = None

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

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._create_send_task()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._end_conversation()
        await self._cancel_send_task()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._end_conversation()
        await self._cancel_send_task()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartInterruptionFrame):
            await self._handle_interruptions()
            await self.push_frame(frame, direction)
        elif isinstance(frame, TTSStartedFrame):
            await self.start_processing_metrics()
            await self.start_ttfb_metrics()
            self._current_idx_str = str(frame.id)
        elif isinstance(frame, TTSAudioRawFrame):
            await self._queue_audio(frame.audio, frame.sample_rate, done=False)
        elif isinstance(frame, TTSStoppedFrame):
            await self._queue_audio(b"\x00\x00", self._sample_rate, done=True)
            await self.stop_ttfb_metrics()
            await self.stop_processing_metrics()
        else:
            await self.push_frame(frame, direction)

    async def _handle_interruptions(self):
        await self._cancel_send_task()
        await self._create_send_task()
        await self._send_interrupt_message()

    async def _end_conversation(self):
        url = f"https://tavusapi.com/v2/conversations/{self._conversation_id}/end"
        headers = {"Content-Type": "application/json", "x-api-key": self._api_key}
        async with self._session.post(url, headers=headers) as r:
            r.raise_for_status()

    async def _queue_audio(self, audio: bytes, in_rate: int, done: bool):
        await self._queue.put((audio, in_rate, done))

    async def _create_send_task(self):
        if not self._send_task:
            self._queue = asyncio.Queue()
            self._send_task = self.create_task(self._send_task_handler())

    async def _cancel_send_task(self):
        if self._send_task:
            await self.cancel_task(self._send_task)
            self._send_task = None

    async def _send_task_handler(self):
        # Daily app-messages have a 4kb limit and also a rate limit of 20
        # messages per second. Below, we only consider the rate limit because 1
        # second of a 24000 sample rate would be 48000 bytes (16-bit samples and
        # 1 channel). So, that is 48000 / 20 = 2400, which is below the 4kb
        # limit (even including base64 encoding). For a sample rate of 16000,
        # that would be 32000 / 20 = 1600.
        MAX_CHUNK_SIZE = int((self._sample_rate * 2) / 20)
        SLEEP_TIME = 1 / 20

        audio_buffer = bytearray()
        while True:
            (audio, in_rate, done) = await self._queue.get()

            if done:
                # Send any remaining audio.
                if len(audio_buffer) > 0:
                    await self._encode_audio_and_send(bytes(audio_buffer), done)
                await self._encode_audio_and_send(audio, done)
                audio_buffer.clear()
            else:
                audio = await self._resampler.resample(audio, in_rate, self._sample_rate)
                audio_buffer.extend(audio)
                while len(audio_buffer) >= MAX_CHUNK_SIZE:
                    chunk = audio_buffer[:MAX_CHUNK_SIZE]
                    audio_buffer = audio_buffer[MAX_CHUNK_SIZE:]
                    await self._encode_audio_and_send(bytes(chunk), done)
                    await asyncio.sleep(SLEEP_TIME)

    async def _encode_audio_and_send(self, audio: bytes, done: bool):
        """Encodes audio to base64 and sends it to Tavus"""
        audio_base64 = base64.b64encode(audio).decode("utf-8")
        logger.trace(f"{self}: sending {len(audio)} bytes")
        await self._send_audio_message(audio_base64, done=done)

    async def _send_interrupt_message(self) -> None:
        transport_frame = TransportMessageUrgentFrame(
            message={
                "message_type": "conversation",
                "event_type": "conversation.interrupt",
                "conversation_id": self._conversation_id,
            }
        )
        await self.push_frame(transport_frame)

    async def _send_audio_message(self, audio_base64: str, done: bool):
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
                    "sample_rate": self._sample_rate,
                },
            }
        )
        await self.push_frame(transport_frame)
