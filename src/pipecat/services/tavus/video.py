#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""This module implements Tavus as a sink transport layer"""

import asyncio
from typing import Optional

import aiohttp
from daily.daily import AudioData, VideoFrame
from loguru import logger

from pipecat.audio.utils import create_default_resampler
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    StartFrame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.ai_service import AIService
from pipecat.transports.services.tavus import TavusCallbacks, TavusParams, TavusTransportClient
from pipecat.utils.asyncio.watchdog_queue import WatchdogQueue


class TavusVideoService(AIService):
    """
    Service class that proxies audio to Tavus and receives both audio and video in return.

    It uses the `TavusTransportClient` to manage the session and handle communication. When
    audio is sent, Tavus responds with both audio and video streams, which are then routed
    through Pipecat’s media pipeline.

    In use cases such as with `DailyTransport`, this results in two distinct virtual rooms:
        - **Tavus room**: Contains the Tavus Avatar and the Pipecat Bot.
        - **User room**: Contains the Pipecat Bot and the user.

    Args:
        api_key (str): Tavus API key used for authentication.
        replica_id (str): ID of the Tavus voice replica to use for speech synthesis.
        persona_id (str): ID of the Tavus persona. Defaults to "pipecat-stream" to use the Pipecat TTS voice.
        session (aiohttp.ClientSession): Async HTTP session used for communication with Tavus.
        **kwargs: Additional arguments passed to the parent `AIService` class.
    """

    def __init__(
        self,
        *,
        api_key: str,
        replica_id: str,
        persona_id: str = "pipecat-stream",
        session: aiohttp.ClientSession,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._api_key = api_key
        self._session = session
        self._replica_id = replica_id
        self._persona_id = persona_id

        self._other_participant_has_joined = False
        self._client: Optional[TavusTransportClient] = None

        self._conversation_id: str
        self._resampler = create_default_resampler()

        self._audio_buffer = bytearray()
        self._send_task: Optional[asyncio.Task] = None
        # This is the custom track destination expected by Tavus
        self._transport_destination: Optional[str] = "stream"

    async def setup(self, setup: FrameProcessorSetup):
        await super().setup(setup)
        callbacks = TavusCallbacks(
            on_participant_joined=self._on_participant_joined,
            on_participant_left=self._on_participant_left,
        )
        self._client = TavusTransportClient(
            bot_name="Pipecat",
            callbacks=callbacks,
            api_key=self._api_key,
            replica_id=self._replica_id,
            persona_id=self._persona_id,
            session=self._session,
            params=TavusParams(
                audio_in_enabled=True,
                video_in_enabled=True,
                audio_out_enabled=True,
                microphone_out_enabled=False,
            ),
        )
        await self._client.setup(setup)

    async def cleanup(self):
        await super().cleanup()
        await self._client.cleanup()
        self._client = None

    async def _on_participant_left(self, participant, reason):
        participant_id = participant["id"]
        logger.info(f"Participant left {participant_id}, reason: {reason}")

    async def _on_participant_joined(self, participant):
        participant_id = participant["id"]
        logger.info(f"Participant joined {participant_id}")
        if not self._other_participant_has_joined:
            self._other_participant_has_joined = True
            await self._client.capture_participant_video(
                participant_id, self._on_participant_video_frame, 30
            )
            await self._client.capture_participant_audio(
                participant_id=participant_id,
                callback=self._on_participant_audio_data,
                sample_rate=self._client.out_sample_rate,
            )

    async def _on_participant_video_frame(
        self, participant_id: str, video_frame: VideoFrame, video_source: str
    ):
        frame = OutputImageRawFrame(
            image=video_frame.buffer,
            size=(video_frame.width, video_frame.height),
            format=video_frame.color_format,
        )
        frame.transport_source = video_source
        await self.push_frame(frame)

    async def _on_participant_audio_data(
        self, participant_id: str, audio: AudioData, audio_source: str
    ):
        frame = OutputAudioRawFrame(
            audio=audio.audio_frames,
            sample_rate=audio.sample_rate,
            num_channels=audio.num_channels,
        )
        frame.transport_source = audio_source
        await self.push_frame(frame)

    def can_generate_metrics(self) -> bool:
        return True

    async def get_persona_name(self) -> str:
        return await self._client.get_persona_name()

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._client.start(frame)
        if self._transport_destination:
            await self._client.register_audio_destination(self._transport_destination)
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
        elif isinstance(frame, TTSAudioRawFrame):
            await self._handle_audio_frame(frame)
        else:
            await self.push_frame(frame, direction)

    async def _handle_interruptions(self):
        await self._cancel_send_task()
        await self._create_send_task()
        await self._client.send_interrupt_message()

    async def _end_conversation(self):
        await self._client.stop()
        self._other_participant_has_joined = False

    async def _create_send_task(self):
        if not self._send_task:
            self._queue = WatchdogQueue(self.task_manager)
            self._send_task = self.create_task(self._send_task_handler())

    async def _cancel_send_task(self):
        if self._send_task:
            await self.cancel_task(self._send_task)
            self._send_task = None

    async def _handle_audio_frame(self, frame: OutputAudioRawFrame):
        sample_rate = self._client.out_sample_rate
        # 40 ms of audio
        chunk_size = int((sample_rate * 2) / 25)
        # We might need to resample if incoming audio doesn't match the
        # transport sample rate.
        resampled = await self._resampler.resample(frame.audio, frame.sample_rate, sample_rate)
        self._audio_buffer.extend(resampled)
        while len(self._audio_buffer) >= chunk_size:
            chunk = OutputAudioRawFrame(
                bytes(self._audio_buffer[:chunk_size]),
                sample_rate=sample_rate,
                num_channels=frame.num_channels,
            )
            chunk.transport_destination = self._transport_destination
            await self._queue.put(chunk)
            self._audio_buffer = self._audio_buffer[chunk_size:]

    async def _send_task_handler(self):
        while True:
            frame = await self._queue.get()
            if isinstance(frame, OutputAudioRawFrame) and self._client:
                await self._client.write_audio_frame(frame)
            self._queue.task_done()
