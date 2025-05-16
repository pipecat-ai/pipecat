#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""This module implements Tavus as a sink transport layer"""

import asyncio
from functools import partial
from typing import Optional

import aiohttp
from daily.daily import VideoFrame
from loguru import logger

from pipecat.audio.utils import create_default_resampler
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    OutputImageRawFrame,
    StartFrame,
    StartInterruptionFrame,
    TransportMessageUrgentFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.ai_service import AIService
from pipecat.transports.services.daily import DailyCallbacks, DailyParams, DailyTransportClient


class TavusVideoService(AIService):
    """Class to send base64 encoded audio to Tavus"""

    def __init__(
        self,
        *,
        api_key: str,
        replica_id: str,
        # persona_id: str = "pipecat0", # The audio input is disabled in this case
        persona_id: str = "p1b65d438817",  # Using pipeline_mode "echo" and the Daily Transport.
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

        self._other_participant_has_joined = False
        self._client: Optional[DailyTransportClient] = None

        self._conversation_id: str

        self._resampler = create_default_resampler()

        self._audio_buffer = bytearray()
        self._queue = asyncio.Queue()
        self._send_task: Optional[asyncio.Task] = None

    async def setup(self, setup: FrameProcessorSetup):
        await super().setup(setup)
        try:
            room_url = await self.initialize()
            daily_params = DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_out_sample_rate=16000,  # TODO: we should probably fix this
                video_in_enabled=True,
            )
            callbacks = DailyCallbacks(
                on_active_speaker_changed=partial(
                    self._on_handle_callback, "on_active_speaker_changed"
                ),
                on_joined=self._on_joined,
                on_left=self._on_left,
                on_error=partial(self._on_handle_callback, "on_error"),
                on_app_message=partial(self._on_handle_callback, "on_app_message"),
                on_call_state_updated=partial(self._on_handle_callback, "on_call_state_updated"),
                on_client_connected=partial(self._on_handle_callback, "on_client_connected"),
                on_client_disconnected=partial(self._on_handle_callback, "on_client_disconnected"),
                on_dialin_connected=partial(self._on_handle_callback, "on_dialin_connected"),
                on_dialin_ready=partial(self._on_handle_callback, "on_dialin_ready"),
                on_dialin_stopped=partial(self._on_handle_callback, "on_dialin_stopped"),
                on_dialin_error=partial(self._on_handle_callback, "on_dialin_error"),
                on_dialin_warning=partial(self._on_handle_callback, "on_dialin_warning"),
                on_dialout_answered=partial(self._on_handle_callback, "on_dialout_answered"),
                on_dialout_connected=partial(self._on_handle_callback, "on_dialout_connected"),
                on_dialout_stopped=partial(self._on_handle_callback, "on_dialout_stopped"),
                on_dialout_error=partial(self._on_handle_callback, "on_dialout_error"),
                on_dialout_warning=partial(self._on_handle_callback, "on_dialout_warning"),
                on_participant_joined=self._on_participant_joined,
                on_participant_left=partial(self._on_handle_callback, "on_participant_left"),
                on_participant_updated=partial(self._on_handle_callback, "on_participant_updated"),
                on_transcription_message=partial(
                    self._on_handle_callback, "on_transcription_message"
                ),
                on_recording_started=partial(self._on_handle_callback, "on_recording_started"),
                on_recording_stopped=partial(self._on_handle_callback, "on_recording_stopped"),
                on_recording_error=partial(self._on_handle_callback, "on_recording_error"),
            )
            self._client = DailyTransportClient(
                room_url, None, "Pipecat", daily_params, callbacks, self.name
            )
            await self._client.setup(setup)
            await self._client.join()
        except Exception as e:
            logger.error(f"Failed to setup Tavus: {e}")
            await self._end_conversation()

    async def cleanup(self):
        await super().cleanup()
        await self._client.cleanup()
        self._client = None

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

    async def _on_joined(self, data):
        logger.debug(f"TavusVideoService Pipecat client joined!")

    async def _on_left(self):
        logger.debug(f"TavusVideoService Pipecat client left!")

    async def _on_handle_callback(self, event_name, *args, **kwargs):
        logger.trace(f"[Callback] {event_name} called with args={args}, kwargs={kwargs}")

    async def _on_participant_joined(self, participant):
        participant_id = participant["id"]
        logger.info(f"Participant joined {participant_id}")
        if not self._other_participant_has_joined:
            self._other_participant_has_joined = True
            await self._client.capture_participant_video(
                participant_id, self._on_participant_video_frame, 30
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
        await self._client.start(frame)
        await self._create_send_task()
        # TODO: implement create receive task

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
        elif isinstance(frame, TTSAudioRawFrame):
            await self._queue_audio(frame.audio, frame.sample_rate, done=False)
            # TODO: need to check if we should push this audio, or use the one received from Tavus
            await self.push_frame(frame, direction)
        elif isinstance(frame, TTSStoppedFrame):
            # TODO: double check if we need to change this silence somehow
            # await self._queue_audio(b"\x00\x00", self._sample_rate, done=True)
            await self.stop_ttfb_metrics()
            await self.stop_processing_metrics()
        else:
            await self.push_frame(frame, direction)

    # TODO check if we are still going to need this
    async def _handle_interruptions(self):
        #await self._cancel_send_task()
        #await self._create_send_task()
        await self._send_interrupt_message()

    async def _end_conversation(self):
        await self._client.leave()
        self._other_participant_has_joined = False
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
        while True:
            (audio, in_rate, done) = await self._queue.get()
            await self._client.write_raw_audio_frames(audio)

    async def _send_interrupt_message(self) -> None:
        transport_frame = TransportMessageUrgentFrame(
            message={
                "message_type": "conversation",
                "event_type": "conversation.interrupt",
                "conversation_id": self._conversation_id,
            }
        )
        await self.push_frame(transport_frame)
