#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import io
import os
import wave
from asyncio import sleep

import aiohttp
from loguru import logger

from pipecat.frames.frames import CancelFrame, EndFrame, Frame
from pipecat.processors.audio import audio_buffer_processor
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import AIService

from twilio.rest import Client


class ConnexityInterface(AIService):
    def __init__(self,
                 call_id: str,
                 assistant_id: str,
                 api_key: str,
                 api_url: str,
                 assistant_speaks_first: bool = True,
                 **kwargs,
                 ):
        super().__init__(**kwargs)
        self._audio_buffer_processor = audio_buffer_processor
        self._audio_memory_buffer = io.BytesIO()
        self._api_key = api_key
        self._api_url = api_url
        self._call_id = call_id
        self._assistant_id = assistant_id
        self._assistant_speaks_first = assistant_speaks_first

    def _request_headers(self):
        return {"Content-Type": "application/json", "X-API-KEY": self._api_key}

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

    async def send_audio_url_to_connexity(self, audio_url):
        answer_list = {"items" : [{
            "agent_id": self._assistant_id,
            "sid": self._call_id,
            "first_speaker_role": "assistant" if self._assistant_speaks_first else "user",
            "audio_link": audio_url
        }]}
        print(answer_list, flush=True)
        print('----------------------------------------------', flush=True)
        async with aiohttp.ClientSession() as session:
            async with session.post(self._api_url, headers=self._request_headers(), json=answer_list) as response:
                # Optionally handle the response, for example:
                if response.status != 200:
                    print(f"Failed to send data: {response.status}")
                else:
                    print(f"Data sent successfully: {response.status}")


class ConnexityLocalMetricsService(ConnexityInterface):
    """Initialize a ConnexityLocalMetricsService instance.

    This class uses an AudioBufferProcessor to get the conversation audio and
    uploads it to Connexity Voice API for audio processing.

    Args:
        call_id (str): Your unique identifier for the call. This is used to match the call in the Connexity Voice system to the call in your system.
        assistant_id (str): Identifier for the AI assistant. This can be whatever you want, it's intended for you convenience so you can distinguish
        between different assistants and a grouping mechanism for calls.
        assistant_speaks_first (bool, optional): Indicates if the assistant speaks first in the conversation. Defaults to True.
        output_dir (str, optional): Directory to save temporary audio files. Defaults to "recordings".

    Attributes:
        call_id (str): Stores the unique call identifier.
        assistant_id (str): Stores the assistant identifier.
        assistant_speaks_first (bool): Indicates whether the assistant speaks first.
        output_dir (str): Directory path for saving temporary audio files.

    The constructor also ensures that the output directory exists.
    """

    def __init__(
        self,
        *,
        audio_buffer_processor: AudioBufferProcessor,
        call_id: str,
        assistant_id: str,
        api_key: str,
        api_url: str = "http://connexity-gateway-owzhcfagkq-uc.a.run.app/process/blackbox/file/pipecat",
        assistant_speaks_first: bool = True,
        **kwargs,
    ):
        super().__init__(call_id=call_id,
                         assistant_id=assistant_id,
                         api_key=api_key,
                         api_url=api_url,
                         assistant_speaks_first=assistant_speaks_first,
                         **kwargs)
        self._audio_buffer_processor = audio_buffer_processor
        self._audio_memory_buffer = io.BytesIO()
        self._api_url = api_url

    async def stop(self, frame: EndFrame):
        await self._process_audio()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        await self._process_audio()
        await super().cancel(frame)

    def _request_headers(self):
        return {"X-API-KEY": self._api_key}

    async def _process_audio(self):
        print('PROCESS AUDIO', flush=True)
        audio_buffer_processor = self._audio_buffer_processor

        if not audio_buffer_processor.has_audio():
            return

        # Merge the raw PCM data from the buffer processor
        new_audio = audio_buffer_processor.merge_audio_buffers()

        try:
            # Build a temporary WAV buffer from the new audio chunk
            temp_buffer = io.BytesIO()
            with wave.open(temp_buffer, "wb") as wf:
                wf.setsampwidth(2)  # 16-bit
                wf.setnchannels(audio_buffer_processor.num_channels)
                wf.setframerate(audio_buffer_processor.sample_rate)
                wf.writeframes(new_audio)
            temp_buffer.seek(0)

            # If this is the *first* chunk, just copy the entire WAV (header + frames)
            if self._audio_memory_buffer.tell() == 0:
                self._audio_memory_buffer.write(temp_buffer.getvalue())
            else:
                # Otherwise, read out existing frames, then rewrite the WAV
                self._audio_memory_buffer.seek(0)
                with wave.open(self._audio_memory_buffer, "rb") as existing_wf:
                    params = existing_wf.getparams()
                    existing_frames = existing_wf.readframes(existing_wf.getnframes())

                # Truncate the main buffer, and rewrite with the old + new frames
                self._audio_memory_buffer.seek(0)
                self._audio_memory_buffer.truncate(0)

                with wave.open(self._audio_memory_buffer, "wb") as new_wf:
                    new_wf.setparams(params)
                    new_wf.writeframes(existing_frames)
                    # Skip the standard 44-byte header from the new chunk
                    new_wf.writeframes(temp_buffer.getvalue()[44:])

            # Reset so we don't double-process the same audio
            audio_buffer_processor.reset_audio_buffers()
            logger.info("Audio processed and appended to the in-memory buffer.")
            print(self._audio_memory_buffer.read(), flush=True)

            await self.send_audio_file_to_connexity()

        except Exception as e:
            print(f"Failed to process audio: {e}", flush=True)
            logger.error(f"Failed to process audio: {e}")

    async def send_audio_file_to_connexity(self):
        data = aiohttp.FormData()
        data.add_field(
            "file",
            self._audio_memory_buffer,
            filename="audio.wav",
            content_type="audio/wav",
        )
        data.add_field("sid", self._call_id)
        data.add_field(
            "first_speaker_role",
            "assistant" if self._assistant_speaks_first else "user",
        )
        data.add_field("agent_id", self._assistant_id)
        print(data.__dict__, flush=True)
        print(self._audio_memory_buffer.read())
        async with aiohttp.ClientSession() as session:
            async with session.post(self._api_url, headers=self._request_headers(), data=data) as response:
                # Optionally handle the response, for example:
                if response.status != 200:
                    print(f"Failed to send data: {response}")
                else:
                    print(f"Data sent successfully: {response}")


class ConnexityTwilioMetricsService(ConnexityInterface):
    """Initialize a ConnexityTwilioMetricsService instance.

    This class uses an AudioBufferProcessor to get the conversation audio and
    uploads it to Connexity Voice API for audio processing.

    Args:
        call_id (str): Your unique identifier for the call. This is used to match the call in the Connexity Voice system to the call in your system.
        assistant_id (str): Identifier for the AI assistant. This can be whatever you want, it's intended for you convenience so you can distinguish
        between different assistants and a grouping mechanism for calls.
        assistant_speaks_first (bool, optional): Indicates if the assistant speaks first in the conversation. Defaults to True.
        output_dir (str, optional): Directory to save temporary audio files. Defaults to "recordings".

    Attributes:
        call_id (str): Stores the unique call identifier.
        assistant (str): Stores the assistant identifier.
        assistant_speaks_first (bool): Indicates whether the assistant speaks first.
        output_dir (str): Directory path for saving temporary audio files.

    The constructor also ensures that the output directory exists.
    """

    def __init__(
        self,
        *,
        sid: str,
        assistant_id: str,
        api_key: str,
        api_url: str = "https://connexity-gateway-owzhcfagkq-uc.a.run.app/process/blackbox/links",
        assistant_speaks_first: bool = True,
        twilio_account_id: str,
        twilio_auth_token: str,
        **kwargs,
    ):
        super().__init__(call_id=sid,
                         assistant_id=assistant_id,
                         api_key=api_key,
                         api_url=api_url,
                         assistant_speaks_first=assistant_speaks_first,
                         **kwargs)
        self._audio_buffer_processor = audio_buffer_processor
        self._api_url = api_url
        self.twilio_account_id = twilio_account_id
        self.twilio_auth_token = twilio_auth_token

    async def cancel(self, frame: CancelFrame):
        await self.send_audio_url_to_connexity(await self._get_twilio_recording())
        await super().cancel(frame)

    async def _get_twilio_recording(self):
        client = Client(
            os.environ["TWILIO_ACCOUNT_ID"], os.environ["TWILIO_AUTH_TOKEN"]
        )
        i = 0
        recording = None

        while not recording:
            i += 1
            recording = client.recordings.list(call_sid=self._call_id)
            if recording:
                recording_url = f"https://api.twilio.com/2010-04-01/Accounts/{os.environ["TWILIO_ACCOUNT_ID"]}/Recordings/{recording[0].sid}.wav"
                return recording_url
            await sleep(3)
            if i == 3:
                return None


class ConnexityDailyMetricsService(ConnexityInterface):
    """Initialize a ConnexityDailyMetricsService instance.

    This class uses an AudioBufferProcessor to get the conversation audio and
    uploads it to Connexity Voice API for audio processing.

    Args:
        call_id (str): Your unique identifier for the call. This is used to match the call in the Connexity Voice system to the call in your system.
        assistant_id (str): Identifier for the AI assistant. This can be whatever you want, it's intended for you convenience so you can distinguish
        between different assistants and a grouping mechanism for calls.
        assistant_speaks_first (bool, optional): Indicates if the assistant speaks first in the conversation. Defaults to True.
        output_dir (str, optional): Directory to save temporary audio files. Defaults to "recordings".

    Attributes:
        call_id (str): Stores the unique call identifier.
        assistant (str): Stores the assistant identifier.
        assistant_speaks_first (bool): Indicates whether the assistant speaks first.
        output_dir (str): Directory path for saving temporary audio files.

    The constructor also ensures that the output directory exists.
    """

    def __init__(
        self,
        *,
        call_id: str,
        assistant_id: str,
        api_key: str,
        api_url: str = "http://connexity-gateway-owzhcfagkq-uc.a.run.app/process/blackbox/links",
        assistant_speaks_first: bool = True,
        daily_api_key: str,
        room_url: str,
        **kwargs,
    ):
        super().__init__(call_id=call_id,
                         assistant_id=assistant_id,
                         api_key=api_key,
                         api_url=api_url,
                         assistant_speaks_first=assistant_speaks_first,
                         **kwargs)
        self._audio_buffer_processor = audio_buffer_processor
        self._api_url = api_url
        self.daily_api_key = daily_api_key
        self._room_url = room_url

    # async def stop(self, frame: EndFrame):
    #     print("END FRAME RECEIVED", flush=True)
    #     await self.send_audio_url_to_connexity(self._get_daily_recording(self._room_url))
    #     await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        print("CANCEL FRAME RECEIVED", flush=True)
        await self.send_audio_url_to_connexity(await self._get_daily_recording(self._room_url))
        await super().cancel(frame)

    async def _get_daily_recording(self, room_url):
        import requests
        call_info_url = 'https://api.daily.co/v1/recordings?room_name={room_name}'
        download_link_url = 'https://api.daily.co/v1/recordings/{call_id}/access-link'
        i = 0

        while i != 3:
            try:
                room_name = room_url.split('/')[-1]
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.daily_api_key}'}
                response = requests.get(call_info_url.format(room_name=room_name), headers=headers)

                call_id = response.json()['data'][0]['id']
                response = requests.get(download_link_url.format(call_id=call_id), headers=headers)
                download_url = response.json()['download_link']

                return download_url
            except Exception:
                i += 1
                await sleep(3)
                continue
