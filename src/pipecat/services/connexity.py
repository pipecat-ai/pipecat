#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import io
import wave
from asyncio import sleep

import aiohttp
from loguru import logger

from pipecat.frames.frames import CancelFrame, EndFrame, Frame
from pipecat.processors.audio import audio_buffer_processor
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import AIService


class ConnexityInterface(AIService):
    """
    Base interface for sending metadata or audio to a Connexity service.

    This class:
    - Stores references to the call ID, the assistant ID, an API key, and a
      target API URL.
    - Provides a generic method to post JSON data to the remote server.

    Args:
        call_id (str): Unique identifier for the call.
        assistant_id (str): Identifier for the AI assistant.
        api_key (str): API key to authenticate requests.
        api_url (str): URL to which data will be posted.
        assistant_speaks_first (bool): Indicates who speaks first in the conversation.

    Attributes:
        _audio_buffer_processor (AudioBufferProcessor): Manages raw audio buffers.
        _audio_memory_buffer (io.BytesIO): In-memory buffer accumulating WAV data.
        _api_key (str): API key included in the request headers.
        _api_url (str): Endpoint where data (e.g., audio) is sent.
        _call_id (str): Unique call identifier.
        _assistant_id (str): Assistant or agent identifier.
        _assistant_speaks_first (bool): Whether the assistant is the first speaker.
    """

    def __init__(
        self,
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
        """
        Prepare headers for the outgoing request.
        This includes the API key for authorization.
        """
        return {"Content-Type": "application/json", "X-API-KEY": self._api_key}

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

    async def send_audio_url_to_connexity(self, audio_url):
        """
        Send an audio URL (instead of a raw file) to the Connexity server.

        Args:
            audio_url (str): The URL pointing to an audio resource.

        Returns:
            None
        """
        answer_list = {
            "items": [
                {
                    "agent_id": self._assistant_id,
                    "sid": self._call_id,
                    "first_speaker_role": (
                        "assistant" if self._assistant_speaks_first else "user"
                    ),
                    "audio_link": audio_url
                }
            ]
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._api_url,
                headers=self._request_headers(),
                json=answer_list
            ) as response:
                if response.status != 200:
                    print(f"Failed to send data: {response.status}")
                else:
                    print(f"Data sent successfully: {response.status}")


class ConnexityLocalMetricsService(ConnexityInterface):
    """
    Maintains an in-memory WAV file for the entire call and uploads it
    to a server endpoint when the call ends.

    This class extends ConnexityInterface by:
    - Gathering raw audio data in an AudioBufferProcessor.
    - Merging all incoming audio chunks into a single WAV file in memory.
    - Sending the final WAV as a `multipart/form-data` upload once the call stops.

    Args:
        audio_buffer_processor (AudioBufferProcessor): Manages the raw audio chunks.
        call_id (str): A unique ID for this call.
        assistant_id (str): Identifier for the AI assistant.
        api_key (str): Your API key, used in request headers.
        api_url (str, optional): Endpoint URL for sending the audio file.
            Defaults to "http://localhost:8080/process/blackbox/file/pipecat".
        assistant_speaks_first (bool, optional): Indicates if the assistant
            speaks first in the conversation. Defaults to True.

    Attributes:
        _audio_buffer_processor (AudioBufferProcessor): Used to retrieve raw PCM data.
        _audio_memory_buffer (io.BytesIO): Holds the ever-growing WAV file in memory.
        _api_url (str): Endpoint for sending the WAV data.
    """

    def __init__(
        self,
        *,
        audio_buffer_processor: AudioBufferProcessor,
        call_id: str,
        assistant_id: str,
        api_key: str,
        api_url: str = "http://localhost:8080/process/blackbox/file/pipecat",
        assistant_speaks_first: bool = True,
        **kwargs,
    ):
        super().__init__(
            call_id=call_id,
            assistant_id=assistant_id,
            api_key=api_key,
            api_url=api_url,
            assistant_speaks_first=assistant_speaks_first,
            **kwargs
        )
        self._audio_buffer_processor = audio_buffer_processor
        self._audio_memory_buffer = io.BytesIO()
        self._api_url = api_url

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._process_audio()
        await self.send_audio_file_to_connexity()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._process_audio()

    def _request_headers(self):
        """
        Overrides the base request headers to only include the API key without
        forcing JSON content type (since we do multipart).
        """
        return {"X-API-KEY": self._api_key}

    async def _process_audio(self):
        """
        Merge the buffered PCM audio into the ongoing in-memory WAV file.

        If no audio is available (buffer is empty), does nothing.
        Otherwise, the first chunk writes the WAV header and frames;
        subsequent chunks skip the header and only append frames.
        """
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

        except Exception as e:
            print(f"Failed to process audio: {e}", flush=True)
            logger.error(f"Failed to process audio: {e}")

    async def send_audio_file_to_connexity(self):
        """
        Perform a multipart/form-data upload of the entire in-memory WAV
        (accumulated so far) to the remote endpoint.

        The form fields `sid`, `first_speaker_role`, and `agent_id`
        are sent along with the file.
        """
        data = aiohttp.FormData()

        # *Important* - Seek to the beginning so all WAV data can be read
        self._audio_memory_buffer.seek(0)

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
        # Debug read
        print(self._audio_memory_buffer.read(), flush=True)

        # Must reset again if you plan to read it a second time
        self._audio_memory_buffer.seek(0)

        async with aiohttp.ClientSession() as session:
            async with session.post(self._api_url, headers=self._request_headers(), data=data) as response:
                if response.status != 200:
                    print(f"Failed to send data: {response}")
                else:
                    print(f"Data sent successfully: {response}")


class ConnexityDailyMetricsService(ConnexityInterface):
    """
    An alternative service that retrieves audio from a third-party
    (Daily.co) instead of building a local WAV buffer.

    Upon stopping or canceling a call, this service:
      1. Retrieves a download link for the call's recorded audio from Daily.co.
      2. Posts the link to the Connexity endpoint.

    Args:
        call_id (str): Unique ID for the call.
        assistant_id (str): Identifier for the AI assistant.
        api_key (str): API key for sending data to Connexity.
        api_url (str): Endpoint for sending the Daily.co audio link.
            Defaults to "http://localhost:8080/process/blackbox/links".
        assistant_speaks_first (bool): Who speaks first (assistant or user).
        daily_api_key (str): API key to authenticate Daily.co requests.
        room_url (str): The Daily.co room URL whose recording we retrieve.

    Attributes:
        _room_url (str): The Daily.co room URL.
    """

    def __init__(
        self,
        *,
        call_id: str,
        assistant_id: str,
        api_key: str,
        api_url: str = "http://localhost:8080/process/blackbox/links",
        assistant_speaks_first: bool = True,
        daily_api_key: str,
        room_url: str,
        **kwargs,
    ):
        super().__init__(
            call_id=call_id,
            assistant_id=assistant_id,
            api_key=api_key,
            api_url=api_url,
            assistant_speaks_first=assistant_speaks_first,
            **kwargs
        )
        self._audio_buffer_processor = audio_buffer_processor
        self._api_url = api_url
        self.daily_api_key = daily_api_key
        self._room_url = room_url

    async def stop(self, frame: EndFrame):
        """
        Called when the call ends.
        Retrieves a Daily.co recording link and sends it to Connexity.
        """
        print("END FRAME RECEIVED", flush=True)
        await self.send_audio_url_to_connexity(await self._get_daily_recording(self._room_url))
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        """
        Called if the call is canceled.
        Retrieves a Daily.co recording link and sends it to Connexity.
        """
        print("CANCEL FRAME RECEIVED", flush=True)
        await self.send_audio_url_to_connexity(await self._get_daily_recording(self._room_url))
        await super().cancel(frame)

    async def _get_daily_recording(self, room_url):
        """
        Queries the Daily.co API for a list of recordings related to a room,
        extracts the latest recording's ID, and retrieves a download link for it.

        Args:
            room_url (str): Daily.co room URL.

        Returns:
            str: Download URL of the latest recording, if available.
        """
        import requests
        call_info_url = 'https://api.daily.co/v1/recordings?room_name={room_name}'
        download_link_url = 'https://api.daily.co/v1/recordings/{call_id}/access-link'
        i = 0

        while i != 3:
            try:
                room_name = room_url.split('/')[-1]
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.daily_api_key}'
                }
                response = requests.get(call_info_url.format(room_name=room_name), headers=headers)

                call_id = response.json()['data'][0]['id']
                response = requests.get(download_link_url.format(call_id=call_id), headers=headers)
                download_url = response.json()['download_link']

                return download_url
            except Exception:
                i += 1
                await sleep(3)
                continue
