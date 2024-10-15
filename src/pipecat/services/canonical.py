#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import aiohttp
import os
import uuid

from datetime import datetime
from typing import Dict, List, Tuple

from pipecat.frames.frames import CancelFrame, EndFrame, Frame
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import AIService

from loguru import logger

try:
    import aiofiles
    import aiofiles.os
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Canonical Metrics, you need to `pip install pipecat-ai[canonical]`. "
        + "Also, set the `CANONICAL_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")


# Multipart upload part size in bytes, cannot be smaller than 5MB
PART_SIZE = 1024 * 1024 * 5


class CanonicalMetricsService(AIService):
    """Initialize a CanonicalAudioProcessor instance.

    This class uses an AudioBufferProcessor to get the conversation audio and
    uploads it to Canonical Voice API for audio processing.

    Args:

        call_id (str): Your unique identifier for the call. This is used to match the call in the Canonical Voice system to the call in your system.
        assistant (str): Identifier for the AI assistant. This can be whatever you want, it's intended for you convenience so you can distinguish
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
        aiohttp_session: aiohttp.ClientSession,
        audio_buffer_processor: AudioBufferProcessor,
        call_id: str,
        assistant: str,
        api_key: str,
        api_url: str = "https://voiceapp.canonical.chat/api/v1",
        assistant_speaks_first: bool = True,
        output_dir: str = "recordings",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._aiohttp_session = aiohttp_session
        self._audio_buffer_processor = audio_buffer_processor
        self._api_key = api_key
        self._api_url = api_url
        self._call_id = call_id
        self._assistant = assistant
        self._assistant_speaks_first = assistant_speaks_first
        self._output_dir = output_dir

    async def stop(self, frame: EndFrame):
        await self._process_audio()

    async def cancel(self, frame: CancelFrame):
        await self._process_audio()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

    async def _process_audio(self):
        pipeline = self._audio_buffer_processor
        if pipeline.has_audio():
            os.makedirs(self._output_dir, exist_ok=True)
            filename = self._get_output_filename()
            wave_data = pipeline.merge_audio_buffers()

            async with aiofiles.open(filename, "wb") as file:
                await file.write(wave_data)

            try:
                await self._multipart_upload(filename)
                pipeline.reset_audio_buffer()
                await aiofiles.os.remove(filename)
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.error(f"Failed to upload recording: {e}")

    def _get_output_filename(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self._output_dir}/{timestamp}-{uuid.uuid4().hex}.wav"

    def _request_headers(self):
        return {"Content-Type": "application/json", "X-Canonical-Api-Key": self._api_key}

    async def _multipart_upload(self, file_path: str):
        upload_request, upload_response = await self._request_upload(file_path)
        if upload_request is None or upload_response is None:
            return
        parts = await self._upload_parts(file_path, upload_response)
        if parts is None:
            return
        await self._upload_complete(parts, upload_request, upload_response)

    async def _request_upload(self, file_path: str) -> Tuple[Dict, Dict]:
        filename = os.path.basename(file_path)
        filesize = os.path.getsize(file_path)
        numparts = int((filesize + PART_SIZE - 1) / PART_SIZE)

        params = {
            "filename": filename,
            "parts": numparts,
            "callId": self._call_id,
            "assistant": {"id": self._assistant, "speaksFirst": self._assistant_speaks_first},
        }
        logger.debug(f"Requesting presigned URLs for {numparts} parts")
        response = await self._aiohttp_session.post(
            f"{self._api_url}/recording/uploadRequest", headers=self._request_headers(), json=params
        )
        if not response.ok:
            logger.error(f"Failed to get presigned URLs: {await response.text()}")
            return None, None
        response_json = await response.json()
        return params, response_json

    async def _upload_parts(self, file_path: str, upload_response: Dict) -> List[Dict]:
        urls = upload_response["urls"]
        parts = []
        try:
            async with aiofiles.open(file_path, "rb") as file:
                for partnum, upload_url in enumerate(urls, start=1):
                    data = await file.read(PART_SIZE)
                    if not data:
                        break

                    response = await self._aiohttp_session.put(upload_url, data=data)
                    if not response.ok:
                        logger.error(f"Failed to upload part {partnum}: {await response.text()}")
                        return None

                    etag = response.headers["ETag"]
                    parts.append({"partnum": str(partnum), "etag": etag})

        except Exception as e:
            logger.error(f"Multipart upload aborted, an error occurred: {str(e)}")
        return parts

    async def _upload_complete(
        self, parts: List[Dict], upload_request: Dict, upload_response: Dict
    ):
        params = {
            "filename": upload_request["filename"],
            "parts": parts,
            "slug": upload_response["slug"],
            "callId": self._call_id,
            "assistant": {"id": self._assistant, "speaksFirst": self._assistant_speaks_first},
        }
        logger.debug(f"Completing upload for {params['filename']}")
        logger.debug(f"Slug: {params['slug']}")
        response = await self._aiohttp_session.post(
            f"{self._api_url}/recording/uploadComplete",
            headers=self._request_headers(),
            json=params,
        )
        if not response.ok:
            logger.error(f"Failed to complete upload: {await response.text()}")
            return
