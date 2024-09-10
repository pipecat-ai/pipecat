import os
import uuid
import wave
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Tuple

import aiohttp
from loguru import logger

try:
    import aiofiles
    import aiofiles.os
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Canonical Metrics, you need to `pip install pipecat-ai[canonical]`. " +
        "Also, set the `CANONICAL_API_KEY` environment variable.")
    raise Exception(f"Missing module: {e}")


from pipecat.frames.frames import CancelFrame, EndFrame, Frame
from pipecat.processors.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.frame_processor import FrameDirection

"""
This class extends AudioBufferProcessor to handle audio processing and uploading
for the Canonical Voice API.
"""


class CanonicalMetrics(AudioBufferProcessor):
    """
    Initialize a CanonicalAudioProcessor instance.

    This class extends AudioBufferProcessor to handle audio processing and uploading
    for the Canonical Voice API.

    Args:
        call_id (str): Your unique identifier for the call. This is used to match the call in the Canonical Voice system to the call in your system.
        assistant (str): Identifier for the AI assistant. This can be whatever you want, it's intended for you convenience so you can distinguish
        between different assistants and a grouping mechanism for calls.
        assistant_speaks_first (bool, optional): Indicates if the assistant speaks first in the conversation. Defaults to True.
        output_dir (str, optional): Directory to save temporary audio files. Defaults to "recordings".
        default_part_size (int, optional): Default size for multipart upload parts in bytes. Defaults to 1MB (1024 * 1024 * 1).

    Attributes:
        call_id (str): Stores the unique call identifier.
        assistant (str): Stores the assistant identifier.
        assistant_speaks_first (bool): Indicates whether the assistant speaks first.
        output_dir (str): Directory path for saving temporary audio files.
        partsize (int): Size of each part for multipart uploads.

    The constructor also ensures that the output directory exists.
    This class requires a Canonical API key to be set in the CANONICAL_API_KEY environment variable.
    """

    def __init__(
            self,
            call_id: str,
            assistant: str,
            assistant_speaks_first: bool = True,
            output_dir: str = "recordings",
            default_part_size: int = 1024 * 1024 * 1):
        super().__init__()
        if not os.environ.get("CANONICAL_API_KEY"):
            raise ValueError(
                "CANONICAL_API_KEY is not set, a Canonical API key is required to use this class")
        self.call_id = call_id
        self.assistant = assistant
        self.assistant_speaks_first = assistant_speaks_first
        self.output_dir = output_dir
        self.partsize = default_part_size
        self.end_of_call = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if self.end_of_call:
            return

        if (isinstance(frame, EndFrame) or isinstance(frame, CancelFrame)):
            self.end_of_call = True
            if self.has_audio():
                os.makedirs(self.output_dir, exist_ok=True)
                filename = self.get_output_filename()
                with BytesIO() as buffer:
                    with wave.open(buffer, 'wb') as wf:
                        wf.setnchannels(self.num_channels)
                        wf.setsampwidth(self.sample_rate // 8000)
                        wf.setframerate(self.sample_rate)
                        wf.writeframes(self.audio_buffer)
                    wave_data = buffer.getvalue()

                async with aiofiles.open(filename, 'wb') as file:
                    await file.write(wave_data)

                try:
                    await self.multipart_upload(filename)
                    await aiofiles.os.remove(filename)
                except FileNotFoundError:
                    pass
                except Exception as e:
                    raise e
                self.audio_buffer = bytearray()

    def get_output_filename(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.output_dir}/{timestamp}-{uuid.uuid4().hex}.wav"

    def canonical_api_url(self):
        return os.environ.get("CANONICAL_API_URL", "https://voiceapp.canonical.chat/api/v1")

    def request_headers(self):
        return {
            "Content-Type": "application/json",
            "X-Canonical-Api-Key": os.environ.get("CANONICAL_API_KEY")
        }

    async def multipart_upload(self, file_path: str):
        upload_request, upload_response = await self.request_upload(file_path)
        parts = await self.upload_parts(file_path, upload_request, upload_response)
        await self.upload_complete(parts, upload_request, upload_response)

    async def request_upload(self, file_path: str) -> Tuple[Dict, Dict]:
        filename = os.path.basename(file_path)
        filename = f"{str(uuid.uuid4())}-{filename}"
        filesize = os.path.getsize(file_path)
        numparts = int((filesize + self.partsize - 1) / self.partsize)

        params = {
            'filename': filename,
            'parts': numparts,
            'assistant': self.assistant,
            'assistantSpeaksFirst': self.assistant_speaks_first
        }
        print(f"Requesting presigned URLs for {numparts} parts")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.canonical_api_url()}/recording/uploadRequest",
                headers=self.request_headers(),
                json=params
            ) as response:
                if not response.ok:
                    raise Exception(f"Failed to get presigned URLs: {await response.text()}")
                response_json = await response.json()
        return params, response_json

    async def upload_parts(
            self,
            file_path: str,
            upload_request: Dict,
            upload_response: Dict) -> List[Dict]:

        urls = upload_response['urls']
        parts = []
        try:
            async with aiofiles.open(file_path, 'rb') as file:
                async with aiohttp.ClientSession() as session:
                    for partnum, upload_url in enumerate(urls, start=1):
                        data = await file.read(self.partsize)
                        if not data:
                            break

                        async with session.put(upload_url, data=data) as response:
                            if not response.ok:
                                logger.error(f"Failed to upload part {partnum}: {await response.text()}")
                                raise Exception(f"Failed to upload part {partnum}: {await response.text()}")

                            etag = response.headers['ETag']
                            parts.append({'partnum': str(partnum), 'etag': etag})

        except Exception as e:
            logger.error(f"Multipart upload aborted, an error occurred: {str(e)}")
        return parts

    async def upload_complete(
            self,
            parts: List[Dict],
            upload_request: Dict,
            upload_response: Dict):

        params = {
            'filename': upload_request['filename'],
            'parts': parts,
            'slug': upload_response['slug'],
            'callId': self.call_id,
            'assistant': {
                'id': self.assistant,
                'speaksFirst': self.assistant_speaks_first
            }
        }
        print(f"Completing upload for {params['filename']}")
        print(f"Slug: {params['slug']}")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.canonical_api_url()}/recording/uploadComplete",
                headers=self.request_headers(),
                json=params
            ) as response:
                if not response.ok:
                    logger.error(f"Failed to complete upload: {await response.text()}")
                    raise Exception(f"Failed to complete upload: {await response.text()}")
