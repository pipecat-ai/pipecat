from typing import AsyncGenerator, Optional
import aiohttp
import json
from loguru import logger
from pydantic import BaseModel
import time

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.ai_services import TTSService
from pipecat.transcriptions.language import Language

class HailuoTTSService(TTSService):
    class InputParams(BaseModel):
        speed: Optional[float] = 1.0
        volume: Optional[float] = 1.0
        pitch: Optional[float] = 0
        
    def __init__(
        self,
        *,
        api_key: str,
        group_id: str,
        model: str = "speech-01-turbo",
        voice_id: str = "Santa_Claus",
        sample_rate: int = 24000,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        
        self._api_key = api_key
        self._group_id = group_id
        self._base_url = f"https://api.minimaxi.chat/v1/t2a_v2?GroupId={group_id}"
        self._session = None
        
        self._settings = {
            "model": model,
            "stream": True,
            "voice_setting": {
                "voice_id": voice_id,
                "speed": params.speed,
                "vol": params.volume,
                "pitch": params.pitch
            },
            "audio_setting": {
                "sample_rate": sample_rate,
                "bitrate": 128000,
                "format": "pcm",
                "channel": 1
            }
        }

    def can_generate_metrics(self) -> bool:
        return True

    async def _init_session(self):
        if self._session is None:
            self._session = aiohttp.ClientSession()

    async def _close_session(self):
        if self._session:
            await self._session.close()
            self._session = None

    async def start(self, frame: Frame):
        await super().start(frame)
        await self._init_session()

    async def stop(self, frame: Frame):
        await super().stop(frame)
        await self._close_session()

    async def cancel(self, frame: Frame):
        await super().cancel(frame)
        await self._close_session()

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        text = text.strip()
        if not text or text in ['"', "'", ']', '[']:
            logger.debug(f"Skipping invalid text for TTS: [{text}]")
            return
            
        logger.debug(f"Generating TTS: [{text}]")
        start_time = time.time()

        try:
            await self._init_session()
            await self.start_ttfb_metrics()
            yield TTSStartedFrame()

            headers = {
                'accept': 'application/json, text/plain, */*',
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self._api_key}'
            }

            payload = {
                "model": self._settings["model"],
                "text": text,
                "stream": True,
                "voice_setting": self._settings["voice_setting"],
                "audio_setting": self._settings["audio_setting"]
            }

            async with self._session.post(
                self._base_url,
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"TTS API error: {response.status} - {error_text}")

                await self.start_tts_usage_metrics(text)
                await self.stop_ttfb_metrics()

                buffer = bytearray()
                # Control initial read size
                async for chunk in response.content.iter_chunked(4096):
                    if not chunk:
                        continue
                        
                    buffer.extend(chunk)
                    
                    # Find complete data blocks
                    while b'data:' in buffer:
                        start = buffer.find(b'data:')
                        next_start = buffer.find(b'data:', start + 5)
                        
                        if next_start == -1:
                            # No next data block found, keep current data for next iteration
                            if start > 0:
                                buffer = buffer[start:]
                            break
                            
                        # Extract a complete data block
                        data_block = buffer[start:next_start]
                        buffer = buffer[next_start:]
                        
                        try:
                            data = json.loads(data_block[5:].decode('utf-8'))
                            # Skip data blocks containing extra_info
                            if "extra_info" in data:
                                logger.debug("Received final chunk with extra info")
                                break

                            chunk_data = data.get("data", {})
                            if not chunk_data:
                                continue

                            audio_data = chunk_data.get("audio")
                            if not audio_data:
                                continue

                            # Process audio data in chunks
                            CHUNK_SIZE = 4096  # 4KB per chunk
                            for i in range(0, len(audio_data), CHUNK_SIZE * 2):  # *2 for hex string
                                # Split hex string
                                hex_chunk = audio_data[i:i + CHUNK_SIZE * 2]
                                if not hex_chunk:
                                    continue
                                    
                                try:
                                    # Convert this chunk of data
                                    audio_chunk = bytes.fromhex(hex_chunk)
                                    if audio_chunk:
                                        yield TTSAudioRawFrame(
                                            audio=audio_chunk,
                                            sample_rate=self._settings["audio_setting"]["sample_rate"],
                                            num_channels=self._settings["audio_setting"]["channel"]
                                        )
                                except ValueError as e:
                                    logger.error(f"Error converting hex to binary: {e}")
                                    continue

                        except json.JSONDecodeError as e:
                            logger.error(f"Error decoding JSON: {e}, data: {data_block[:100]}")
                            continue

            yield TTSStoppedFrame()
            
            total_time = time.time() - start_time
            logger.debug(f"Total TTS processing time: {total_time:.4f}s for {len(text)} chars")

        except Exception as e:
            logger.error(f"{self} error generating TTS: {e}")
            yield ErrorFrame(f"{self} error: {str(e)}")
        finally:
            await self.stop_all_metrics()
