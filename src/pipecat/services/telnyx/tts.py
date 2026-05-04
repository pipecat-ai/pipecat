import sys
import json
import base64
import urllib.parse
from typing import AsyncGenerator, Optional

from loguru import logger

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.services.telnyx.utils import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_CHANNELS,
    TTS_WS_URL,
    get_api_key,
)

try:
    import websockets
    from websockets.exceptions import InvalidStatusCode
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    raise Exception(f"Missing module: {e}")


class TelnyxTTSService(TTSService):
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        voice: str = "Telnyx.NaturalHD.astra",
        model: Optional[str] = None,
        sample_rate: Optional[int] = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate or DEFAULT_SAMPLE_RATE, **kwargs)
        self._api_key = get_api_key(api_key)
        self._voice = voice
        self._model = model
        self._websocket = None

    def can_generate_metrics(self) -> bool:
        return True

    def set_voice(self, voice: str):
        self._voice = voice

    @override
    async def start(self, frame: StartFrame):
        await super().start(frame)

    @override
    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    @override
    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    @override
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")
        yield TTSStartedFrame()

        await self._connect()

        if not self._websocket:
            yield ErrorFrame(error=f"{self} failed to connect to Telnyx TTS")
            yield TTSStoppedFrame()
            return

        try:
            await self._websocket.send(json.dumps({"text": " "}))
            await self._websocket.send(json.dumps({"text": text}))
            await self._websocket.send(json.dumps({"text": ""}))

            async for message in self._websocket:
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue

                if "error" in data:
                    logger.error(f"Telnyx TTS error: {data['error']}")
                    continue

                audio_b64 = data.get("audio")
                if audio_b64:
                    pcm_bytes = base64.b64decode(audio_b64)

                    if pcm_bytes:
                        yield TTSAudioRawFrame(
                            audio=pcm_bytes,
                            sample_rate=self.sample_rate,
                            num_channels=DEFAULT_CHANNELS,
                        )

                if data.get("isFinal"):
                    break

        except InvalidStatusCode as e:
            logger.error(f"WebSocket Handshake Failed: {e.status_code}")
            yield ErrorFrame(f"Telnyx Connection Rejected ({e.status_code}). Check API Key.")

        except Exception as e:
            logger.error(f"{self} TTS stream error: {e}")
            yield ErrorFrame(error=f"Telnyx TTS stream error: {e}")

        finally:
            await self._disconnect()
            yield TTSStoppedFrame()

    async def _connect(self):
        if self._websocket:
            await self._disconnect()

        params = {
            "voice": self._voice,
            "audio_format": "pcm"
        }
        if self._model:
            params["model"] = self._model

        query_string = urllib.parse.urlencode(params)
        url = f"{TTS_WS_URL}?{query_string}"
        headers = {"Authorization": f"Bearer {self._api_key}"}

        self._websocket = await websockets.connect(url, extra_headers=headers)

    async def _disconnect(self):
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception as e:
                logger.debug(f"{self} error closing socket: {e}")
            finally:
                self._websocket = None
