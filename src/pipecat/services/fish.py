import asyncio
import base64
from typing import Any, AsyncGenerator, Dict, Optional, Literal

import websockets
from loguru import logger
from pydantic import BaseModel
import ormsgpack  # Import ormsgpack for MessagePack encoding/decoding

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    StartFrame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import TTSService
from pipecat.transcriptions.language import Language

# FishAudio supports various output formats
FishAudioOutputFormat = Literal["opus", "mp3", "wav"]

def language_to_fishaudio_language(language: Language) -> str:
    # Map Language enum to fish.audio language codes
    language_map = {
        Language.EN: "en-US",
        Language.EN_US: "en-US",
        Language.EN_GB: "en-GB",
        Language.ES: "es-ES",
        Language.FR: "fr-FR",
        Language.DE: "de-DE",
        # Add other mappings as needed
    }
    return language_map.get(language, "en-US")  # Default to 'en-US' if not found

def sample_rate_from_output_format(output_format: str) -> int:
    # FishAudio might have specific sample rates per format
    format_sample_rates = {
        "opus": 24000,
        "mp3": 24000,
        "wav": 24000,
    }
    return format_sample_rates.get(output_format, 24000)  # Default to 24kHz

class FishAudioTTSService(TTSService):
    class InputParams(BaseModel):
        language: Optional[Language] = Language.EN
        latency: Optional[str] = "normal"  # "normal" or "balanced"
        prosody_speed: Optional[float] = 1.0  # Speech speed (0.5-2.0)
        prosody_volume: Optional[int] = 0     # Volume adjustment in dB

    def __init__(
        self,
        *,
        api_key: str,
        model_id: str,
        output_format: FishAudioOutputFormat = "wav",
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate_from_output_format(output_format),
            **kwargs,
        )

        self._api_key = api_key
        self._model_id = model_id
        self._url = "wss://api.fish.audio/v1/tts/live"
        self._output_format = output_format

        self._settings = {
            "sample_rate": sample_rate_from_output_format(output_format),
            # "language": self.language_to_service_language(params.language)
            #     if params.language else "en-US",
            "latency": params.latency,
            "prosody": {
                "speed": params.prosody_speed,
                "volume": params.prosody_volume,
            },
            "format": output_format,
            "reference_id": model_id,
        }

        self._websocket = None
        self._receive_task = None
        self._started = False

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> str:
        return language_to_fishaudio_language(language)

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        try:
            headers = {
                "Authorization": f"Bearer {self._api_key}",
            }

            self._websocket = await websockets.connect(self._url, extra_headers=headers)
            self._receive_task = asyncio.create_task(self._receive_task_handler())

            # Send 'start' event to initialize the session
            start_message = {
                "event": "start",
                "request": {
                    "text": "",  # Initial empty text
                    "latency": self._settings["latency"],
                    "format": self._output_format,
                    "prosody": self._settings["prosody"],
                    "reference_id": self._settings["reference_id"],
                    "sample_rate": self._settings["sample_rate"],
                },
                "debug": True,  # Added debug flag
            }
            await self._websocket.send(ormsgpack.packb(start_message))
            # logger.debug("Sent start event to fish.audio WebSocket")

        except Exception as e:
            # logger.exception(f"Error connecting to fish.audio WebSocket: {e}")
            self._websocket = None

    async def _disconnect(self):
        try:
            await self.stop_all_metrics()

            if self._websocket:
                # Send 'stop' event to end the session
                stop_message = {
                    "event": "stop"
                }
                await self._websocket.send(ormsgpack.packb(stop_message))
                await self._websocket.close()
                self._websocket = None

            if self._receive_task:
                self._receive_task.cancel()
                await self._receive_task
                self._receive_task = None

            self._started = False
        except Exception as e:
            logger.error(f"Error disconnecting from fish.audio WebSocket: {e}")

    async def _receive_task_handler(self):
        try:
            while True:
                try:
                    message = await self._websocket.recv()
                    if isinstance(message, bytes):
                        msg = ormsgpack.unpackb(message)
                        event = msg.get("event")

                        if event == "audio":
                            await self.stop_ttfb_metrics()
                            audio_data = msg.get("audio")
                            # Audio data is binary, no need to base64 decode
                            frame = TTSAudioRawFrame(
                                audio_data, self._settings["sample_rate"], 1)
                            await self.push_frame(frame)
                        elif event == "finish":
                            reason = msg.get("reason")
                            if reason == "stop":
                                await self.push_frame(TTSStoppedFrame())
                                self._started = False
                            elif reason == "error":
                                error_msg = msg.get("error", "Unknown error")
                                logger.error(f"fish.audio error: {error_msg}")
                                await self.push_error(ErrorFrame(f"fish.audio error: {error_msg}"))
                                self._started = False
                        elif event == "error":
                            error_msg = msg.get("error", "Unknown error")
                            logger.error(f"fish.audio error: {error_msg}")
                            await self.push_error(ErrorFrame(f"fish.audio error: {error_msg}"))
                    else:
                        logger.warning(f"Received unexpected message type: {type(message)}")
                except asyncio.TimeoutError:
                    logger.warning("No message received from fish.audio within timeout period")
                except websockets.ConnectionClosed as e:
                    logger.error(f"WebSocket connection closed: {e}")
                    break
        except Exception as e:
            logger.exception(f"Exception in receive task: {e}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSSpeakFrame):
            await self.pause_processing_frames()
        elif isinstance(frame, LLMFullResponseEndFrame) and self._started:
            await self.pause_processing_frames()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self.resume_processing_frames()

    async def _handle_interruption(self, frame: StartInterruptionFrame, direction: FrameDirection):
        await super()._handle_interruption(frame, direction)
        await self.stop_all_metrics()

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating Fish TTS: [{text}]")

        try:
            if not self._websocket or self._websocket.closed:
                await self._connect()

            if not self._started:
                await self.start_ttfb_metrics()
                yield TTSStartedFrame()
                self._started = True

            # Send 'text' event to stream text chunks
            text_message = {
                "event": "text",
                "text": text + " "  # Ensure a space at the end
            }
            logger.debug(f"Sending text message: {text_message}")
            await self._websocket.send(ormsgpack.packb(text_message))
            logger.debug("Sent text message to fish.audio WebSocket")

            await self.start_tts_usage_metrics(text)

            # The audio frames will be received in _receive_task_handler
            yield None

        except Exception as e:
            logger.error(f"Error in run_tts: {e}")
            yield ErrorFrame(f"Error in run_tts: {str(e)}")
