#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import json
import os
import random
import string
from typing import AsyncGenerator, Optional

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.services.aws.utils import build_event_message, decode_event, get_presigned_url
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use AWS services, you need to `pip install pipecat-ai[aws]`.")
    raise Exception(f"Missing module: {e}")


class AWSTranscribeSTTService(STTService):
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        region: Optional[str] = "us-east-1",
        sample_rate: int = 16000,
        language: Language = Language.EN,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._settings = {
            "sample_rate": sample_rate,
            "language": language,
            "media_encoding": "linear16",  # AWS expects raw PCM
            "number_of_channels": 1,
            "show_speaker_label": False,
            "enable_channel_identification": False,
        }

        # Validate sample rate - AWS Transcribe only supports 8000 Hz or 16000 Hz
        if sample_rate not in [8000, 16000]:
            logger.warning(
                f"AWS Transcribe only supports 8000 Hz or 16000 Hz sample rates. Converting from {sample_rate} Hz to 16000 Hz."
            )
            self._settings["sample_rate"] = 16000

        self._credentials = {
            "aws_access_key_id": aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID"),
            "aws_secret_access_key": api_key or os.getenv("AWS_SECRET_ACCESS_KEY"),
            "aws_session_token": aws_session_token or os.getenv("AWS_SESSION_TOKEN"),
            "region": region or os.getenv("AWS_REGION", "us-east-1"),
        }

        self._ws_client = None
        self._connection_lock = asyncio.Lock()
        self._connecting = False
        self._receive_task = None

    def get_service_encoding(self, encoding: str) -> str:
        """Convert internal encoding format to AWS Transcribe format."""
        encoding_map = {
            "linear16": "pcm",  # AWS expects "pcm" for 16-bit linear PCM
        }
        return encoding_map.get(encoding, encoding)

    async def start(self, frame: StartFrame):
        """Initialize the connection when the service starts."""
        await super().start(frame)
        logger.info("Starting AWS Transcribe service...")
        retry_count = 0
        max_retries = 3

        while retry_count < max_retries:
            try:
                await self._connect()
                if self._ws_client and self._ws_client.open:
                    logger.info("Successfully established WebSocket connection")
                    return
                logger.warning("WebSocket connection not established after connect")
            except Exception as e:
                logger.error(f"Failed to connect (attempt {retry_count + 1}/{max_retries}): {e}")
                retry_count += 1
                if retry_count < max_retries:
                    await asyncio.sleep(1)  # Wait before retrying

        raise RuntimeError("Failed to establish WebSocket connection after multiple attempts")

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Process audio data and send to AWS Transcribe"""
        try:
            # Ensure WebSocket is connected
            if not self._ws_client or not self._ws_client.open:
                logger.debug("WebSocket not connected, attempting to reconnect...")
                try:
                    await self._connect()
                except Exception as e:
                    logger.error(f"Failed to reconnect: {e}")
                    yield ErrorFrame("Failed to reconnect to AWS Transcribe", fatal=False)
                    return

            # Format the audio data according to AWS event stream format
            event_message = build_event_message(audio)

            # Send the formatted event message
            try:
                await self._ws_client.send(event_message)
                # Start metrics after first chunk sent
                await self.start_processing_metrics()
                await self.start_ttfb_metrics()
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"Connection closed while sending: {e}")
                await self._disconnect()
                # Don't yield error here - we'll retry on next frame
            except Exception as e:
                logger.error(f"Error sending audio: {e}")
                yield ErrorFrame(f"AWS Transcribe error: {str(e)}", fatal=False)
                await self._disconnect()

        except Exception as e:
            logger.error(f"Error in run_stt: {e}")
            yield ErrorFrame(f"AWS Transcribe error: {str(e)}", fatal=False)
            await self._disconnect()

    async def _connect(self):
        """Connect to AWS Transcribe with connection state management."""
        if self._ws_client and self._ws_client.open and self._receive_task:
            logger.debug(f"{self} Already connected")
            return

        async with self._connection_lock:
            if self._connecting:
                logger.debug(f"{self} Connection already in progress")
                return

            try:
                self._connecting = True
                logger.debug(f"{self} Starting connection process...")

                if self._ws_client:
                    await self._disconnect()

                language_code = self.language_to_service_language(
                    Language(self._settings["language"])
                )
                if not language_code:
                    raise ValueError(f"Unsupported language: {self._settings['language']}")

                # Generate random websocket key
                websocket_key = "".join(
                    random.choices(
                        string.ascii_uppercase + string.ascii_lowercase + string.digits, k=20
                    )
                )

                # Add required headers
                extra_headers = {
                    "Origin": "https://localhost",
                    "Sec-WebSocket-Key": websocket_key,
                    "Sec-WebSocket-Version": "13",
                    "Connection": "keep-alive",
                }

                # Get presigned URL
                presigned_url = get_presigned_url(
                    region=self._credentials["region"],
                    credentials={
                        "access_key": self._credentials["aws_access_key_id"],
                        "secret_key": self._credentials["aws_secret_access_key"],
                        "session_token": self._credentials["aws_session_token"],
                    },
                    language_code=language_code,
                    media_encoding=self.get_service_encoding(
                        self._settings["media_encoding"]
                    ),  # Convert to AWS format
                    sample_rate=self._settings["sample_rate"],
                    number_of_channels=self._settings["number_of_channels"],
                    enable_partial_results_stabilization=True,
                    partial_results_stability="high",
                    show_speaker_label=self._settings["show_speaker_label"],
                    enable_channel_identification=self._settings["enable_channel_identification"],
                )

                logger.debug(f"{self} Connecting to WebSocket with URL: {presigned_url[:100]}...")

                # Connect with the required headers and settings
                self._ws_client = await websockets.connect(
                    presigned_url,
                    extra_headers=extra_headers,
                    subprotocols=["mqtt"],
                    ping_interval=None,
                    ping_timeout=None,
                    compression=None,
                )

                logger.debug(f"{self} WebSocket connected, starting receive task...")

                # Start receive task
                self._receive_task = self.create_task(self._receive_loop())

                logger.info(f"{self} Successfully connected to AWS Transcribe")

            except Exception as e:
                logger.error(f"{self} Failed to connect to AWS Transcribe: {e}")
                await self._disconnect()
                raise

            finally:
                self._connecting = False

    async def _disconnect(self):
        """Disconnect from AWS Transcribe."""
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        try:
            if self._ws_client and self._ws_client.open:
                # Send end-stream message
                end_stream = {"message-type": "event", "event": "end"}
                await self._ws_client.send(json.dumps(end_stream))
            await self._ws_client.close()
        except Exception as e:
            logger.warning(f"{self} Error closing WebSocket connection: {e}")
        finally:
            self._ws_client = None

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert internal language enum to AWS Transcribe language code."""
        language_map = {
            Language.EN: "en-US",
            Language.ES: "es-US",
            Language.FR: "fr-FR",
            Language.DE: "de-DE",
            Language.IT: "it-IT",
            Language.PT: "pt-BR",
            Language.JA: "ja-JP",
            Language.KO: "ko-KR",
            Language.ZH: "zh-CN",
            Language.PL: "pl-PL",
        }
        return language_map.get(language)

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[str] = None
    ):
        pass

    async def _receive_loop(self):
        """Background task to receive and process messages from AWS Transcribe."""
        while True:
            if not self._ws_client or not self._ws_client.open:
                logger.warning(f"{self} WebSocket closed in receive loop")
                break

            try:
                response = await asyncio.wait_for(self._ws_client.recv(), timeout=1.0)

                headers, payload = decode_event(response)

                if headers.get(":message-type") == "event":
                    # Process transcription results
                    results = payload.get("Transcript", {}).get("Results", [])
                    if results:
                        result = results[0]
                        alternatives = result.get("Alternatives", [])
                        if alternatives:
                            transcript = alternatives[0].get("Transcript", "")
                            is_final = not result.get("IsPartial", True)

                            if transcript:
                                await self.stop_ttfb_metrics()
                                if is_final:
                                    await self.push_frame(
                                        TranscriptionFrame(
                                            transcript,
                                            "",
                                            time_now_iso8601(),
                                            self._settings["language"],
                                            result=result,
                                        )
                                    )
                                    await self._handle_transcription(
                                        transcript,
                                        is_final,
                                        self._settings["language"],
                                    )
                                    await self.stop_processing_metrics()
                                else:
                                    await self.push_frame(
                                        InterimTranscriptionFrame(
                                            transcript,
                                            "",
                                            time_now_iso8601(),
                                            self._settings["language"],
                                            result=result,
                                        )
                                    )
                elif headers.get(":message-type") == "exception":
                    error_msg = payload.get("Message", "Unknown error")
                    logger.error(f"{self} Exception from AWS: {error_msg}")
                    await self.push_frame(
                        ErrorFrame(f"AWS Transcribe error: {error_msg}", fatal=False)
                    )
                else:
                    logger.debug(f"{self} Other message type received: {headers}")
                    logger.debug(f"{self} Payload: {payload}")
            except asyncio.TimeoutError:
                self.reset_watchdog()
            except websockets.exceptions.ConnectionClosed as e:
                logger.error(
                    f"{self} WebSocket connection closed in receive loop with code {e.code}: {e.reason}"
                )
                break
            except Exception as e:
                logger.error(f"{self} Unexpected error in receive loop: {e}")
                break
