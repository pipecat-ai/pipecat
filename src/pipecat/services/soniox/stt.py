#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import json
import time
from typing import AsyncGenerator, List, Optional

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.soniox.config import SonioxInputParams
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Soniox, you need to `pip install pipecat-ai[soniox]`.")
    raise Exception(f"Missing module: {e}")


KEEPALIVE_MESSAGE = '{"type": "keepalive"}'

FINALIZE_MESSAGE = '{"type": "finalize"}'

END_TOKEN = "<end>"

FINALIZED_TOKEN = "<fin>"


def is_end_token(token: dict) -> bool:
    return token["text"] == END_TOKEN or token["text"] == FINALIZED_TOKEN


def language_to_soniox_language(language: Language) -> str:
    """Pipecat Language enum uses same ISO 2-letter codes as Soniox, except with added regional variants.

    For a list of all supported languages, see: https://soniox.com/docs/speech-to-text/core-concepts/supported-languages
    """
    lang_str = str(language.value).lower()
    if "-" in lang_str:
        return lang_str.split("-")[0]
    return lang_str


def _prepare_language_hints(
    language_hints: Optional[List[Language]],
) -> Optional[List[str]]:
    if language_hints is None:
        return None

    prepared_languages = [language_to_soniox_language(lang) for lang in language_hints]
    # Remove duplicates (in case of language_hints with multiple regions).
    return list(set(prepared_languages))


class SonioxSTTService(STTService):
    """Speech-to-Text service using Soniox's WebSocket API.

    This service connects to Soniox's WebSocket API for real-time transcription
    with support for multiple languages, custom context, speaker diarization,
    and more.

    For complete API documentation, see: https://soniox.com/docs/speech-to-text/api-reference/websocket-api
    """

    def __init__(
        self,
        *,
        api_key: str,
        url: str = "wss://stt-rt.soniox.com/transcribe-websocket",
        sample_rate: Optional[int] = None,
        params: SonioxInputParams = SonioxInputParams(),
        enable_vad: bool = True,
        auto_finalize_delay_ms: Optional[int] = 3000,
        **kwargs,
    ):
        """Initialize the Soniox STT service.

        Args:
            api_key: Soniox API key.
            url: Soniox WebSocket API URL.
            model: Transcription model to use.
            params: Additional configuration parameters, such as language hints, context and
                speaker diarization.
            enable_vad: Listen to `UserStoppedSpeakingFrame` to send finalize message to Soniox.
            auto_finalize_delay_ms: If no new tokens are received for a while and there is active
                transcription (only InterimTranscriptionFrame), finalize the transcription by
                sending the finalize message so user can receive the final transcript. If set
                to `None`, the auto finalize feature is disabled.
            **kwargs: Additional arguments passed to the STTService.
        """
        sample_rate = sample_rate or (params.sample_rate if params.sample_rate else None)
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = api_key
        self._url = url
        self.set_model_name(params.model)
        self._params = params
        self._enable_vad = enable_vad
        self._auto_finalize_delay_ms = auto_finalize_delay_ms
        self._websocket = None

        self._final_transcription_buffer = ""
        self._last_tokens_received: Optional[float] = None

        self._receive_task = None
        self._keepalive_task = None
        self._finalize_if_no_tokens_task = None

    async def start(self, frame: StartFrame):
        """Start the Soniox STT websocket connection."""
        await super().start(frame)
        if self._websocket:
            return

        self._websocket = await websockets.connect(self._url)

        if not self._websocket:
            logger.error(f"Unable to connect to Soniox API at {self._url}")

        # Send the initial configuration message.
        config = {
            "api_key": self._api_key,
            "model": self._model_name,
            "audio_format": self._params.audio_format,
            "num_channels": self._params.num_channels or 1,
            "enable_endpoint_detection": self._params.enable_endpoint_detection,
            "sample_rate": self._sample_rate,
            "language_hints": _prepare_language_hints(self._params.language_hints),
            "context": self._params.context,
            "enable_non_final_tokens": self._params.enable_non_final_tokens,
            "max_non_final_tokens_duration_ms": self._params.max_non_final_tokens_duration_ms,
            "client_reference_id": self._params.client_reference_id,
        }

        # Send the configuration message.
        await self._websocket.send(json.dumps(config))

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler())
        if self._websocket and not self._keepalive_task:
            self._keepalive_task = self.create_task(self._keepalive_task_handler())
        if (
            self._websocket
            and not self._finalize_if_no_tokens_task
            and self._auto_finalize_delay_ms is not None
        ):
            self._finalize_if_no_tokens_task = self.create_task(
                self._finalize_if_no_tokens_task_handler()
            )

    async def _cleanup(self):
        if self._keepalive_task:
            await self.cancel_task(self._keepalive_task)
            self._keepalive_task = None

        if self._websocket:
            await self._websocket.close()
            self._websocket = None

        if self._receive_task:
            await self.wait_for_task(self._receive_task)
            self._receive_task = None

        if self._finalize_if_no_tokens_task:
            await self.cancel_task(self._finalize_if_no_tokens_task)
            self._finalize_if_no_tokens_task = None

    async def stop(self, frame: EndFrame):
        """Stop the Soniox STT websocket connection.

        Stopping waits for the server to close the connection as we might receive
        additional final tokens after sending the stop recording message.
        """
        await super().stop(frame)
        await self._send_stop_recording()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Soniox STT websocket connection.

        Compared to stop, this method closes the connection immediately without waiting
        for the server to close it. This is useful when we want to stop the connection
        immediately without waiting for the server to send any final tokens.
        """
        await super().cancel(frame)
        await self._cleanup()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Send audio data to Soniox STT Service."""
        await self.start_processing_metrics()
        if self._websocket and not self._websocket.closed:
            await self._websocket.send(audio)
        await self.stop_processing_metrics()

        yield None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Processes a frame of audio data, either buffering or transcribing it."""
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStoppedSpeakingFrame) and self._enable_vad:
            # Send finalize message to Soniox so we get the final tokens asap.
            if self._websocket and not self._websocket.closed:
                await self._websocket.send(FINALIZE_MESSAGE)
                logger.debug(f"Triggered finalize event on: {frame.name=}, {direction=}")

    async def _send_stop_recording(self):
        if self._websocket and not self._websocket.closed:
            # Send stop recording message
            await self._websocket.send("")

    async def _keepalive_task_handler(self):
        """Connection has to be open all the time."""
        try:
            while True:
                logger.debug("Sending keepalive message")
                if self._websocket and not self._websocket.closed:
                    await self._websocket.send(KEEPALIVE_MESSAGE)
                else:
                    logger.debug("WebSocket connection closed.")
                    break
                await asyncio.sleep(5)

        except websockets.exceptions.ConnectionClosed:
            # Expected when closing the connection
            logger.debug("WebSocket connection closed, keepalive task stopped.")
        except Exception as e:
            logger.error(f"{self} error (_keepalive_task_handler): {e}")
            await self.push_error(ErrorFrame(f"{self} error (_keepalive_task_handler): {e}"))

    async def _receive_task_handler(self):
        if not self._websocket:
            return

        # Transcription frame will be only sent after we get the "endpoint" event.
        self._final_transcription_buffer = ""

        async def send_endpoint_transcript():
            if self._final_transcription_buffer:
                await self.push_frame(
                    TranscriptionFrame(
                        self._final_transcription_buffer,
                        "",
                        time_now_iso8601(),
                    )
                )
                self._final_transcription_buffer = ""

        try:
            async for message in self._websocket:
                content = json.loads(message)

                tokens = content["tokens"]

                if tokens:
                    # Got at least one token, so we can reset the auto finalize delay.
                    self._last_tokens_received = time.time()

                # We will only send the final tokens after we get the "endpoint" event.
                non_final_transcription = ""

                for token in tokens:
                    if token["is_final"]:
                        if is_end_token(token):
                            # Found an endpoint, tokens until here will be sent as transcript,
                            # the rest will be sent as interim tokens (even final tokens).
                            await send_endpoint_transcript()
                        else:
                            self._final_transcription_buffer += token["text"]
                    else:
                        non_final_transcription += token["text"]

                if self._final_transcription_buffer or non_final_transcription:
                    await self.push_frame(
                        InterimTranscriptionFrame(
                            # Even final tokens are sent as interim tokens as we want to send
                            # nicely formatted messages - therefore waiting for the endpoint.
                            self._final_transcription_buffer + non_final_transcription,
                            "",
                            time_now_iso8601(),
                        )
                    )

                error_code = content.get("error_code")
                error_message = content.get("error_message")
                if error_code or error_message:
                    # In case of error, still send the final transcript (if any remaining in the buffer).
                    await send_endpoint_transcript()
                    logger.error(
                        f"{self} error: {error_code} (_receive_task_handler) - {error_message}"
                    )
                    await self.push_error(
                        ErrorFrame(
                            f"{self} error: {error_code} (_receive_task_handler) - {error_message}"
                        )
                    )

                finished = content.get("finished")
                if finished:
                    # When finished, still send the final transcript (if any remaining in the buffer).
                    await send_endpoint_transcript()
                    logger.debug("Transcription finished.")
                    await self._cleanup()

        except websockets.exceptions.ConnectionClosed:
            # Expected when closing the connection.
            pass
        except Exception as e:
            logger.error(f"{self} error: {e}")
            await self.push_error(ErrorFrame(f"{self} error: {e}"))

    async def _finalize_if_no_tokens_task_handler(self):
        """Call finalize if no new tokens are received for a configured duration."""
        if not self._websocket or self._websocket.closed or self._auto_finalize_delay_ms is None:
            return

        try:
            while True:
                await asyncio.sleep(0.5)

                if not self._websocket or self._websocket.closed:
                    break

                # Check if we have anything to send.
                if not self._final_transcription_buffer:
                    continue

                # Check if enough time has passed since the last tokens were received.
                if self._last_tokens_received:
                    last_token_age_ms = (time.time() - self._last_tokens_received) * 1000

                    if last_token_age_ms > self._auto_finalize_delay_ms:
                        # No new tokens received for a while, finalize the transcription.
                        logger.debug("No pending frames, sending finalize message")
                        self._last_tokens_received = None
                        await self._websocket.send(FINALIZE_MESSAGE)
        except websockets.exceptions.ConnectionClosed:
            # Expected when closing the connection.
            pass
        except Exception as e:
            logger.error(f"{self} error (_finalize_if_no_tokens_task_handler): {e}")
            await self.push_error(
                ErrorFrame(f"{self} error (_finalize_if_no_tokens_task_handler): {e}")
            )
