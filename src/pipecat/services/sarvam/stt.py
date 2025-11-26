"""Sarvam AI Speech-to-Text service implementation.

This module provides a streaming Speech-to-Text service using Sarvam AI's WebSocket-based
API. It supports real-time transcription with Voice Activity Detection (VAD) and
can handle multiple audio formats for Indian language speech recognition.
"""

import base64
from typing import Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from sarvamai import AsyncSarvamAI
    from sarvamai.core.api_error import ApiError
    from sarvamai.core.events import EventType
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Sarvam, you need to `pip install pipecat-ai[sarvam]`.")
    raise Exception(f"Missing module: {e}")


def language_to_sarvam_language(language: Language) -> str:
    """Convert a Language enum to Sarvam's language code format.

    Args:
        language: The Language enum value to convert.

    Returns:
        The Sarvam language code string.
    """
    # Mapping of pipecat Language enum to Sarvam language codes
    LANGUAGE_MAP = {
        Language.BN_IN: "bn-IN",
        Language.GU_IN: "gu-IN",
        Language.HI_IN: "hi-IN",
        Language.KN_IN: "kn-IN",
        Language.ML_IN: "ml-IN",
        Language.MR_IN: "mr-IN",
        Language.TA_IN: "ta-IN",
        Language.TE_IN: "te-IN",
        Language.PA_IN: "pa-IN",
        Language.OR_IN: "od-IN",
        Language.EN_IN: "en-IN",
        Language.AS_IN: "as-IN",
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=False)


class SarvamSTTService(STTService):
    """Sarvam speech-to-text service.

    Provides real-time speech recognition using Sarvam's WebSocket API.
    """

    class InputParams(BaseModel):
        """Configuration parameters for Sarvam STT service.

        Parameters:
            language: Target language for transcription. Defaults to None (required for saarika models).
            prompt: Optional prompt to guide translation style/context for STT-Translate models.
                   Only applicable to saaras (STT-Translate) models. Defaults to None.
            vad_signals: Enable VAD signals in response. Defaults to True.
            high_vad_sensitivity: Enable high VAD (Voice Activity Detection) sensitivity. Defaults to False.
        """

        language: Optional[Language] = None
        prompt: Optional[str] = None
        vad_signals: bool = True
        high_vad_sensitivity: bool = False

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "saarika:v2.5",
        sample_rate: Optional[int] = None,
        input_audio_codec: str = "wav",
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the Sarvam STT service.

        Args:
            api_key: Sarvam API key for authentication.
            model: Sarvam model to use for transcription.
            sample_rate: Audio sample rate. Defaults to 16000 if not specified.
            input_audio_codec: Audio codec/format of the input file. Defaults to "wav".
            params: Configuration parameters for Sarvam STT service.
            **kwargs: Additional arguments passed to the parent STTService.
        """
        params = params or SarvamSTTService.InputParams()

        # Validate that saaras models don't accept language parameter
        if "saaras" in model.lower():
            if params.language is not None:
                raise ValueError(
                    f"Model '{model}' does not accept language parameter. "
                    "STT-Translate models auto-detect language."
                )

        # Validate that saarika models don't accept prompt parameter
        if "saarika" in model.lower():
            if params.prompt is not None:
                raise ValueError(
                    f"Model '{model}' does not accept prompt parameter. "
                    "Prompts are only supported for STT-Translate models"
                )

        super().__init__(sample_rate=sample_rate, **kwargs)

        self.set_model_name(model)
        self._api_key = api_key
        self._language_code = params.language
        # For saarika models, default to "unknown" if language is not provided
        if params.language:
            self._language_string = language_to_sarvam_language(params.language)
        elif "saarika" in model.lower():
            self._language_string = "unknown"
        else:
            self._language_string = None
        self._prompt = params.prompt

        # Store connection parameters
        self._vad_signals = params.vad_signals
        self._high_vad_sensitivity = params.high_vad_sensitivity
        self._input_audio_codec = input_audio_codec

        # Initialize Sarvam SDK client
        self._sarvam_client = AsyncSarvamAI(api_subscription_key=api_key)
        self._websocket_context = None
        self._socket_client = None
        self._receive_task = None

    def language_to_service_language(self, language: Language) -> str:
        """Convert pipecat Language enum to Sarvam's language code.

        Args:
            language: The Language enum value to convert.

        Returns:
            The Sarvam language code string.
        """
        return language_to_sarvam_language(language)

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Sarvam service supports metrics generation.
        """
        return True

    async def set_language(self, language: Language):
        """Set the recognition language and reconnect.

        Args:
            language: The language to use for speech recognition.
        """
        # saaras models do not accept a language parameter
        if "saaras" in self.model_name.lower():
            raise ValueError(
                f"Model '{self.model_name}' (saaras) does not accept language parameter. "
                "saaras models auto-detect language."
            )

        logger.info(f"Switching STT language to: [{language}]")
        self._language_code = language
        self._language_string = language_to_sarvam_language(language)
        await self._disconnect()
        await self._connect()

    async def set_prompt(self, prompt: Optional[str]):
        """Set the translation prompt and reconnect.

        Args:
            prompt: Prompt text to guide translation style/context.
                   Pass None to clear/disable prompt.
                   Only applicable to STT-Translate models, not STT models.
        """
        # saarika models do not accept prompt parameter
        if "saarika" in self.model_name.lower():
            if prompt is not None:
                raise ValueError(
                    f"Model '{self.model_name}' does not accept prompt parameter. "
                    "Prompts are only supported for STT-Translate models."
                )
            # If prompt is None and it's saarika, just silently return (no-op)
            return

        logger.info("Updating STT-Translate prompt.")
        self._prompt = prompt
        await self._disconnect()
        await self._connect()

    async def start(self, frame: StartFrame):
        """Start the Sarvam STT service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Sarvam STT service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Sarvam STT service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes):
        """Send audio data to Sarvam for transcription.

        Args:
            audio: Raw audio bytes to transcribe.

        Yields:
            Frame: None (transcription results come via WebSocket callbacks).
        """
        if not self._socket_client:
            logger.warning("WebSocket not connected, cannot process audio")
            yield None
            return

        try:
            # Convert audio bytes to base64 for Sarvam API
            audio_base64 = base64.b64encode(audio).decode("utf-8")

            # Convert input_audio_codec to encoding format (prepend "audio/" if needed)
            encoding = (
                self._input_audio_codec
                if self._input_audio_codec.startswith("audio/")
                else f"audio/{self._input_audio_codec}"
            )

            # Build method arguments
            method_kwargs = {
                "audio": audio_base64,
                "encoding": encoding,
                "sample_rate": self.sample_rate,
            }

            # Use appropriate method based on service type
            if "saarika" in self.model_name.lower():
                # STT service
                await self._socket_client.transcribe(**method_kwargs)
            else:
                # STT-Translate service - auto-detects input language and returns translated text
                await self._socket_client.translate(**method_kwargs)

        except Exception as e:
            yield ErrorFrame(error=f"Error sending audio to Sarvam: {e}", exception=e)

        yield None

    async def _connect(self):
        """Connect to Sarvam WebSocket API using the SDK."""
        logger.debug("Connecting to Sarvam")

        try:
            # Convert boolean parameters to string for SDK
            vad_signals_str = "true" if self._vad_signals else "false"
            high_vad_sensitivity_str = "true" if self._high_vad_sensitivity else "false"

            # Build common connection parameters
            connect_kwargs = {
                "model": self.model_name,
                "vad_signals": vad_signals_str,
                "high_vad_sensitivity": high_vad_sensitivity_str,
                "input_audio_codec": self._input_audio_codec,
                "sample_rate": str(self.sample_rate),
            }

            # Choose the appropriate service based on model
            if "saarika" in self.model_name.lower():
                # STT service - requires language_code
                connect_kwargs["language_code"] = self._language_string
                self._websocket_context = self._sarvam_client.speech_to_text_streaming.connect(
                    **connect_kwargs
                )
            else:
                # STT-Translate service - auto-detects input language and returns translated text
                self._websocket_context = (
                    self._sarvam_client.speech_to_text_translate_streaming.connect(**connect_kwargs)
                )

            # Enter the async context manager
            self._socket_client = await self._websocket_context.__aenter__()

            # Set prompt if provided (only for STT-Translate models, after connection)
            if self._prompt is not None and "saaras" in self.model_name.lower():
                await self._socket_client.set_prompt(self._prompt)

            # Register event handler for incoming messages
            def _message_handler(message):
                """Wrapper to handle async response handler."""
                # Use Pipecat's built-in task management
                self.create_task(self._handle_message(message))

            self._socket_client.on(EventType.MESSAGE, _message_handler)

            # Start receive task using Pipecat's task management
            self._receive_task = self.create_task(self._receive_task_handler())

            logger.info("Connected to Sarvam successfully")

        except ApiError as e:
            await self.push_error(error_msg=f"Sarvam API error: {e}", exception=e)
        except Exception as e:
            self._socket_client = None
            self._websocket_context = None
            await self.push_error(error_msg=f"Failed to connect to Sarvam: {e}", exception=e)

    async def _disconnect(self):
        """Disconnect from Sarvam WebSocket API using SDK."""
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        if self._websocket_context and self._socket_client:
            try:
                # Exit the async context manager
                await self._websocket_context.__aexit__(None, None, None)
            except Exception as e:
                await self.push_error(
                    error_msg=f"Error closing WebSocket connection: {e}", exception=e
                )
            finally:
                logger.debug("Disconnected from Sarvam WebSocket")
                self._socket_client = None
                self._websocket_context = None

    async def _receive_task_handler(self):
        """Handle incoming messages from Sarvam WebSocket.

        This task wraps the SDK's start_listening() method which processes
        messages via the registered event handler callback.
        """
        if not self._socket_client:
            return

        try:
            # Start listening for messages from the Sarvam SDK
            # Messages will be handled via the _message_handler callback
            await self._socket_client.start_listening()
        except Exception as e:
            await self.push_error(error_msg=f"Sarvam receive task error: {e}", exception=e)

    async def _handle_message(self, message):
        """Handle incoming WebSocket message from Sarvam SDK.

        Processes transcription data and VAD events from the Sarvam service.

        Args:
            message: The parsed response object from Sarvam WebSocket.
        """
        logger.debug(f"Received response: {message}")

        try:
            if message.type == "events":
                # VAD event
                signal = message.data.signal_type
                timestamp = message.data.occured_at
                logger.debug(f"VAD Signal: {signal}, Occurred at: {timestamp}")

                if signal == "START_SPEECH":
                    await self.start_metrics()
                    logger.debug("User started speaking")
                    await self._call_event_handler("on_speech_started")

            elif message.type == "data":
                await self.stop_ttfb_metrics()
                transcript = message.data.transcript
                language_code = message.data.language_code
                # Prefer language from message (auto-detected for translate models). Fallback to configured.
                if language_code:
                    language = self._map_language_code_to_enum(language_code)
                elif self._language_string:
                    language = self._map_language_code_to_enum(self._language_string)
                else:
                    language = Language.HI_IN

                # Emit utterance end event
                await self._call_event_handler("on_utterance_end")

                if transcript and transcript.strip():
                    # Record tracing for this transcription event
                    await self._handle_transcription(transcript, True, language)
                    await self.push_frame(
                        TranscriptionFrame(
                            transcript,
                            self._user_id,
                            time_now_iso8601(),
                            language,
                            result=(message.dict() if hasattr(message, "dict") else str(message)),
                        )
                    )

                await self.stop_processing_metrics()

        except Exception as e:
            await self.push_error(error_msg=f"Failed to handle message: {e}", exception=e)
            await self.stop_all_metrics()

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing.

        This method is decorated with @traced_stt for observability.
        """
        pass

    def _map_language_code_to_enum(self, language_code: str) -> Language:
        """Map Sarvam language code to pipecat Language enum."""
        mapping = {
            "bn-IN": Language.BN_IN,
            "gu-IN": Language.GU_IN,
            "hi-IN": Language.HI_IN,
            "kn-IN": Language.KN_IN,
            "ml-IN": Language.ML_IN,
            "mr-IN": Language.MR_IN,
            "ta-IN": Language.TA_IN,
            "te-IN": Language.TE_IN,
            "pa-IN": Language.PA_IN,
            "od-IN": Language.OR_IN,
            "en-US": Language.EN_US,
            "en-IN": Language.EN_IN,
            "as-IN": Language.AS_IN,
        }
        return mapping.get(language_code, Language.HI_IN)

    async def start_metrics(self):
        """Start TTFB and processing metrics collection."""
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()
