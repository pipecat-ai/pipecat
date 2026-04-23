#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""RTVIProcessor: main RTVI protocol processor."""

import asyncio
import base64
from collections.abc import Mapping
from typing import Any

from loguru import logger
from pydantic import BaseModel, ValidationError

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat import version as pipecat_version
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    EndTaskFrame,
    ErrorFrame,
    Frame,
    FunctionCallResultFrame,
    InputAudioRawFrame,
    InputTextRawFrame,
    InputTransportMessageFrame,
    LLMConfigureOutputFrame,
    LLMMessagesAppendFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
    SystemFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.frames import RTVIClientMessageFrame
from pipecat.processors.frameworks.rtvi.observer import RTVIObserver, RTVIObserverParams
from pipecat.services.llm_service import (
    FunctionCallParams,  # TODO(aleix): we shouldn't import `services` from `processors`
)
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_transport import BaseTransport


class RTVIProcessor(FrameProcessor):
    """Main processor for handling RTVI protocol messages and actions.

    This processor manages the RTVI protocol communication including client-server
    handshaking, configuration management, action execution, and message routing.
    It serves as the central hub for RTVI protocol operations.
    """

    def __init__(
        self,
        *,
        transport: BaseTransport | None = None,
        **kwargs,
    ):
        """Initialize the RTVI processor.

        Args:
            transport: Transport layer for communication.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)

        self._bot_ready = False
        self._client_ready = False
        self._client_ready_id = ""
        # Default to 0.0.0 to indicate unknown version.
        self._client_version = [0, 0, 0]
        self._llm_skip_tts: bool = False  # Keep in sync with llm_service.py's configuration.

        # A task to process incoming transport messages.
        self._message_task: asyncio.Task | None = None

        self._register_event_handler("on_bot_started")
        self._register_event_handler("on_client_ready")
        self._register_event_handler("on_client_message")

        self._input_transport = None
        self._transport = transport
        if self._transport:
            input_transport = self._transport.input()
            if isinstance(input_transport, BaseInputTransport):
                self._input_transport = input_transport
                self._input_transport.enable_audio_in_stream_on_start(False)

    def create_rtvi_observer(self, *, params: RTVIObserverParams | None = None, **kwargs):
        """Creates a new RTVI Observer.

        Args:
            params: Settings to enable/disable specific messages.
            **kwargs: Additional arguments passed to the observer.

        Returns:
            A new RTVI observer.
        """
        return RTVIObserver(self, params=params, **kwargs)

    async def set_client_ready(self):
        """Mark the client as ready and trigger the ready event."""
        self._client_ready = True
        await self._call_event_handler("on_client_ready")

    async def set_bot_ready(self, about: Mapping[str, Any] = None):
        """Mark the bot as ready and send the bot-ready message.

        Args:
            about: Optional information about the bot to include in the ready message.
                   If left as None, the Pipecat library and version will be used.
        """
        self._bot_ready = True
        await self._send_bot_ready(about=about)

    async def interrupt_bot(self):
        """Send a bot interruption frame upstream."""
        await self.broadcast_interruption()

    async def send_server_message(self, data: Any):
        """Send a server message to the client."""
        message = RTVI.ServerMessage(data=data)
        await self._send_server_message(message)

    async def send_server_response(self, client_msg: RTVI.ClientMessage, data: Any):
        """Send a server response for a given client message."""
        message = RTVI.ServerResponse(
            id=client_msg.msg_id, data=RTVI.RawServerResponseData(t=client_msg.type, d=data)
        )
        await self._send_server_message(message)

    async def send_error_response(self, client_msg: RTVI.ClientMessage, error: str):
        """Send an error response for a given client message."""
        await self._send_error_response(id=client_msg.msg_id, error=error)

    async def send_error(self, error: str):
        """Send an error message to the client.

        Args:
            error: The error message to send.
        """
        await self._send_error_frame(ErrorFrame(error=error))

    async def push_transport_message(self, model: BaseModel, exclude_none: bool = True):
        """Push a transport message frame."""
        frame = OutputTransportMessageUrgentFrame(
            message=model.model_dump(exclude_none=exclude_none)
        )
        await self.push_frame(frame)

    async def handle_message(self, message: RTVI.Message):
        """Handle an incoming RTVI message.

        Args:
            message: The RTVI message to handle.
        """
        await self._message_queue.put(message)

    async def handle_function_call(self, params: FunctionCallParams):
        """Handle a function call from the LLM.

        Args:
            params: The function call parameters.

        .. deprecated:: 0.0.102
            This method is deprecated. Function call events are now automatically
            sent by ``RTVIObserver`` using the ``llm-function-call-in-progress`` event.
            Configure reporting level via ``RTVIObserverParams.function_call_report_level``.
        """
        import warnings

        warnings.warn(
            "handle_function_call is deprecated. Function call events are now "
            "automatically sent by RTVIObserver using llm-function-call-in-progress.",
            DeprecationWarning,
            stacklevel=2,
        )
        fn = RTVI.LLMFunctionCallMessageData(
            function_name=params.function_name,
            tool_call_id=params.tool_call_id,
            args=params.arguments,
        )
        message = RTVI.LLMFunctionCallMessage(data=fn)
        await self.push_transport_message(message, exclude_none=False)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames through the RTVI processor.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
        await super().process_frame(frame, direction)

        # Specific system frames
        if isinstance(frame, StartFrame):
            # Push StartFrame before start(), because we want StartFrame to be
            # processed by every processor before any other frame is processed.
            await self.push_frame(frame, direction)
            await self._start(frame)
        elif isinstance(frame, CancelFrame):
            await self._cancel(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, ErrorFrame):
            await self._send_error_frame(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, InputTransportMessageFrame):
            await self._handle_transport_message(frame)
        # All other system frames
        elif isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
        # Control frames
        elif isinstance(frame, EndFrame):
            # Push EndFrame before stop(), because stop() waits on the task to
            # finish and the task finishes when EndFrame is processed.
            await self.push_frame(frame, direction)
            await self._stop(frame)
        # Data frames
        elif isinstance(frame, LLMConfigureOutputFrame):
            self._llm_skip_tts = frame.skip_tts
            await self.push_frame(frame, direction)
        # Other frames
        else:
            await self.push_frame(frame, direction)

    async def _start(self, frame: StartFrame):
        """Start the RTVI processor tasks."""
        if not self._message_task:
            self._message_queue = asyncio.Queue()
            self._message_task = self.create_task(self._message_task_handler())
        await self._call_event_handler("on_bot_started")

    async def _stop(self, frame: EndFrame):
        """Stop the RTVI processor tasks."""
        await self._cancel_tasks()

    async def _cancel(self, frame: CancelFrame):
        """Cancel the RTVI processor tasks."""
        await self._cancel_tasks()

    async def _cancel_tasks(self):
        """Cancel all running tasks."""
        if self._message_task:
            await self.cancel_task(self._message_task)
            self._message_task = None

    async def _message_task_handler(self):
        """Handle incoming transport messages."""
        while True:
            message = await self._message_queue.get()
            await self._handle_message(message)
            self._message_queue.task_done()

    async def _handle_transport_message(self, frame: InputTransportMessageFrame):
        """Handle an incoming transport message frame."""
        try:
            transport_message = frame.message
            if transport_message.get("label") != RTVI.MESSAGE_LABEL:
                logger.warning(f"Ignoring not RTVI message: {transport_message}")
                return
            message = RTVI.Message.model_validate(transport_message)
            await self._message_queue.put(message)
        except ValidationError as e:
            await self.send_error(f"Invalid RTVI transport message: {e}")
            logger.warning(f"Invalid RTVI transport message: {e}")

    async def _handle_message(self, message: RTVI.Message):
        """Handle a parsed RTVI message."""
        try:
            match message.type:
                case "client-ready":
                    data = None
                    raw = message.data or {}
                    version = raw.get("version")
                    if isinstance(version, str):
                        about = RTVI.AboutClientData(library="unknown")
                        about_raw = raw.get("about")
                        if about_raw is not None:
                            try:
                                about = RTVI.AboutClientData.model_validate(about_raw)
                            except ValidationError:
                                logger.warning(
                                    "Invalid 'about' data in client-ready message, ignoring."
                                )
                        data = RTVI.ClientReadyData(version=version, about=about)
                    await self._handle_client_ready(message.id, data)
                case "disconnect-bot":
                    await self.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
                case "client-message":
                    data = RTVI.RawClientMessageData.model_validate(message.data)
                    await self._handle_client_message(message.id, data)
                case "llm-function-call-result":
                    data = RTVI.LLMFunctionCallResultData.model_validate(message.data)
                    await self._handle_function_call_result(data)
                case "send-text":
                    data = RTVI.SendTextData.model_validate(message.data)
                    await self._handle_send_text(data)
                case "raw-audio" | "raw-audio-batch":
                    await self._handle_audio_buffer(message.data)

                case _:
                    await self._send_error_response(message.id, f"Unsupported type {message.type}")

        except ValidationError as e:
            await self._send_error_response(message.id, f"Invalid message: {e}")
            logger.warning(f"Invalid message: {e}")
        except Exception as e:
            await self._send_error_response(message.id, f"Exception processing message: {e}")
            logger.warning(f"Exception processing message: {e}")

    async def _handle_client_ready(self, request_id: str, data: RTVI.ClientReadyData | None):
        """Handle the client-ready message from the client."""
        version = data.version if data else None
        logger.debug(f"Received client-ready: version {version}")
        version_error = None
        if version:
            try:
                parts = [int(v) for v in version.split(".")]
                if len(parts) != 3:
                    raise ValueError
                self._client_version = parts
                protocol_major = int(RTVI.PROTOCOL_VERSION.split(".")[0])
                if self._client_version[0] != protocol_major:
                    version_error = f"RTVI version {version} is not compatible with server protocol {RTVI.PROTOCOL_VERSION}."
            except ValueError:
                version_error = f"Invalid client version format ({version})."
        else:
            version_error = "Client version unknown."
        about = data.about if data else {"library": "unknown"}
        if version_error:
            version_error += " Compatibility issues may occur."
            logger.warning(version_error)
            await self._send_error_response(request_id, version_error)

        logger.debug(f"Client Details: {about}")
        if self._input_transport:
            await self._input_transport.start_audio_in_streaming()

        self._client_ready_id = request_id
        await self.set_client_ready()

    async def _handle_audio_buffer(self, data):
        """Handle incoming audio buffer data."""
        if not self._input_transport:
            return

        # Extract audio batch ensuring it's a list
        audio_list = data.get("base64AudioBatch") or [data.get("base64Audio")]

        try:
            for base64_audio in filter(None, audio_list):  # Filter out None values
                pcm_bytes = base64.b64decode(base64_audio)
                frame = InputAudioRawFrame(
                    audio=pcm_bytes,
                    sample_rate=data["sampleRate"],
                    num_channels=data["numChannels"],
                )
                await self._input_transport.push_audio_frame(frame)

        except (KeyError, TypeError, ValueError) as e:
            # Handle missing keys, decoding errors, and invalid types
            logger.error(f"Error processing audio buffer: {e}")

    async def _handle_send_text(self, data: RTVI.SendTextData):
        """Handle a send-text message from the client."""
        opts = data.options if data.options is not None else RTVI.SendTextOptions()
        if opts.run_immediately:
            await self.interrupt_bot()
        cur_llm_skip_tts = self._llm_skip_tts
        should_skip_tts = not opts.audio_response
        toggle_skip_tts = cur_llm_skip_tts != should_skip_tts
        if toggle_skip_tts:
            output_frame = LLMConfigureOutputFrame(skip_tts=should_skip_tts)
            await self.push_frame(output_frame)
        text_frame = LLMMessagesAppendFrame(
            messages=[{"role": "user", "content": data.content}],
            run_llm=opts.run_immediately,
        )
        await self.push_frame(text_frame)
        # Speech-to-speech LLM services (OpenAI Realtime, AWS Nova Sonic,
        # Gemini Live, etc.) manage their own conversation state on the
        # provider side and do not act on ``LLMContextFrame`` updates the way
        # a text LLM does. Emit an ``InputTextRawFrame`` as well so those
        # services receive the typed user turn and produce a response, matching
        # the STT-LLM-TTS behavior for ``send-text`` (issue #3829).
        if opts.run_immediately:
            await self.push_frame(InputTextRawFrame(text=data.content))
        if toggle_skip_tts:
            output_frame = LLMConfigureOutputFrame(skip_tts=cur_llm_skip_tts)
            await self.push_frame(output_frame)

    async def _handle_client_message(self, msg_id: str, data: RTVI.RawClientMessageData):
        """Handle a client message frame."""
        # Create a RTVIClientMessageFrame to push the message
        frame = RTVIClientMessageFrame(msg_id=msg_id, type=data.t, data=data.d)
        await self.push_frame(frame)
        await self._call_event_handler(
            "on_client_message",
            RTVI.ClientMessage(
                msg_id=msg_id,
                type=data.t,
                data=data.d,
            ),
        )

    async def _handle_function_call_result(self, data):
        """Handle a function call result from the client."""
        frame = FunctionCallResultFrame(
            function_name=data.function_name,
            tool_call_id=data.tool_call_id,
            arguments=data.arguments,
            result=data.result,
        )
        await self.push_frame(frame)

    async def _send_bot_ready(self, about: Mapping[str, Any] = None):
        """Send the bot-ready message to the client.

        Args:
            about: Optional information about the bot to include in the ready message.
                   If left as None, the pipecat library and version will be used.
        """
        if not about:
            about = {"library": "pipecat-ai", "library_version": f"{pipecat_version()}"}
        message = RTVI.BotReady(
            id=self._client_ready_id,
            data=RTVI.BotReadyData(version=RTVI.PROTOCOL_VERSION, about=about),
        )
        await self.push_transport_message(message)

    async def _send_server_message(self, message: RTVI.ServerMessage | RTVI.ServerResponse):
        """Send a message or response to the client."""
        await self.push_transport_message(message)

    async def _send_error_frame(self, frame: ErrorFrame):
        """Send an error frame as an RTVI error message."""
        message = RTVI.Error(data=RTVI.ErrorData(error=frame.error, fatal=frame.fatal))
        await self.push_transport_message(message)

    async def _send_error_response(self, id: str, error: str):
        """Send an error response message."""
        message = RTVI.ErrorResponse(id=id, data=RTVI.ErrorResponseData(error=error))
        await self.push_transport_message(message)
