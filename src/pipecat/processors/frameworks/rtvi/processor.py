#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""RTVIProcessor: main RTVI protocol processor."""

import asyncio
import base64
import io
import os
from typing import Any, Dict, Mapping, Optional

import aiohttp
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
    InputTransportMessageFrame,
    LLMConfigureOutputFrame,
    LLMMessagesAppendFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
    SystemFrame,
    UserFileRawFrame,
    UserImageRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.frames import RTVIActionFrame, RTVIClientMessageFrame
from pipecat.processors.frameworks.rtvi.models_deprecated import (
    RTVIAction,
    RTVIActionResponse,
    RTVIActionResponseData,
    RTVIActionRun,
    RTVIBotReadyDataDeprecated,
    RTVIConfig,
    RTVIConfigResponse,
    RTVIDescribeActions,
    RTVIDescribeActionsData,
    RTVIDescribeConfig,
    RTVIDescribeConfigData,
    RTVIService,
    RTVIServiceConfig,
    RTVIServiceOptionConfig,
    RTVIUpdateConfig,
)
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
        config: Optional[RTVIConfig] = None,
        transport: Optional[BaseTransport] = None,
        uploads_folder: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the RTVI processor.

        Args:
            config: Initial RTVI configuration.
            transport: Transport layer for communication.
            uploads_folder: Path to folder where client uploads (e.g. POST /files) are
                stored; required for send-file with /files/ URLs. Use
                runner_uploads_folder() when using the development runner.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._config = config or RTVIConfig(config=[])
        self._folder = uploads_folder or ""

        self._bot_ready = False
        self._client_ready = False
        self._client_ready_id = ""
        # Default to 0.3.0 which is the last version before actually having a
        # "client-version".
        self._client_version = [0, 3, 0]
        self._llm_skip_tts: bool = False  # Keep in sync with llm_service.py's configuration.

        self._registered_actions: Dict[str, RTVIAction] = {}
        self._registered_services: Dict[str, RTVIService] = {}

        # A task to process incoming action frames.
        self._action_task: Optional[asyncio.Task] = None

        # A task to process incoming transport messages.
        self._message_task: Optional[asyncio.Task] = None

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

    def register_action(self, action: RTVIAction):
        """Register an action that can be executed via RTVI.

        Args:
            action: The action to register.
        """
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "The actions API is deprecated, use server and client messages instead.",
                DeprecationWarning,
            )

        id = self._action_id(action.service, action.action)
        self._registered_actions[id] = action

    def register_service(self, service: RTVIService):
        """Register a service that can be configured via RTVI.

        Args:
            service: The service to register.
        """
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "The actions API is deprecated, use server and client messages instead.",
                DeprecationWarning,
            )

        self._registered_services[service.name] = service

    def create_rtvi_observer(self, *, params: Optional[RTVIObserverParams] = None, **kwargs):
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
        # Only call the (deprecated) _update_config method if the we're using a
        # config (which is deprecated). Otherwise we'd always print an
        # unnecessary deprecation warning.
        if self._config.config:
            await self._update_config(self._config, False)
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
        elif isinstance(frame, RTVIActionFrame):
            await self._action_queue.put(frame)
        elif isinstance(frame, LLMConfigureOutputFrame):
            self._llm_skip_tts = frame.skip_tts
            await self.push_frame(frame, direction)
        # Other frames
        else:
            await self.push_frame(frame, direction)

    async def _start(self, frame: StartFrame):
        """Start the RTVI processor tasks."""
        if not self._action_task:
            self._action_queue = asyncio.Queue()
            self._action_task = self.create_task(self._action_task_handler())
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
        if self._action_task:
            await self.cancel_task(self._action_task)
            self._action_task = None

        if self._message_task:
            await self.cancel_task(self._message_task)
            self._message_task = None

    async def _action_task_handler(self):
        """Handle incoming action frames."""
        while True:
            frame = await self._action_queue.get()
            await self._handle_action(frame.message_id, frame.rtvi_action_run)
            self._action_queue.task_done()

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
                    try:
                        data = RTVI.ClientReadyData.model_validate(message.data)
                    except ValidationError:
                        # Not all clients have been updated to RTVI 1.0.0.
                        # For now, that's okay, we just log their info as unknown.
                        data = None
                        pass
                    await self._handle_client_ready(message.id, data)
                case "describe-actions":
                    await self._handle_describe_actions(message.id)
                case "describe-config":
                    await self._handle_describe_config(message.id)
                case "get-config":
                    await self._handle_get_config(message.id)
                case "update-config":
                    update_config = RTVIUpdateConfig.model_validate(message.data)
                    await self._handle_update_config(message.id, update_config)
                case "disconnect-bot":
                    await self.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
                case "client-message":
                    data = RTVI.RawClientMessageData.model_validate(message.data)
                    await self._handle_client_message(message.id, data)
                case "action":
                    action = RTVIActionRun.model_validate(message.data)
                    action_frame = RTVIActionFrame(message_id=message.id, rtvi_action_run=action)
                    await self._action_queue.put(action_frame)
                case "llm-function-call-result":
                    data = RTVI.LLMFunctionCallResultData.model_validate(message.data)
                    await self._handle_function_call_result(data)
                case "send-text":
                    data = RTVI.SendTextData.model_validate(message.data)
                    await self._handle_send_text(data)
                case "send-file":
                    data = RTVI.SendFileData.model_validate(message.data)
                    await self._handle_send_file(data)
                case "append-to-context":
                    logger.warning(
                        f"The append-to-context message is deprecated, use send-text instead."
                    )
                    data = RTVI.AppendToContextData.model_validate(message.data)
                    await self._handle_update_context(data)
                case "raw-audio" | "raw-audio-batch":
                    await self._handle_audio_buffer(message.data)

                case _:
                    logger.warning(f"Unsupported RTVI message type: {message.type}")
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
        if version:
            try:
                self._client_version = [int(v) for v in version.split(".")]
            except ValueError:
                logger.warning(f"Invalid client version format: {version}")
        about = data.about if data else {"library": "unknown"}
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

    async def _handle_describe_config(self, request_id: str):
        """Handle a describe-config request."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "Configuration helpers are deprecated. If your application needs this behavior, use custom server and client messages.",
                DeprecationWarning,
            )

        services = list(self._registered_services.values())
        message = RTVIDescribeConfig(id=request_id, data=RTVIDescribeConfigData(config=services))
        await self.push_transport_message(message)

    async def _handle_describe_actions(self, request_id: str):
        """Handle a describe-actions request."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "The Actions API is deprecated, use custom server and client messages instead.",
                DeprecationWarning,
            )

        actions = list(self._registered_actions.values())
        message = RTVIDescribeActions(id=request_id, data=RTVIDescribeActionsData(actions=actions))
        await self.push_transport_message(message)

    async def _handle_get_config(self, request_id: str):
        """Handle a get-config request."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "Configuration helpers are deprecated. If your application needs this behavior, use custom server and client messages.",
                DeprecationWarning,
            )

        message = RTVIConfigResponse(id=request_id, data=self._config)
        await self.push_transport_message(message)

    def _update_config_option(self, service: str, config: RTVIServiceOptionConfig):
        """Update a specific configuration option."""
        for service_config in self._config.config:
            if service_config.service == service:
                for option_config in service_config.options:
                    if option_config.name == config.name:
                        option_config.value = config.value
                        return
                # If we couldn't find a value for this config, we simply need to
                # add it.
                service_config.options.append(config)

    async def _update_service_config(self, config: RTVIServiceConfig):
        """Update configuration for a specific service."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "Configuration helpers are deprecated. If your application needs this behavior, use custom server and client messages.",
                DeprecationWarning,
            )

        service = self._registered_services[config.service]
        for option in config.options:
            handler = service._options_dict[option.name].handler
            await handler(self, service.name, option)
            self._update_config_option(service.name, option)

    async def _update_config(self, data: RTVIConfig, interrupt: bool):
        """Update the RTVI configuration."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "Configuration helpers are deprecated. If your application needs this behavior, use custom server and client messages.",
                DeprecationWarning,
            )

        if interrupt:
            await self.interrupt_bot()
        for service_config in data.config:
            await self._update_service_config(service_config)

    async def _handle_update_config(self, request_id: str, data: RTVIUpdateConfig):
        """Handle an update-config request."""
        await self._update_config(RTVIConfig(config=data.config), data.interrupt)
        await self._handle_get_config(request_id)

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
        if toggle_skip_tts:
            output_frame = LLMConfigureOutputFrame(skip_tts=cur_llm_skip_tts)
            await self.push_frame(output_frame)

    async def _handle_send_file(self, data: RTVI.SendFileData):
        """Handle a send-file message from the client."""
        file = data.file
        source = None
        type = file.source.type
        opts = data.options if data.options is not None else RTVI.SendFileOptions()

        match type:
            case "bytes":
                source = file.source.bytes
            case "url":
                if not file.source.public:
                    # read bytes from URL and encode to base64
                    type = "bytes"
                    async with aiohttp.ClientSession() as session:
                        async with session.get(file.source.url) as response:
                            content = io.BytesIO(await response.content.read())
                            base64_string = base64.b64encode(content.getvalue()).decode("utf-8")
                            source = f"data:{file.format};base64,{base64_string}"
                else:
                    source = file.source.url
            case "id":
                if not file.source.id.startswith("pipecat:"):
                    logger.warning(f"Unsupported file ID: {file.source.id}")
                    self.send_error_response(data.id, f"Unsupported file ID: {file.source.id}")
                    return
                if not self._folder:
                    logger.warning(
                        "Send-file with a pipecat id requires uploads_folder on RTVIProcessor "
                        "(e.g. uploads_folder=runner_uploads_folder())."
                    )
                    self.send_error_response(data.id, "Uploads folder not set")
                    return
                # read bytes from file system, encode to base64, then delete the file
                type = "bytes"
                file_path = os.path.join(self._folder, file.source.id.removeprefix("pipecat:"))
                with open(file_path, "rb") as f:
                    raw_bytes = f.read()
                    encoded_file = base64.b64encode(raw_bytes).decode("utf-8")
                    source = f"data:{file.format};base64,{encoded_file}"
                try:
                    os.remove(file_path)
                except OSError as e:
                    logger.warning(f"Failed to remove uploaded file {file_path}: {e}")
            case _:
                logger.warning(f"Unsupported file source type: {type}")
                return

        if type == "bytes" and file.format.startswith("image/"):
            # Only access width/height if the original source is RTVIFileBytes (not RTVIFileUrl)
            if file.source.type == "bytes":
                size = [file.source.width or 0, file.source.height or 0]
            else:
                size = [0, 0]
            file_frame = UserImageRawFrame(
                text=data.content,
                image=source,
                size=size,
                format=f"url/{file.format}",
                append_to_context=True,
            )
        else:
            file_frame = UserFileRawFrame(
                text=data.content,
                file=source,
                type=type,
                format=file.format,
                custom_options=opts.custom_options,
                append_to_context=True,
            )

        if opts.run_immediately:
            await self.interrupt_bot()

        cur_llm_skip_tts = self._llm_skip_tts
        should_skip_tts = not opts.audio_response
        toggle_skip_tts = cur_llm_skip_tts != should_skip_tts
        if toggle_skip_tts:
            output_frame = LLMConfigureOutputFrame(skip_tts=should_skip_tts)
            await self.push_frame(output_frame)
        await self.push_frame(file_frame)
        if toggle_skip_tts:
            output_frame = LLMConfigureOutputFrame(skip_tts=cur_llm_skip_tts)
            await self.push_frame(output_frame)

    async def _handle_update_context(self, data: RTVI.AppendToContextData):
        if data.run_immediately:
            await self.interrupt_bot()
        frame = LLMMessagesAppendFrame(
            messages=[{"role": data.role, "content": data.content}],
            run_llm=data.run_immediately,
        )
        await self.push_frame(frame)

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

    async def _handle_action(self, request_id: Optional[str], data: RTVIActionRun):
        """Handle an action execution request."""
        action_id = self._action_id(data.service, data.action)
        if action_id not in self._registered_actions:
            await self._send_error_response(request_id, f"Action {action_id} not registered")
            return
        action = self._registered_actions[action_id]
        arguments = {}
        if data.arguments:
            for arg in data.arguments:
                arguments[arg.name] = arg.value
        result = await action.handler(self, action.service, arguments)
        # Only send a response if request_id is present. Things that don't care about
        # action responses (such as webhooks) don't set a request_id
        if request_id:
            message = RTVIActionResponse(id=request_id, data=RTVIActionResponseData(result=result))
            await self.push_transport_message(message)

    async def _send_bot_ready(self, about: Mapping[str, Any] = None):
        """Send the bot-ready message to the client.

        Args:
            about: Optional information about the bot to include in the ready message.
                   If left as None, the pipecat library and version will be used.
        """
        if not about:
            about = {"library": "pipecat-ai", "library_version": f"{pipecat_version()}"}
        if self._client_version and self._client_version[0] < 1:
            config = self._config.config
            message = RTVI.BotReady(
                id=self._client_ready_id,
                data=RTVIBotReadyDataDeprecated(
                    version=RTVI.PROTOCOL_VERSION, about=about, config=config
                ),
            )
        else:
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

    def _action_id(self, service: str, action: str) -> str:
        """Generate an action ID from service and action names."""
        return f"{service}:{action}"
