#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import base64
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import websockets
from loguru import logger
from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InputImageRawFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMSetToolsFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    StartFrame,
    StartInterruptionFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import LLMService
from pipecat.services.openai import (
    OpenAIAssistantContextAggregator,
    OpenAIUserContextAggregator,
)
from pipecat.utils.time import time_now_iso8601

from . import events
from .audio_transcriber import AudioTranscriber


class GeminiMultimodalLiveContext(OpenAILLMContext):
    @staticmethod
    def upgrade(obj: OpenAILLMContext) -> "GeminiMultimodalLiveContext":
        if isinstance(obj, OpenAILLMContext) and not isinstance(obj, GeminiMultimodalLiveContext):
            logger.debug(f"Upgrading to Gemini Multimodal Live Context: {obj}")
            obj.__class__ = GeminiMultimodalLiveContext
            obj._restructure_from_openai_messages()
        return obj

    def _restructure_from_openai_messages(self):
        pass

    def extract_system_instructions(self):
        system_instruction = ""
        for item in self.messages:
            if item.get("role") == "system":
                content = item.get("content", "")
                if content:
                    if system_instruction and not system_instruction.endswith("\n"):
                        system_instruction += "\n"
                    system_instruction += str(content)
        return system_instruction

    def get_messages_for_initializing_history(self):
        messages = []
        for item in self.messages:
            role = item.get("role")

            if role == "system":
                continue

            elif role == "assistant":
                role = "model"

            content = item.get("content")
            parts = []
            if isinstance(content, str):
                parts = [{"text": content}]
            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        parts.append({"text": part.get("text")})
                    else:
                        logger.warning(f"Unsupported content type: {str(part)[:80]}")
            else:
                logger.warning(f"Unsupported content type: {str(content)[:80]}")
            messages.append({"role": role, "parts": parts})
        return messages


class GeminiMultimodalLiveUserContextAggregator(OpenAIUserContextAggregator):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        # kind of a hack just to pass the LLMMessagesAppendFrame through, but it's fine for now
        if isinstance(frame, LLMMessagesAppendFrame):
            await self.push_frame(frame, direction)


class GeminiMultimodalLiveAssistantContextAggregator(OpenAIAssistantContextAggregator):
    async def _push_aggregation(self):
        # We don't want to store any images in the context. Revisit this later when the API evolves.
        self._pending_image_frame_message = None
        await super()._push_aggregation()


@dataclass
class GeminiMultimodalLiveContextAggregatorPair:
    _user: GeminiMultimodalLiveUserContextAggregator
    _assistant: GeminiMultimodalLiveAssistantContextAggregator

    def user(self) -> GeminiMultimodalLiveUserContextAggregator:
        return self._user

    def assistant(self) -> GeminiMultimodalLiveAssistantContextAggregator:
        return self._assistant


class GeminiMultimodalModalities(Enum):
    TEXT = "TEXT"
    AUDIO = "AUDIO"


class InputParams(BaseModel):
    frequency_penalty: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=4096, ge=1)
    presence_penalty: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(default=None, ge=0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    modalities: Optional[GeminiMultimodalModalities] = Field(
        default=GeminiMultimodalModalities.AUDIO
    )
    extra: Optional[Dict[str, Any]] = Field(default_factory=dict)


class GeminiMultimodalLiveLLMService(LLMService):
    def __init__(
        self,
        *,
        api_key: str,
        base_url="generativelanguage.googleapis.com",
        model="models/gemini-2.0-flash-exp",
        voice_id: str = "Charon",
        start_audio_paused: bool = False,
        start_video_paused: bool = False,
        system_instruction: Optional[str] = None,
        tools: Optional[List[dict]] = None,
        transcribe_user_audio: bool = False,
        transcribe_model_audio: bool = False,
        params: InputParams = InputParams(),
        inference_on_context_initialization: bool = True,
        **kwargs,
    ):
        super().__init__(base_url=base_url, **kwargs)
        self.api_key = api_key
        self.base_url = base_url
        self.set_model_name(model)
        self._voice_id = voice_id

        self._system_instruction = system_instruction
        self._tools = tools
        self._inference_on_context_initialization = inference_on_context_initialization
        self._needs_turn_complete_message = False

        self._audio_input_paused = start_audio_paused
        self._video_input_paused = start_video_paused
        self._websocket = None
        self._receive_task = None
        self._context = None

        self._disconnecting = False
        self._api_session_ready = False
        self._run_llm_when_api_session_ready = False

        self._transcriber = AudioTranscriber(api_key)
        self._transcribe_user_audio = transcribe_user_audio
        self._transcribe_model_audio = transcribe_model_audio
        self._user_is_speaking = False
        self._bot_is_speaking = False
        self._user_audio_buffer = bytearray()
        self._bot_audio_buffer = bytearray()
        self._bot_text_buffer = ""

        self._settings = {
            "frequency_penalty": params.frequency_penalty,
            "max_tokens": params.max_tokens,
            "presence_penalty": params.presence_penalty,
            "temperature": params.temperature,
            "top_k": params.top_k,
            "top_p": params.top_p,
            "modalities": params.modalities,
            "extra": params.extra if isinstance(params.extra, dict) else {},
        }

    def can_generate_metrics(self) -> bool:
        return True

    def set_audio_input_paused(self, paused: bool):
        self._audio_input_paused = paused

    def set_video_input_paused(self, paused: bool):
        self._video_input_paused = paused

    def set_model_modalities(self, modalities: GeminiMultimodalModalities):
        self._settings["modalities"] = modalities

    async def set_context(self, context: OpenAILLMContext):
        """Set the context explicitly from outside the pipeline.

        This is useful when initializing a conversation because in server-side VAD mode we might not have a
        way to trigger the pipeline. This sends the history to the server. The `inference_on_context_initialization`
        flag controls whether to set the turnComplete flag when we do this. Without that flag, the model will
        not respond. This is often what we want when setting the context at the beginning of a conversation.
        """
        if self._context:
            logger.error(
                "Context already set. Can only set up Gemini Multimodal Live context once."
            )
            return
        self._context = GeminiMultimodalLiveContext.upgrade(context)
        await self._create_initial_response()

    #
    # standard AIService frame handling
    #

    async def start(self, frame: StartFrame):
        await super().start(frame)

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    #
    # speech and interruption handling
    #

    async def _handle_interruption(self):
        pass

    async def _handle_user_started_speaking(self, frame):
        self._user_is_speaking = True
        pass

    async def _handle_user_stopped_speaking(self, frame):
        self._user_is_speaking = False
        audio = self._user_audio_buffer
        self._user_audio_buffer = bytearray()
        if self._needs_turn_complete_message:
            self._needs_turn_complete_message = False
            evt = events.ClientContentMessage.model_validate(
                {"clientContent": {"turnComplete": True}}
            )
            await self.send_client_event(evt)
        if self._transcribe_user_audio and self._context:
            asyncio.create_task(self._handle_transcribe_user_audio(audio, self._context))

    async def _handle_transcribe_user_audio(self, audio, context):
        text = await self._transcribe_audio(audio, context)
        if not text:
            return
        logger.debug(f"[Transcription:user] {text}")
        context.add_message({"role": "user", "content": [{"type": "text", "text": text}]})
        await self.push_frame(
            TranscriptionFrame(text=text, user_id="user", timestamp=time_now_iso8601())
        )

    async def _handle_transcribe_model_audio(self, audio, context):
        # Early return if modalities are not set to audio.
        if self._settings["modalities"] != GeminiMultimodalModalities.AUDIO:
            return

        text = await self._transcribe_audio(audio, context)
        logger.debug(f"[Transcription:model] {text}")
        # We add user messages directly to the context. We don't do that for assistant messages,
        # because we assume the frames we emit will work normally in this downstream case. This
        # definitely feels like a hack. Need to revisit when the API evolves.
        # context.add_message({"role": "assistant", "content": [{"type": "text", "text": text}]})
        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(LLMTextFrame(text=text))
        await self.push_frame(LLMFullResponseEndFrame())

    async def _transcribe_audio(self, audio, context):
        (text, prompt_tokens, completion_tokens, total_tokens) = await self._transcriber.transcribe(
            audio, context
        )
        if not text:
            return ""
        # The only usage metrics we have right now are for the transcriber LLM. The Live API is free.
        await self.start_llm_usage_metrics(
            LLMTokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
        )
        return text

    #
    # frame processing
    #
    # StartFrame, StopFrame, CancelFrame implemented in base class
    #

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # logger.debug(f"Processing frame: {frame}")

        if isinstance(frame, TranscriptionFrame):
            pass
        elif isinstance(frame, OpenAILLMContextFrame):
            context: GeminiMultimodalLiveContext = GeminiMultimodalLiveContext.upgrade(
                frame.context
            )
            # For now, we'll only trigger inference here when either:
            #   1. We have not seen a context frame before
            #   2. The last message is a tool call result
            if not self._context:
                self._context = context
                if frame.context.tools:
                    self._tools = frame.context.tools
                await self._create_initial_response()
            elif context.messages and context.messages[-1].get("role") == "tool":
                # Support just one tool call per context frame for now
                tool_result_message = context.messages[-1]
                await self._tool_result(tool_result_message)

        elif isinstance(frame, InputAudioRawFrame):
            await self._send_user_audio(frame)
        elif isinstance(frame, InputImageRawFrame):
            await self._send_user_video(frame)
        elif isinstance(frame, StartInterruptionFrame):
            await self._handle_interruption()
        elif isinstance(frame, UserStartedSpeakingFrame):
            await self._handle_user_started_speaking(frame)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._handle_user_stopped_speaking(frame)
        elif isinstance(frame, BotStartedSpeakingFrame):
            # Ignore this frame. Use the serverContent API message instead
            pass
        elif isinstance(frame, BotStoppedSpeakingFrame):
            # ignore this frame. Use the serverContent.turnComplete API message
            pass
        elif isinstance(frame, LLMMessagesAppendFrame):
            await self._create_single_response(frame.messages)
        elif isinstance(frame, LLMUpdateSettingsFrame):
            await self._update_settings(frame.settings)
        elif isinstance(frame, LLMSetToolsFrame):
            await self._update_settings()

        await self.push_frame(frame, direction)

    #
    # websocket communication
    #

    async def send_client_event(self, event):
        await self._ws_send(event.model_dump(exclude_none=True))

    async def _connect(self):
        logger.info("Connecting to Gemini service")
        try:
            if self._websocket:
                # Here we assume that if we have a websocket, we are connected. We
                # handle disconnections in the send/recv code paths.
                return

            uri = f"wss://{self.base_url}/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={self.api_key}"
            logger.info(f"Connecting to {uri}")
            self._websocket = await websockets.connect(uri=uri)
            self._receive_task = self.get_event_loop().create_task(self._receive_task_handler())
            config = events.Config.model_validate(
                {
                    "setup": {
                        "model": self._model_name,
                        "generation_config": {
                            "frequency_penalty": self._settings["frequency_penalty"],
                            "max_output_tokens": self._settings["max_tokens"],  # Not supported yet
                            "presence_penalty": self._settings["presence_penalty"],
                            "temperature": self._settings["temperature"],
                            "top_k": self._settings["top_k"],
                            "top_p": self._settings["top_p"],
                            "response_modalities": self._settings["modalities"].value,
                            "speech_config": {
                                "voice_config": {
                                    "prebuilt_voice_config": {"voice_name": self._voice_id}
                                },
                            },
                        },
                    },
                }
            )

            system_instruction = self._system_instruction or ""
            if self._context and hasattr(self._context, "extract_system_instructions"):
                system_instruction += "\n" + self._context.extract_system_instructions()
            if system_instruction:
                logger.debug(f"Setting system instruction: {system_instruction}")
                config.setup.system_instruction = events.SystemInstruction(
                    parts=[events.ContentPart(text=system_instruction)]
                )
            if self._tools:
                logger.debug(f"Gemini is configuring to use tools{self._tools}")
                config.setup.tools = self._tools
            await self.send_client_event(config)

        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None

    async def _disconnect(self):
        logger.info("Disconnecting from Gemini service")
        try:
            self._disconnecting = True
            self._api_session_ready = False
            await self.stop_all_metrics()
            if self._websocket:
                await self._websocket.close()
                self._websocket = None
            if self._receive_task:
                self._receive_task.cancel()
                try:
                    await asyncio.wait_for(self._receive_task, timeout=1.0)
                except asyncio.TimeoutError:
                    logger.warning("Timed out waiting for receive task to finish")
                self._receive_task = None
            self._disconnecting = False
        except Exception as e:
            logger.error(f"{self} error disconnecting: {e}")

    async def _ws_send(self, message):
        # logger.debug(f"Sending message to websocket: {message}")
        try:
            if not self._websocket:
                await self._connect()
            await self._websocket.send(json.dumps(message))
        except Exception as e:
            if self._disconnecting:
                return
            logger.error(f"Error sending message to websocket: {e}")
            # In server-to-server contexts, a WebSocket error should be quite rare. Given how hard
            # it is to recover from a send-side error with proper state management, and that exponential
            # backoff for retries can have cost/stability implications for a service cluster, let's just
            # treat a send-side error as fatal.
            await self.push_error(ErrorFrame(error=f"Error sending client event: {e}", fatal=True))

    #
    # inbound server event handling
    # todo: docs link here
    #

    async def _receive_task_handler(self):
        try:
            async for message in self._websocket:
                evt = events.parse_server_event(message)
                # logger.debug(f"Received event: {message[:500]}")
                # logger.debug(f"Received event: {evt}")

                if evt.setupComplete:
                    await self._handle_evt_setup_complete(evt)
                elif evt.serverContent and evt.serverContent.modelTurn:
                    await self._handle_evt_model_turn(evt)
                elif evt.serverContent and evt.serverContent.turnComplete:
                    await self._handle_evt_turn_complete(evt)
                elif evt.toolCall:
                    await self._handle_evt_tool_call(evt)

                elif False:  # !!! todo: error events?
                    await self._handle_evt_error(evt)
                    # errors are fatal, so exit the receive loop
                    return

                else:
                    pass
        except asyncio.CancelledError:
            logger.debug("websocket receive task cancelled")
        except Exception as e:
            logger.error(f"{self} exception: {e}")

    #
    #
    #

    async def _send_user_audio(self, frame):
        if self._audio_input_paused:
            return
        # Send all audio to Gemini
        evt = events.AudioInputMessage.from_raw_audio(frame.audio)
        await self.send_client_event(evt)
        # Manage a buffer of audio to use for transcription
        audio = frame.audio
        if self._user_is_speaking:
            self._user_audio_buffer.extend(audio)
        else:
            # Keep 1/2 second of audio in the buffer even when not speaking.
            self._user_audio_buffer.extend(audio)
            length = int((frame.sample_rate * frame.num_channels * 2) * 0.5)
            self._user_audio_buffer = self._user_audio_buffer[-length:]

    async def _send_user_video(self, frame):
        if self._video_input_paused:
            return
        # logger.debug(f"Sending video frame to Gemini: {frame}")
        evt = events.VideoInputMessage.from_image_frame(frame)
        await self.send_client_event(evt)

    async def _create_initial_response(self):
        if not self._api_session_ready:
            self._run_llm_when_api_session_ready = True
            return

        messages = self._context.get_messages_for_initializing_history()
        if not messages:
            return

        logger.debug(f"Creating initial response: {messages}")

        evt = events.ClientContentMessage.model_validate(
            {
                "clientContent": {
                    "turns": messages,
                    "turnComplete": self._inference_on_context_initialization,
                }
            }
        )
        await self.send_client_event(evt)
        if not self._inference_on_context_initialization:
            self._needs_turn_complete_message = True

    async def _create_single_response(self, messages_list):
        # refactor to combine this logic with same logic in GeminiMultimodalLiveContext
        messages = []
        for item in messages_list:
            role = item.get("role")

            if role == "system":
                continue

            elif role == "assistant":
                role = "model"

            content = item.get("content")
            parts = []
            if isinstance(content, str):
                parts = [{"text": content}]
            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        parts.append({"text": part.get("text")})
                    else:
                        logger.warning(f"Unsupported content type: {str(part)[:80]}")
            else:
                logger.warning(f"Unsupported content type: {str(content)[:80]}")
            messages.append({"role": role, "parts": parts})
        if not messages:
            return
        logger.debug(f"Creating response: {messages}")

        evt = events.ClientContentMessage.model_validate(
            {
                "clientContent": {
                    "turns": messages,
                    "turnComplete": True,
                }
            }
        )
        await self.send_client_event(evt)

    async def _tool_result(self, tool_result_message):
        # For now we're shoving the name into the tool_call_id field, so this
        # will work until we revisit that.
        id = tool_result_message.get("tool_call_id")
        name = tool_result_message.get("tool_call_name")
        result = json.loads(tool_result_message.get("content") or "")
        response_message = json.dumps(
            {
                "toolResponse": {
                    "functionResponses": [
                        {
                            "id": id,
                            "name": name,
                            "response": {
                                "result": result,
                            },
                        }
                    ],
                }
            }
        )
        await self._websocket.send(response_message)
        # await self._websocket.send(json.dumps({"clientContent": {"turnComplete": True}}))

    async def _handle_evt_setup_complete(self, evt):
        # If this is our first context frame, run the LLM
        self._api_session_ready = True
        # Now that we've configured the session, we can run the LLM if we need to.
        if self._run_llm_when_api_session_ready:
            self._run_llm_when_api_session_ready = False
            await self._create_initial_response()

    async def _handle_evt_model_turn(self, evt):
        part = evt.serverContent.modelTurn.parts[0]
        if not part:
            return

        text = part.text
        if text:
            if not self._bot_text_buffer:
                await self.push_frame(LLMFullResponseStartFrame())

            self._bot_text_buffer += text
            await self.push_frame(LLMTextFrame(text=text))

        inline_data = part.inlineData
        if not inline_data:
            return
        if inline_data.mimeType != "audio/pcm;rate=24000":
            logger.warning(f"Unrecognized server_content format {inline_data.mimeType}")
            return

        audio = base64.b64decode(inline_data.data)
        if not audio:
            return

        if not self._bot_is_speaking:
            self._bot_is_speaking = True
            await self.push_frame(TTSStartedFrame())

        self._bot_audio_buffer.extend(audio)
        frame = TTSAudioRawFrame(
            audio=audio,
            sample_rate=24000,
            num_channels=1,
        )
        await self.push_frame(frame)

    async def _handle_evt_tool_call(self, evt):
        function_calls = evt.toolCall.functionCalls
        if not function_calls:
            return
        if not self._context:
            logger.error("Function calls are not supported without a context object.")
        for call in function_calls:
            await self.call_function(
                context=self._context,
                tool_call_id=call.id,
                function_name=call.name,
                arguments=call.args,
            )

    async def _handle_evt_turn_complete(self, evt):
        self._bot_is_speaking = False
        audio = self._bot_audio_buffer
        text = self._bot_text_buffer
        self._bot_audio_buffer = bytearray()
        self._bot_text_buffer = ""

        if audio and self._transcribe_model_audio and self._context:
            asyncio.create_task(self._handle_transcribe_model_audio(audio, self._context))
        elif text:
            await self.push_frame(LLMFullResponseEndFrame())

        await self.push_frame(TTSStoppedFrame())

    def create_context_aggregator(
        self, context: OpenAILLMContext, *, assistant_expect_stripped_words: bool = False
    ) -> GeminiMultimodalLiveContextAggregatorPair:
        GeminiMultimodalLiveContext.upgrade(context)
        user = GeminiMultimodalLiveUserContextAggregator(context)
        assistant = GeminiMultimodalLiveAssistantContextAggregator(
            user, expect_stripped_words=assistant_expect_stripped_words
        )
        return GeminiMultimodalLiveContextAggregatorPair(_user=user, _assistant=assistant)
