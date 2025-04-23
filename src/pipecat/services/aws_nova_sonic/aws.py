import base64
import json
import uuid
from enum import Enum

from aws_sdk_bedrock_runtime.client import (
    BedrockRuntimeClient,
    InvokeModelWithBidirectionalStreamOperationInput,
)
from aws_sdk_bedrock_runtime.config import Config, HTTPAuthSchemeResolver, SigV4AuthScheme
from aws_sdk_bedrock_runtime.models import (
    BidirectionalInputPayloadPart,
    InvokeModelWithBidirectionalStreamInput,
    InvokeModelWithBidirectionalStreamInputChunk,
    InvokeModelWithBidirectionalStreamOperationOutput,
    InvokeModelWithBidirectionalStreamOutput,
)
from loguru import logger
from smithy_aws_core.credentials_resolvers.static import StaticCredentialsResolver
from smithy_aws_core.identity import AWSCredentialsIdentity
from smithy_core.aio.eventstream import DuplexEventStream

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    StartFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService


class Role(Enum):
    SYSTEM = "SYSTEM"
    USER = "USER"


class AWSNovaSonicService(LLMService):
    def __init__(
        self,
        *,
        instruction: str,
        secret_access_key: str,
        access_key_id: str,
        region: str,
        model: str = "amazon.nova-sonic-v1:0",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._instruction = instruction
        self._secret_access_key = secret_access_key
        self._access_key_id = access_key_id
        self._region = region
        self._model = model
        self._client: BedrockRuntimeClient = None
        self._stream: DuplexEventStream[
            InvokeModelWithBidirectionalStreamInput,
            InvokeModelWithBidirectionalStreamOutput,
            InvokeModelWithBidirectionalStreamOperationOutput,
        ] = None
        self._receive_task = None
        self._prompt_name = str(uuid.uuid4())
        self._input_audio_content_name = str(uuid.uuid4())

    #
    # standard AIService frame handling
    #

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    #
    # frame processing
    #

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            # TODO: check if _audio_input_paused? what causes that?
            await self._send_user_audio(frame)

        await self.push_frame(frame, direction)

    #
    # communication with LLM
    #

    async def _connect(self):
        try:
            # TODO: remove after debugging
            logger.debug("[pk] started connecting!")
            if self._client:
                # Here we assume that if we have a client we are connected
                return

            # Create the client
            self._client = self._create_client()

            # Start the bidirectional stream
            self._stream = await self._client.invoke_model_with_bidirectional_stream(
                InvokeModelWithBidirectionalStreamOperationInput(model_id=self._model)
            )

            # Send session start events
            await self._send_session_start()

            # Send initial system instruction
            await self._send_text(text=self._instruction, role=Role.SYSTEM)

            # Start audio input
            await self._send_audio_input_start()

            self._receive_task = self.create_task(self._receive_task_handler())

            logger.debug("[pk] finished connecting!")
        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._client = None

    async def _disconnect(self):
        pass

    def _create_client(self) -> BedrockRuntimeClient:
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self._region}.amazonaws.com",
            region=self._region,
            aws_credentials_identity_resolver=StaticCredentialsResolver(
                credentials=AWSCredentialsIdentity(
                    access_key_id=self._access_key_id,
                    secret_access_key=self._secret_access_key,
                    # TODO: add additional stuff like aws_session_token
                )
            ),
            http_auth_scheme_resolver=HTTPAuthSchemeResolver(),
            http_auth_schemes={"aws.auth#sigv4": SigV4AuthScheme()},
        )
        return BedrockRuntimeClient(config=config)

    # TODO: make params configurable?
    async def _send_session_start(self):
        session_start = """
        {
          "event": {
            "sessionStart": {
              "inferenceConfiguration": {
                "maxTokens": 1024,
                "topP": 0.9,
                "temperature": 0.7
              }
            }
          }
        }
        """
        await self._send_client_event(session_start)

        prompt_start = f'''
        {{
          "event": {{
            "promptStart": {{
              "promptName": "{self._prompt_name}",
              "textOutputConfiguration": {{
                "mediaType": "text/plain"
              }},
              "audioOutputConfiguration": {{
                "mediaType": "audio/lpcm",
                "sampleRateHertz": 24000,
                "sampleSizeBits": 16,
                "channelCount": 1,
                "voiceId": "matthew",
                "encoding": "base64",
                "audioType": "SPEECH"
              }}
            }}
          }}
        }}
        '''
        await self._send_client_event(prompt_start)

    async def _send_audio_input_start(self):
        audio_content_start = f'''
        {{
            "event": {{
                "contentStart": {{
                    "promptName": "{self._prompt_name}",
                    "contentName": "{self._input_audio_content_name}",
                    "type": "AUDIO",
                    "interactive": true,
                    "role": "USER",
                    "audioInputConfiguration": {{
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": 16000,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "audioType": "SPEECH",
                        "encoding": "base64"
                    }}
                }}
            }}
        }}
        '''
        await self._send_client_event(audio_content_start)

    async def _send_text(self, text: str, role: Role):
        content_name = str(uuid.uuid4())

        text_content_start = f'''
        {{
            "event": {{
                "contentStart": {{
                    "promptName": "{self._prompt_name}",
                    "contentName": "{content_name}",
                    "type": "TEXT",
                    "interactive": true,
                    "role": "{role.value}",
                    "textInputConfiguration": {{
                        "mediaType": "text/plain"
                    }}
                }}
            }}
        }}
        '''
        await self._send_client_event(text_content_start)

        text_input = f'''
        {{
            "event": {{
                "textInput": {{
                    "promptName": "{self._prompt_name}",
                    "contentName": "{content_name}",
                    "content": "{text}"
                }}
            }}
        }}
        '''
        await self._send_client_event(text_input)

        text_content_end = f'''
        {{
            "event": {{
                "contentEnd": {{
                    "promptName": "{self._prompt_name}",
                    "contentName": "{content_name}"
                }}
            }}
        }}
        '''
        await self._send_client_event(text_content_end)

    async def _send_user_audio(self, frame: InputAudioRawFrame):
        if not self._client:
            return

        blob = base64.b64encode(frame.audio)
        audio_event = f'''
        {{
            "event": {{
                "audioInput": {{
                    "promptName": "{self._prompt_name}",
                    "contentName": "{self._input_audio_content_name}",
                    "content": "{blob.decode("utf-8")}"
                }}
            }}
        }}
        '''
        await self._send_client_event(audio_event)

    async def _send_client_event(self, event_json: str):
        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode("utf-8"))
        )
        await self._stream.input_stream.send(event)

    async def _receive_task_handler(self):
        try:
            while self._client:
                # TODO: remove after debugging
                logger.debug(f"[pk] awaiting output from server...")

                output = await self._stream.await_output()

                # TODO: remove after debugging
                logger.debug(f"[pk] got output from server: {result}")

                result = await output[1].receive()

                # TODO: remove after debugging
                logger.debug(f"[pk] got result from server: {result}")

                if result.value and result.value.bytes_:
                    response_data = result.value.bytes_.decode("utf-8")
                    json_data = json.loads(response_data)

                # TODO: remove after debugging
                logger.debug(f"[pk] got JSON from server: {json_data}")

                if "audioOutput" in json_data["event"]:
                    await self._handle_audio_output_event(json_data["event"])
        except Exception as e:
            logger.error(f"{self} error processing responses: {e}")

    async def _handle_audio_output_event(self, event):
        # TODO: remove after debugging
        logger.debug("[pk] got output audio!")
        audio_content = event["audioOutput"]["content"]
        audio = base64.b64decode(audio_content)
        # TODO: how is _current_audio_response used?
        # TODO: make sample rate + channels (used in multiple places) consts
        frame = TTSAudioRawFrame(
            audio=audio,
            sample_rate=24000,
            num_channels=1,
        )
        await self.push_frame(frame)
