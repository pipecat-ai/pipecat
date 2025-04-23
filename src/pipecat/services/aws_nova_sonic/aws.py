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
from smithy_aws_core.credentials_resolvers.static import StaticCredentialsResolver
from smithy_aws_core.identity import AWSCredentialsIdentity
from smithy_core.aio.eventstream import DuplexEventStream

from pipecat.frames.frames import CancelFrame, EndFrame, StartFrame
from pipecat.services.llm_service import LLMService


class AWSNovaSonicService(LLMService):
    def __init__(
        self,
        *,
        secret_access_key: str,
        access_key_id: str,
        region: str,
        model: str = "amazon.nova-sonic-v1:0",
        **kwargs,
    ):
        super().__init__(**kwargs)
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
    # communication
    #

    async def _connect(self):
        if self._client:
            # Here we assume that if we have a client we are connected.
            return
        self._initialize_client()
        self._stream = await self._client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self._model)
        )
        self._receive_task = self.create_task(self._receive_task_handler())
        pass

    async def _disconnect(self):
        pass

    def _initialize_client(self) -> BedrockRuntimeClient:
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
        self._client = BedrockRuntimeClient(config=config)

    async def _send_client_event(self, event_json):
        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode("utf-8"))
        )
        await self._stream.input_stream.send(event)

    async def _receive_task_handler(self):
        pass
