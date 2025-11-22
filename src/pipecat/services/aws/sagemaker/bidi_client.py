#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AWS SageMaker bidirectional streaming client.

This module provides a client for streaming bidirectional communication with
SageMaker endpoints using the HTTP/2 protocol. Supports sending audio, text,
and JSON data to SageMaker model endpoints and receiving streaming responses.
"""

import os
from typing import Optional

from loguru import logger

try:
    from aws_sdk_sagemaker_runtime_http2.client import SageMakerRuntimeHTTP2Client
    from aws_sdk_sagemaker_runtime_http2.config import Config, HTTPAuthSchemeResolver
    from aws_sdk_sagemaker_runtime_http2.models import (
        InvokeEndpointWithBidirectionalStreamInput,
        RequestPayloadPart,
        RequestStreamEventPayloadPart,
        ResponseStreamEvent,
    )
    from smithy_aws_core.auth.sigv4 import SigV4AuthScheme
    from smithy_aws_core.identity import EnvironmentCredentialsResolver
    from smithy_core.aio.eventstream import DuplexEventStream
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use SageMaker BiDi client, you need to `pip install pipecat-ai[sagemaker]`."
    )
    raise Exception(f"Missing module: {e}")


class SageMakerBidiClient:
    """Client for bidirectional streaming with AWS SageMaker endpoints.

    Handles low-level HTTP/2 bidirectional streaming protocol for communicating
    with SageMaker model endpoints. Provides methods for sending various data
    types (audio, text, JSON) and receiving streaming responses.

    This client uses AWS SigV4 authentication and supports credential resolution
    from environment variables, AWS CLI configuration, and instance metadata.

    Example::

        client = SageMakerBidiClient(
            endpoint_name="my-deepgram-endpoint",
            region="us-east-2",
            model_invocation_path="v1/listen",
            model_query_string="model=nova-3&language=en"
        )
        await client.start_session()
        await client.send_audio_chunk(audio_bytes)
        response = await client.receive_response()
        await client.close_session()
    """

    def __init__(
        self,
        endpoint_name: str,
        region: str,
        model_invocation_path: str = "",
        model_query_string: str = "",
    ):
        """Initialize the SageMaker BiDi client.

        Args:
            endpoint_name: Name of the SageMaker endpoint to connect to.
            region: AWS region where the endpoint is deployed.
            model_invocation_path: API path for the model invocation (e.g., "v1/listen").
            model_query_string: Query string parameters for the model (e.g., "model=nova-3").
        """
        self.endpoint_name = endpoint_name
        self.region = region
        self.model_invocation_path = model_invocation_path
        self.model_query_string = model_query_string
        self.bidi_endpoint = f"https://runtime.sagemaker.{region}.amazonaws.com:8443"
        self._client: Optional[SageMakerRuntimeHTTP2Client] = None
        self._stream: Optional[
            DuplexEventStream[RequestStreamEventPayloadPart, ResponseStreamEvent, any]
        ] = None
        self._output_stream = None
        self._is_active = False

    def _initialize_client(self):
        """Initialize the SageMaker Runtime HTTP2 client with AWS credentials.

        Creates and configures the SageMaker Runtime HTTP2 client with SigV4
        authentication. Attempts to resolve AWS credentials from environment
        variables, AWS CLI configuration, or instance metadata.
        """
        logger.debug(f"Initializing SageMaker BiDi client for region: {self.region}")
        logger.debug(f"Using endpoint URI: {self.bidi_endpoint}")

        # Check for AWS credentials
        has_env_creds = bool(os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))

        if not has_env_creds:
            logger.warning(
                "AWS credentials not found in environment variables. "
                "Attempting to use EnvironmentCredentialsResolver which will check "
                "AWS CLI configuration and instance metadata."
            )

        config = Config(
            endpoint_uri=self.bidi_endpoint,
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
            auth_scheme_resolver=HTTPAuthSchemeResolver(),
            auth_schemes={"aws.auth#sigv4": SigV4AuthScheme(service="sagemaker")},
        )
        self._client = SageMakerRuntimeHTTP2Client(config=config)

    async def start_session(self):
        """Start a bidirectional streaming session with the SageMaker endpoint.

        Initializes the client if needed, creates the bidirectional stream, and
        establishes the connection to the SageMaker endpoint. Must be called
        before sending or receiving data.

        Returns:
            The output stream for receiving responses.

        Raises:
            RuntimeError: If client initialization or connection fails.
        """
        if not self._client:
            self._initialize_client()

        logger.debug(f"Starting BiDi session with endpoint: {self.endpoint_name}")
        logger.debug(f"Model invocation path: {self.model_invocation_path}")
        logger.debug(f"Model query string: {self.model_query_string}")

        # Create the bidirectional stream
        stream_input = InvokeEndpointWithBidirectionalStreamInput(
            endpoint_name=self.endpoint_name,
            model_invocation_path=self.model_invocation_path,
            model_query_string=self.model_query_string,
        )

        try:
            self._stream = await self._client.invoke_endpoint_with_bidirectional_stream(
                stream_input
            )
            self._is_active = True

            # Get output stream
            output = await self._stream.await_output()
            self._output_stream = output[1]

            logger.debug("BiDi session started successfully")
            return self._output_stream

        except Exception as e:
            logger.error(f"Failed to start BiDi session: {e}")
            self._is_active = False
            raise RuntimeError(f"Failed to start SageMaker BiDi session: {e}")

    async def send_data(self, data_bytes: bytes, data_type: Optional[str] = None):
        """Send a chunk of data to the stream.

        Generic method for sending any type of data to the SageMaker endpoint.
        Use the convenience methods (send_audio_chunk, send_text, send_json)
        for common data types.

        Args:
            data_bytes: Raw bytes to send.
            data_type: Optional data type header. Common values are "BINARY" for
                audio/binary data and "UTF8" for text/JSON data.

        Raises:
            RuntimeError: If session is not active or send fails.
        """
        if not self._is_active or not self._stream:
            raise RuntimeError("BiDi session not active")

        try:
            payload = RequestPayloadPart(bytes_=data_bytes, data_type=data_type)
            event = RequestStreamEventPayloadPart(value=payload)
            await self._stream.input_stream.send(event)
        except Exception as e:
            logger.error(f"Failed to send data: {e}")
            raise

    async def send_audio_chunk(self, audio_bytes: bytes):
        """Send a chunk of audio data to the stream.

        Convenience method for sending audio data. Automatically sets the data
        type to "BINARY".

        Args:
            audio_bytes: Raw audio bytes to send (e.g., PCM audio data).

        Raises:
            RuntimeError: If session is not active or send fails.
        """
        await self.send_data(audio_bytes, data_type="BINARY")

    async def send_text(self, text: str):
        """Send text data to the stream.

        Convenience method for sending text data. Automatically encodes the text
        as UTF-8 and sets the data type to "UTF8".

        Args:
            text: Text string to send.

        Raises:
            RuntimeError: If session is not active or send fails.
        """
        await self.send_data(text.encode("utf-8"), data_type="UTF8")

    async def send_json(self, data: dict):
        """Send JSON data to the stream.

        Convenience method for sending JSON-encoded messages. Useful for control
        messages like KeepAlive or CloseStream. Automatically serializes the
        dictionary to JSON, encodes as UTF-8, and sets the data type to "UTF8".

        Args:
            data: Dictionary to send as JSON (e.g., {"type": "KeepAlive"}).

        Raises:
            RuntimeError: If session is not active or send fails.
        """
        import json

        await self.send_data(json.dumps(data).encode("utf-8"), data_type="UTF8")

    async def receive_response(self) -> Optional[ResponseStreamEvent]:
        """Receive a response from the stream.

        Blocks until a response is available from the SageMaker endpoint. Returns
        None when the stream is closed.

        Returns:
            The response event containing payload data, or None if stream is closed.

        Raises:
            RuntimeError: If session is not active.
        """
        if not self._is_active or not self._output_stream:
            raise RuntimeError("BiDi session not active")

        try:
            result = await self._output_stream.receive()
            return result
        except Exception as e:
            logger.error(f"Failed to receive response: {e}")
            raise

    async def close_session(self):
        """Close the bidirectional streaming session.

        Gracefully closes the input stream and marks the session as inactive.
        Safe to call multiple times.
        """
        if not self._is_active:
            return

        logger.debug("Closing BiDi session...")
        self._is_active = False

        try:
            if self._stream:
                await self._stream.input_stream.close()
            logger.debug("BiDi session closed successfully")
        except Exception as e:
            logger.warning(f"Error closing BiDi session: {e}")

    @property
    def is_active(self) -> bool:
        """Check if the session is currently active.

        Returns:
            True if session is active, False otherwise.
        """
        return self._is_active
