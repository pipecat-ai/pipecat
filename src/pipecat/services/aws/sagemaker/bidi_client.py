#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AWS SageMaker bidirectional streaming client.

This module provides a client for streaming bidirectional communication with
SageMaker endpoints using the HTTP/2 protocol. Supports sending audio, text,
and JSON data to SageMaker model endpoints and receiving streaming responses.
"""

import asyncio
import os
import random
import re

from loguru import logger

from pipecat.utils.errors import ErrorCategory
from pipecat.utils.network import exponential_backoff_time

try:
    from aws_sdk_sagemaker_runtime_http2.client import SageMakerRuntimeHTTP2Client
    from aws_sdk_sagemaker_runtime_http2.config import Config
    from aws_sdk_sagemaker_runtime_http2.models import (
        InvokeEndpointWithBidirectionalStreamInput,
        InvokeEndpointWithBidirectionalStreamOutput,
        RequestPayloadPart,
        RequestStreamEvent,
        RequestStreamEventPayloadPart,
        ResponseStreamEvent,
    )
    from smithy_aws_core.identity import EnvironmentCredentialsResolver
    from smithy_core.aio.eventstream import DuplexEventStream
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        'In order to use SageMaker BiDi client, you need to `uv add "pipecat-ai[sagemaker]"`.'
    )
    raise ImportError(f"Missing module: {e}") from e

# The SDK reports an unmodeled error (ThrottlingException among them) as a
# generic error whose only record of the HTTP status and error id is its message.
_STATUS_PATTERN = re.compile(r"status: (\d{3})")
_ERROR_ID_PATTERN = re.compile(r"id: (?:\S*#)?(\w+)")

_SERVER_ERROR_IDS = frozenset(
    {"InternalServerError", "InternalStreamFailure", "ServiceUnavailableError"}
)
_INVALID_REQUEST_ERROR_IDS = frozenset({"InputValidationError", "ValidationError"})
_RETRYABLE_CATEGORIES = frozenset(
    {ErrorCategory.RATE_LIMIT, ErrorCategory.SERVER, ErrorCategory.CONNECTIVITY}
)
_AUTH_STATUSES = frozenset({401, 403})

_CONNECT_BACKOFF_MULTIPLIER = 0.5
_CONNECT_BACKOFF_MAX_WAIT = 2.0


class SageMakerBidiSessionError(RuntimeError):
    """A bidirectional streaming session failed to start.

    Parameters:
        error_id: The AWS error id (e.g. ``"ThrottlingException"``), if known.
        http_status: The HTTP status AWS returned, if known. SageMaker reports
            throttling with a 400, so the status alone does not say whether a
            failure is worth retrying; use :func:`classify_sagemaker_bidi_error`.
        attempts: How many times the session was attempted.
    """

    def __init__(
        self,
        message: str,
        *,
        error_id: str | None = None,
        http_status: int | None = None,
        attempts: int = 1,
    ):
        """Initialize the error.

        Args:
            message: Description of the failure.
            error_id: The AWS error id, if known.
            http_status: The HTTP status AWS returned, if known.
            attempts: How many times the session was attempted.
        """
        super().__init__(message)
        self.error_id = error_id
        self.http_status = http_status
        self.attempts = attempts


def _error_id_and_status(exception: BaseException) -> tuple[str | None, int | None]:
    if isinstance(exception, SageMakerBidiSessionError):
        return exception.error_id, exception.http_status
    message = str(exception)
    status_match = _STATUS_PATTERN.search(message)
    id_match = _ERROR_ID_PATTERN.search(message)
    status = int(status_match.group(1)) if status_match else None
    if status is None:
        # A ModelError carries the status the model container itself returned.
        status = getattr(exception, "original_status_code", None)
    if id_match:
        return id_match.group(1), status
    # Modeled errors are raised as their own exception types.
    name = type(exception).__name__
    if name in _SERVER_ERROR_IDS or name in _INVALID_REQUEST_ERROR_IDS or name == "ModelError":
        return name, status
    return None, status


def classify_sagemaker_bidi_error(exception: BaseException) -> ErrorCategory | None:
    """Classify a failure to start or use a SageMaker bidirectional stream.

    Args:
        exception: The exception to classify.

    Returns:
        The category, or None if the exception is not recognized.
    """
    if getattr(exception, "is_throttling_error", False):
        return ErrorCategory.RATE_LIMIT
    error_id, status = _error_id_and_status(exception)
    if error_id == "ThrottlingException" or status == 429:
        return ErrorCategory.RATE_LIMIT
    if error_id == "ModelError":
        # SageMaker reports any failure to open the stream to the model container
        # as a 424, whether the container is at capacity, shedding load, or
        # rejected the request's settings, so a 424 is treated as transient.
        if status == 424 or (status is not None and 500 <= status < 600):
            return ErrorCategory.SERVER
        return ErrorCategory.INVALID_REQUEST
    if error_id in _SERVER_ERROR_IDS or (status is not None and 500 <= status < 600):
        return ErrorCategory.SERVER
    if status in _AUTH_STATUSES:
        # Credentials are resolved when the session starts, so a rejection can be
        # an expired credential that a new session clears.
        return ErrorCategory.CONNECTIVITY
    if error_id in _INVALID_REQUEST_ERROR_IDS:
        return ErrorCategory.INVALID_REQUEST
    if isinstance(exception, (ConnectionError, TimeoutError)):
        return ErrorCategory.CONNECTIVITY
    return None


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
        model_invocation_path: str | None = "",
        model_query_string: str | None = "",
        max_connect_attempts: int = 4,
    ):
        """Initialize the SageMaker BiDi client.

        Args:
            endpoint_name: Name of the SageMaker endpoint to connect to.
            region: AWS region where the endpoint is deployed.
            model_invocation_path: API path for the model invocation (e.g., "v1/listen").
            model_query_string: Query string parameters for the model (e.g., "model=nova-3").
            max_connect_attempts: How many times ``start_session`` attempts the
                session when SageMaker throttles it or fails transiently. The
                AWS SDK does not retry bidirectional streams itself.
        """
        self.endpoint_name = endpoint_name
        self.max_connect_attempts = max(1, max_connect_attempts)
        self.region = region
        self.model_invocation_path = model_invocation_path
        self.model_query_string = model_query_string
        self.bidi_endpoint = f"https://runtime.sagemaker.{region}.amazonaws.com:8443"
        self._client: SageMakerRuntimeHTTP2Client | None = None
        self._stream: (
            DuplexEventStream[
                RequestStreamEvent,
                ResponseStreamEvent,
                InvokeEndpointWithBidirectionalStreamOutput,
            ]
            | None
        ) = None
        self._output_stream = None
        self._is_active = False

    def _initialize_client(self) -> SageMakerRuntimeHTTP2Client:
        """Initialize the SageMaker Runtime HTTP2 client with AWS credentials.

        Creates and configures the SageMaker Runtime HTTP2 client with SigV4
        authentication. Attempts to resolve AWS credentials from environment
        variables, AWS CLI configuration, or instance metadata.

        Returns:
            The initialized client, also stored on the instance.
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

        # SigV4 auth for the sagemaker service is the Config default, so
        # auth_schemes and auth_scheme_resolver are left unset.
        config = Config(
            endpoint_uri=self.bidi_endpoint,
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
        )
        self._client = SageMakerRuntimeHTTP2Client(config=config)
        return self._client

    async def start_session(self):
        """Start a bidirectional streaming session with the SageMaker endpoint.

        Initializes the client if needed, creates the bidirectional stream, and
        establishes the connection to the SageMaker endpoint. Must be called
        before sending or receiving data. A throttled or transiently failed
        attempt is retried with jittered exponential backoff, up to
        ``max_connect_attempts`` attempts in total.

        Returns:
            The output stream for receiving responses.

        Raises:
            SageMakerBidiSessionError: If the session could not be started.
        """
        client = self._client or self._initialize_client()

        logger.debug(f"Starting BiDi session with endpoint: {self.endpoint_name}")
        logger.debug(f"Model invocation path: {self.model_invocation_path}")
        logger.debug(f"Model query string: {self.model_query_string}")

        # Create the bidirectional stream
        stream_input = InvokeEndpointWithBidirectionalStreamInput(
            endpoint_name=self.endpoint_name,
            model_invocation_path=self.model_invocation_path,
            model_query_string=self.model_query_string,
        )

        attempt = 0
        while True:
            attempt += 1
            try:
                self._stream = await client.invoke_endpoint_with_bidirectional_stream(stream_input)
                self._is_active = True

                # Get output stream
                output = await self._stream.await_output()
                self._output_stream = output[1]

                logger.debug("BiDi session started successfully")
                return self._output_stream

            except Exception as e:
                self._is_active = False
                category = classify_sagemaker_bidi_error(e)
                error_id, status = _error_id_and_status(e)
                # A rejected credential is rejected again on every attempt: the
                # client resolves its credentials once, so only a new session
                # (and client) can pick up refreshed ones.
                if (
                    category is not None
                    and category in _RETRYABLE_CATEGORIES
                    and status not in _AUTH_STATUSES
                    and attempt < self.max_connect_attempts
                ):
                    wait = random.uniform(
                        0,
                        exponential_backoff_time(
                            attempt,
                            min_wait=0,
                            max_wait=_CONNECT_BACKOFF_MAX_WAIT,
                            multiplier=_CONNECT_BACKOFF_MULTIPLIER,
                        ),
                    )
                    logger.warning(
                        f"BiDi session attempt {attempt}/{self.max_connect_attempts} failed "
                        f"({category.value}), retrying in {wait:.2f}s: {e}"
                    )
                    await asyncio.sleep(wait)
                    continue

                logger.error(f"Failed to start BiDi session after {attempt} attempt(s): {e}")
                raise SageMakerBidiSessionError(
                    f"Failed to start SageMaker BiDi session: {e}",
                    error_id=error_id,
                    http_status=status,
                    attempts=attempt,
                ) from e

    async def send_data(self, data_bytes: bytes, data_type: str | None = None):
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

    async def receive_response(self) -> ResponseStreamEvent | None:
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
