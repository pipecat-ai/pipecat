import asyncio
from typing import AsyncGenerator, Optional, Dict
import os
import datetime
from urllib.parse import urlencode
import json
import struct
import urllib.parse
import hashlib
import hmac
import random
import string
import binascii

from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TranscriptionFrame,
    InterimTranscriptionFrame,
    StartFrame
)
from pipecat.services.ai_services import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use AWS services, you need to `pip install pipecat-ai[aws]`. Also, remember to set `AWS_SECRET_ACCESS_KEY`, `AWS_ACCESS_KEY_ID`, and `AWS_REGION` environment variable."
    )
    raise Exception(f"Missing module: {e}")


def get_presigned_url(
    *,
    region: str,
    credentials: Dict[str, Optional[str]],
    language_code: str,
    media_encoding: str = "pcm",
    sample_rate: int = 16000,
    number_of_channels: int = 1,
    enable_partial_results_stabilization: bool = True,
    partial_results_stability: str = "high",
    vocabulary_name: Optional[str] = None,
    vocabulary_filter_name: Optional[str] = None,
    show_speaker_label: bool = False,
    enable_channel_identification: bool = False,
) -> str:
    """Create a presigned URL for AWS Transcribe streaming."""
    access_key = credentials.get("access_key")
    secret_key = credentials.get("secret_key")
    session_token = credentials.get("session_token")

    if not access_key or not secret_key:
        raise ValueError("AWS credentials are required")

    # Initialize the URL generator
    url_generator = AWSTranscribePresignedURL(
        access_key=access_key, secret_key=secret_key, session_token=session_token, region=region
    )

    # Get the presigned URL
    return url_generator.get_request_url(
        sample_rate=sample_rate,
        language_code=language_code,
        media_encoding=media_encoding,
        vocabulary_name=vocabulary_name,
        vocabulary_filter_name=vocabulary_filter_name,
        show_speaker_label=show_speaker_label,
        enable_channel_identification=enable_channel_identification,
        number_of_channels=number_of_channels,
        enable_partial_results_stabilization=enable_partial_results_stabilization,
        partial_results_stability=partial_results_stability,
    )


class AWSTranscribePresignedURL:
    def __init__(
        self, access_key: str, secret_key: str, session_token: str, region: str = "us-east-1"
    ):
        self.access_key = access_key
        self.secret_key = secret_key
        self.session_token = session_token
        self.method = "GET"
        self.service = "transcribe"
        self.region = region
        self.endpoint = ""
        self.host = ""
        self.amz_date = ""
        self.datestamp = ""
        self.canonical_uri = "/stream-transcription-websocket"
        self.canonical_headers = ""
        self.signed_headers = "host"
        self.algorithm = "AWS4-HMAC-SHA256"
        self.credential_scope = ""
        self.canonical_querystring = ""
        self.payload_hash = ""
        self.canonical_request = ""
        self.string_to_sign = ""
        self.signature = ""
        self.request_url = ""

    def get_request_url(
        self,
        sample_rate: int,
        language_code: str = "",
        media_encoding: str = "pcm",
        vocabulary_name: str = "",
        vocabulary_filter_name: str = "",
        show_speaker_label: bool = False,
        enable_channel_identification: bool = False,
        number_of_channels: int = 1,
        enable_partial_results_stabilization: bool = False,
        partial_results_stability: str = "",
    ) -> str:
        self.endpoint = f"wss://transcribestreaming.{self.region}.amazonaws.com:8443"
        self.host = f"transcribestreaming.{self.region}.amazonaws.com:8443"

        now = datetime.datetime.utcnow()
        self.amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        self.datestamp = now.strftime("%Y%m%d")
        self.canonical_headers = f"host:{self.host}\n"
        self.credential_scope = f"{self.datestamp}%2F{self.region}%2F{self.service}%2Faws4_request"

        # Create canonical querystring
        self.canonical_querystring = "X-Amz-Algorithm=" + self.algorithm
        self.canonical_querystring += (
            "&X-Amz-Credential=" + self.access_key + "%2F" + self.credential_scope
        )
        self.canonical_querystring += "&X-Amz-Date=" + self.amz_date
        self.canonical_querystring += "&X-Amz-Expires=300"
        if self.session_token:
            self.canonical_querystring += "&X-Amz-Security-Token=" + urllib.parse.quote(
                self.session_token, safe=""
            )
        self.canonical_querystring += "&X-Amz-SignedHeaders=" + self.signed_headers

        if enable_channel_identification:
            self.canonical_querystring += "&enable-channel-identification=true"
        if enable_partial_results_stabilization:
            self.canonical_querystring += "&enable-partial-results-stabilization=true"
        if language_code:
            self.canonical_querystring += "&language-code=" + language_code
        if media_encoding:
            self.canonical_querystring += "&media-encoding=" + media_encoding
        if number_of_channels > 1:
            self.canonical_querystring += "&number-of-channels=" + str(number_of_channels)
        if partial_results_stability:
            self.canonical_querystring += "&partial-results-stability=" + partial_results_stability
        if sample_rate:
            self.canonical_querystring += "&sample-rate=" + str(sample_rate)
        if show_speaker_label:
            self.canonical_querystring += "&show-speaker-label=true"
        if vocabulary_filter_name:
            self.canonical_querystring += "&vocabulary-filter-name=" + vocabulary_filter_name
        if vocabulary_name:
            self.canonical_querystring += "&vocabulary-name=" + vocabulary_name

        # Create payload hash
        self.payload_hash = hashlib.sha256("".encode("utf-8")).hexdigest()

        # Create canonical request
        self.canonical_request = f"{self.method}\n{self.canonical_uri}\n{self.canonical_querystring}\n{self.canonical_headers}\n{self.signed_headers}\n{self.payload_hash}"

        # Create string to sign
        credential_scope = f"{self.datestamp}/{self.region}/{self.service}/aws4_request"
        string_to_sign = (
            f"{self.algorithm}\n{self.amz_date}\n{credential_scope}\n"
            + hashlib.sha256(self.canonical_request.encode("utf-8")).hexdigest()
        )

        # Calculate signature
        k_date = hmac.new(
            f"AWS4{self.secret_key}".encode("utf-8"), self.datestamp.encode("utf-8"), hashlib.sha256
        ).digest()
        k_region = hmac.new(k_date, self.region.encode("utf-8"), hashlib.sha256).digest()
        k_service = hmac.new(k_region, self.service.encode("utf-8"), hashlib.sha256).digest()
        k_signing = hmac.new(k_service, b"aws4_request", hashlib.sha256).digest()
        self.signature = hmac.new(
            k_signing, string_to_sign.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        # Add signature to query string
        self.canonical_querystring += "&X-Amz-Signature=" + self.signature

        # Create request URL
        self.request_url = self.endpoint + self.canonical_uri + "?" + self.canonical_querystring
        return self.request_url


def get_headers(header_name: str, header_value: str) -> bytearray:
    """Build a header following AWS event stream format."""
    name = header_name.encode("utf-8")
    name_byte_length = bytes([len(name)])
    value_type = bytes([7])  # 7 represents a string
    value = header_value.encode("utf-8")
    value_byte_length = struct.pack(">H", len(value))

    # Construct the header
    header_list = bytearray()
    header_list.extend(name_byte_length)
    header_list.extend(name)
    header_list.extend(value_type)
    header_list.extend(value_byte_length)
    header_list.extend(value)
    return header_list


def build_event_message(payload: bytes) -> bytes:
    """
    Build an event message for AWS Transcribe streaming.
    Matches AWS sample: https://github.com/aws-samples/amazon-transcribe-streaming-python-websockets/blob/main/eventstream.py
    """
    # Build headers
    content_type_header = get_headers(":content-type", "application/octet-stream")
    event_type_header = get_headers(":event-type", "AudioEvent")
    message_type_header = get_headers(":message-type", "event")

    headers = bytearray()
    headers.extend(content_type_header)
    headers.extend(event_type_header)
    headers.extend(message_type_header)

    # Calculate total byte length and headers byte length
    # 16 accounts for 8 byte prelude, 2x 4 byte CRCs
    total_byte_length = struct.pack(">I", len(headers) + len(payload) + 16)
    headers_byte_length = struct.pack(">I", len(headers))

    # Build the prelude
    prelude = bytearray([0] * 8)
    prelude[:4] = total_byte_length
    prelude[4:] = headers_byte_length

    # Calculate checksum for prelude
    prelude_crc = struct.pack(">I", binascii.crc32(prelude) & 0xFFFFFFFF)

    # Construct the message
    message_as_list = bytearray()
    message_as_list.extend(prelude)
    message_as_list.extend(prelude_crc)
    message_as_list.extend(headers)
    message_as_list.extend(payload)

    # Calculate checksum for message
    message = bytes(message_as_list)
    message_crc = struct.pack(">I", binascii.crc32(message) & 0xFFFFFFFF)

    # Add message checksum
    message_as_list.extend(message_crc)

    return bytes(message_as_list)


def decode_event(message):
    # Extract the prelude, headers, payload and CRC
    prelude = message[:8]
    total_length, headers_length = struct.unpack(">II", prelude)
    prelude_crc = struct.unpack(">I", message[8:12])[0]
    headers = message[12 : 12 + headers_length]
    payload = message[12 + headers_length : -4]
    message_crc = struct.unpack(">I", message[-4:])[0]

    # Check the CRCs
    assert prelude_crc == binascii.crc32(prelude) & 0xFFFFFFFF, "Prelude CRC check failed"
    assert message_crc == binascii.crc32(message[:-4]) & 0xFFFFFFFF, "Message CRC check failed"

    # Parse the headers
    headers_dict = {}
    while headers:
        name_len = headers[0]
        name = headers[1 : 1 + name_len].decode("utf-8")
        value_type = headers[1 + name_len]
        value_len = struct.unpack(">H", headers[2 + name_len : 4 + name_len])[0]
        value = headers[4 + name_len : 4 + name_len + value_len].decode("utf-8")
        headers_dict[name] = value
        headers = headers[4 + name_len + value_len :]

    return headers_dict, json.loads(payload)


class TranscribeSTTService(STTService):
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

    async def run_stt(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        """Process audio data and send to AWS Transcribe"""
        try:
            # Skip if no speech detected
            if hasattr(frame, "is_speech") and not frame.is_speech:
                logger.debug("Skipping non-speech frame")
                return

            # Ensure WebSocket is connected
            if not self._ws_client or not self._ws_client.open:
                logger.info("WebSocket not connected, attempting to reconnect...")
                try:
                    await self._connect()
                except Exception as e:
                    logger.error(f"Failed to reconnect: {e}")
                    yield ErrorFrame("Failed to reconnect to AWS Transcribe", fatal=False)
                    return

            # Get the audio data - if frame is bytes, use directly, otherwise get audio attribute
            audio_data = frame if isinstance(frame, bytes) else frame.audio

            # Format the audio data according to AWS event stream format
            event_message = build_event_message(audio_data)
            # logger.debug(f"Sending audio chunk of size {len(audio_data)} bytes")

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
        if (
            self._ws_client
            and self._ws_client.open
            and self._receive_task
            and not self._receive_task.done()
        ):
            logger.debug("Already connected")
            return

        async with self._connection_lock:
            if self._connecting:
                logger.debug("Connection already in progress")
                return

            try:
                self._connecting = True
                logger.debug("Starting connection process...")

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

                logger.debug(f"Connecting to WebSocket with URL: {presigned_url[:100]}...")

                # Connect with the required headers and settings
                self._ws_client = await websockets.connect(
                    presigned_url,
                    extra_headers=extra_headers,
                    subprotocols=["mqtt"],
                    ping_interval=None,
                    ping_timeout=None,
                    compression=None,
                )
                logger.debug("WebSocket connected, starting receive task...")

                # Start receive task
                self._receive_task = asyncio.create_task(self._receive_loop())

                logger.info("Successfully connected to AWS Transcribe")

            except Exception as e:
                logger.error(f"Failed to connect to AWS Transcribe: {e}")
                await self._disconnect()
                raise

            finally:
                self._connecting = False

    async def _disconnect(self):
        """Disconnect from AWS Transcribe."""
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        if self._ws_client:
            try:
                if self._ws_client.open:
                    # Send end-stream message
                    end_stream = {"message-type": "event", "event": "end"}
                    await self._ws_client.send(json.dumps(end_stream))
                await self._ws_client.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket connection: {e}")
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
        }
        return language_map.get(language)

    async def _receive_loop(self):
        """Background task to receive and process messages from AWS Transcribe."""
        try:
            logger.debug("Receive loop started")
            while True:
                if not self._ws_client or not self._ws_client.open:
                    logger.warning("WebSocket closed in receive loop")
                    break

                try:
                    response = await self._ws_client.recv()
                    headers, payload = decode_event(response)

                    # logger.debug(f"Received message type: {headers.get(':message-type')}")

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
                                            )
                                        )
                                        await self.stop_processing_metrics()
                                    else:
                                        await self.push_frame(
                                            InterimTranscriptionFrame(
                                                transcript,
                                                "",
                                                time_now_iso8601(),
                                                self._settings["language"],
                                            )
                                        )
                    elif headers.get(":message-type") == "exception":
                        error_msg = payload.get("Message", "Unknown error")
                        logger.error(f"Exception from AWS: {error_msg}")
                        await self.push_frame(
                            ErrorFrame(f"AWS Transcribe error: {error_msg}", fatal=False)
                        )
                    else:
                        logger.debug(f"Other message type received: {headers}")
                        logger.debug(f"Payload: {payload}")

                except websockets.exceptions.ConnectionClosed as e:
                    logger.error(
                        f"WebSocket connection closed in receive loop with code {e.code}: {e.reason}"
                    )
                    break
                except Exception as e:
                    logger.error(f"Error in receive loop: {e}")
                    break

        except asyncio.CancelledError:
            logger.debug("Receive loop cancelled")
        except Exception as e:
            logger.error(f"Unexpected error in receive loop: {e}")
        finally:
            logger.debug("Receive loop ended")