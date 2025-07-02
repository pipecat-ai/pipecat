#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AWS Transcribe utility functions and classes for WebSocket streaming.

This module provides utilities for creating presigned URLs, building event messages,
and handling AWS event stream protocol for real-time transcription services.
"""

import binascii
import datetime
import hashlib
import hmac
import json
import struct
import urllib.parse
from typing import Dict, Optional


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
    """Create a presigned URL for AWS Transcribe streaming.

    Args:
        region: AWS region for the service.
        credentials: Dictionary containing AWS credentials. Must include
            'access_key' and 'secret_key', with optional 'session_token'.
        language_code: Language code for transcription (e.g., "en-US").
        media_encoding: Audio encoding format. Defaults to "pcm".
        sample_rate: Audio sample rate in Hz. Defaults to 16000.
        number_of_channels: Number of audio channels. Defaults to 1.
        enable_partial_results_stabilization: Whether to enable partial result stabilization.
        partial_results_stability: Stability level for partial results.
        vocabulary_name: Custom vocabulary name to use.
        vocabulary_filter_name: Vocabulary filter name to apply.
        show_speaker_label: Whether to include speaker labels.
        enable_channel_identification: Whether to enable channel identification.

    Returns:
        Presigned WebSocket URL for AWS Transcribe streaming.

    Raises:
        ValueError: If required AWS credentials are missing.
    """
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
    """Generator for AWS Transcribe presigned WebSocket URLs.

    Handles AWS Signature Version 4 signing process to create authenticated
    WebSocket URLs for streaming transcription requests.
    """

    def __init__(
        self, access_key: str, secret_key: str, session_token: str, region: str = "us-east-1"
    ):
        """Initialize the presigned URL generator.

        Args:
            access_key: AWS access key ID.
            secret_key: AWS secret access key.
            session_token: AWS session token for temporary credentials.
            region: AWS region for the service. Defaults to "us-east-1".
        """
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
        """Generate a presigned WebSocket URL for AWS Transcribe.

        Args:
            sample_rate: Audio sample rate in Hz.
            language_code: Language code for transcription.
            media_encoding: Audio encoding format.
            vocabulary_name: Custom vocabulary name.
            vocabulary_filter_name: Vocabulary filter name.
            show_speaker_label: Whether to include speaker labels.
            enable_channel_identification: Whether to enable channel identification.
            number_of_channels: Number of audio channels.
            enable_partial_results_stabilization: Whether to enable partial result stabilization.
            partial_results_stability: Stability level for partial results.

        Returns:
            Presigned WebSocket URL with authentication parameters.
        """
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
    """Build a header following AWS event stream format.

    Args:
        header_name: Name of the header.
        header_value: Value of the header.

    Returns:
        Encoded header as a bytearray following AWS event stream protocol.
    """
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
    """Build an event message for AWS Transcribe streaming.

    Creates a properly formatted AWS event stream message containing audio data
    for real-time transcription. Follows the AWS event stream protocol with
    prelude, headers, payload, and CRC checksums.

    Args:
        payload: Raw audio bytes to include in the event message.

    Returns:
        Complete event message as bytes, ready to send via WebSocket.

    Note:
        Implementation matches AWS sample:
        https://github.com/aws-samples/amazon-transcribe-streaming-python-websockets/blob/main/eventstream.py
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
    """Decode an AWS event stream message.

    Parses an AWS event stream message to extract headers and payload,
    verifying CRC checksums for data integrity.

    Args:
        message: Raw event stream message bytes received from AWS.

    Returns:
        A tuple of (headers, payload) where:

        - headers: Dictionary of parsed headers
        - payload: Dictionary of parsed JSON payload

    Raises:
        AssertionError: If CRC checksum verification fails.
    """
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
