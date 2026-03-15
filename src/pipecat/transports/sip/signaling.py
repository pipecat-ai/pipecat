#
# Copyright (c) 2024-2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""SIP message parsing and response building.

Handles the minimum SIP signaling needed for a UAS: parsing incoming
INVITE/BYE/ACK requests and building 100 Trying, 200 OK, and BYE responses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from pipecat.transports.sip.sdp import generate_sdp


class SIPMethod(Enum):
    """Supported SIP request methods."""

    INVITE = "INVITE"
    ACK = "ACK"
    BYE = "BYE"
    CANCEL = "CANCEL"
    OPTIONS = "OPTIONS"


@dataclass
class SIPMessage:
    """Parsed SIP message (request or response).

    Parameters:
        method: SIP method for requests (None for responses).
        request_uri: Request URI (None for responses).
        status_code: Status code for responses (None for requests).
        headers: Dict of header name -> value.
        body: Message body (SDP, etc.) or None.
    """

    method: Optional[SIPMethod]
    request_uri: Optional[str]
    status_code: Optional[int]
    headers: Dict[str, str]
    body: Optional[str]

    @property
    def call_id(self) -> str:
        """Return the Call-ID header value."""
        return self.headers.get("Call-ID", "")

    @property
    def from_header(self) -> str:
        """Return the From header value."""
        return self.headers.get("From", "")

    @property
    def to_header(self) -> str:
        """Return the To header value."""
        return self.headers.get("To", "")

    @property
    def via(self) -> str:
        """Return the Via header value."""
        return self.headers.get("Via", "")

    @property
    def cseq(self) -> str:
        """Return the CSeq header value."""
        return self.headers.get("CSeq", "")

    @classmethod
    def parse(cls, data: bytes) -> SIPMessage:
        """Parse raw bytes into a SIPMessage.

        Args:
            data: Raw SIP message bytes.

        Returns:
            Parsed SIPMessage.
        """
        text = data.decode("utf-8", errors="replace")
        head, _, body = text.partition("\r\n\r\n")
        lines = head.split("\r\n")
        first_line = lines[0]

        method = None
        request_uri = None
        status_code = None

        if first_line.startswith("SIP/"):
            parts = first_line.split(" ", 2)
            status_code = int(parts[1])
        else:
            parts = first_line.split(" ", 2)
            try:
                method = SIPMethod(parts[0])
            except ValueError:
                method = None
            request_uri = parts[1] if len(parts) > 1 else None

        headers: Dict[str, str] = {}
        for line in lines[1:]:
            if ":" in line:
                key, _, value = line.partition(":")
                headers[key.strip()] = value.strip()

        return cls(
            method=method,
            request_uri=request_uri,
            status_code=status_code,
            headers=headers,
            body=body.strip() if body.strip() else None,
        )


def build_100_trying(*, invite: SIPMessage) -> bytes:
    """Build a 100 Trying response to an INVITE.

    Args:
        invite: The parsed INVITE message.

    Returns:
        Encoded SIP response bytes.
    """
    response = (
        f"SIP/2.0 100 Trying\r\n"
        f"Via: {invite.via}\r\n"
        f"From: {invite.from_header}\r\n"
        f"To: {invite.to_header}\r\n"
        f"Call-ID: {invite.call_id}\r\n"
        f"CSeq: {invite.cseq}\r\n"
        f"Content-Length: 0\r\n"
        f"\r\n"
    )
    return response.encode("utf-8")


def build_200_ok(
    *,
    invite: SIPMessage,
    local_ip: str,
    local_port: int,
    session_id: int,
    local_sip_port: int = 5060,
) -> bytes:
    """Build a 200 OK response to an INVITE with SDP answer.

    Args:
        invite: The parsed INVITE message.
        local_ip: Local IP for SDP.
        local_port: Local RTP port for SDP.
        session_id: Session ID for SDP and To tag.
        local_sip_port: Local SIP port for the Contact header.

    Returns:
        Encoded SIP response bytes with SDP body.
    """
    sdp = generate_sdp(local_ip=local_ip, local_port=local_port, session_id=session_id)
    sdp_bytes = sdp.encode("utf-8")
    response = (
        f"SIP/2.0 200 OK\r\n"
        f"Via: {invite.via}\r\n"
        f"From: {invite.from_header}\r\n"
        f"To: {invite.to_header};tag=bot-{session_id}\r\n"
        f"Call-ID: {invite.call_id}\r\n"
        f"CSeq: {invite.cseq}\r\n"
        f"Contact: <sip:pipecat@{local_ip}:{local_sip_port}>\r\n"
        f"Content-Type: application/sdp\r\n"
        f"Content-Length: {len(sdp_bytes)}\r\n"
        f"\r\n"
    )
    return response.encode("utf-8") + sdp_bytes


def build_200_ok_bye(*, bye: SIPMessage) -> bytes:
    """Build a 200 OK response to a BYE request.

    Args:
        bye: The parsed BYE message.

    Returns:
        Encoded SIP response bytes.
    """
    response = (
        f"SIP/2.0 200 OK\r\n"
        f"Via: {bye.via}\r\n"
        f"From: {bye.from_header}\r\n"
        f"To: {bye.to_header}\r\n"
        f"Call-ID: {bye.call_id}\r\n"
        f"CSeq: {bye.cseq}\r\n"
        f"Content-Length: 0\r\n"
        f"\r\n"
    )
    return response.encode("utf-8")


def build_200_ok_options(*, options: SIPMessage) -> bytes:
    """Build a 200 OK response to an OPTIONS keepalive probe.

    Args:
        options: The parsed OPTIONS message.

    Returns:
        Encoded SIP response bytes.
    """
    response = (
        f"SIP/2.0 200 OK\r\n"
        f"Via: {options.via}\r\n"
        f"From: {options.from_header}\r\n"
        f"To: {options.to_header}\r\n"
        f"Call-ID: {options.call_id}\r\n"
        f"CSeq: {options.cseq}\r\n"
        f"Allow: INVITE, ACK, BYE, CANCEL, OPTIONS\r\n"
        f"Content-Length: 0\r\n"
        f"\r\n"
    )
    return response.encode("utf-8")


def _extract_uri(header: str) -> str:
    """Extract SIP URI from a header like '<sip:user@host>;tag=...'."""
    m = re.search(r"<(sip:[^>]+)>", header)
    return m.group(1) if m else "sip:unknown@0.0.0.0"


def _append_tag(header: str, tag: str) -> str:
    """Append ;tag=... to a SIP header if not already present."""
    if "tag=" in header:
        return header
    return f"{header};tag={tag}"


def build_bye(
    *,
    call_id: str,
    from_header: str,
    to_header: str,
    local_tag: str,
    local_ip: str,
    local_sip_port: int,
) -> bytes:
    """Build a UAS-initiated BYE request.

    In a UAS-initiated BYE, From/To are swapped relative to the original INVITE:
    From = us (original To + our tag), To = remote (original From with their tag).

    Args:
        call_id: Call-ID from the original INVITE.
        from_header: Original INVITE From header (remote party, has their tag).
        to_header: Original INVITE To header (us, may not have tag).
        local_tag: Our tag from the 200 OK.
        local_ip: Our IP for the Via header.
        local_sip_port: Our SIP port for the Via header.

    Returns:
        Encoded BYE request bytes.
    """
    remote_uri = _extract_uri(from_header)
    local_from = _append_tag(to_header, local_tag)
    request = (
        f"BYE {remote_uri} SIP/2.0\r\n"
        f"Via: SIP/2.0/UDP {local_ip}:{local_sip_port};branch=z9hG4bK{call_id[:8]}\r\n"
        f"From: {local_from}\r\n"
        f"To: {from_header}\r\n"
        f"Call-ID: {call_id}\r\n"
        f"CSeq: 1 BYE\r\n"
        f"Content-Length: 0\r\n"
        f"\r\n"
    )
    return request.encode("utf-8")
