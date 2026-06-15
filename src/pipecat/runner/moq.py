#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""MoQ runner helpers.

Configuration helpers used by the development runner to construct the
MoQ relay config sent to the browser at ``/start``.
"""

import argparse
from typing import Any

from loguru import logger


def _cert_hash_from_pem(path: str) -> str | None:
    """Compute the base64 SHA-256 of a PEM-encoded cert on disk.

    Used in client mode when ``--moq-cert`` is set and we need the
    fingerprint to send to the browser for WebTransport pinning.
    """
    try:
        import base64
        import hashlib

        from cryptography import x509
        from cryptography.hazmat.primitives import serialization

        with open(path, "rb") as f:
            cert = x509.load_pem_x509_certificate(f.read())
        der_bytes = cert.public_bytes(serialization.Encoding.DER)
        digest = hashlib.sha256(der_bytes).digest()
        return base64.b64encode(digest).decode()
    except Exception as e:
        logger.warning(f"Could not compute cert fingerprint from {path}: {e}")
        return None


def _hex_to_b64(hex_str: str) -> str | None:
    """Convert a hex-encoded SHA-256 digest to base64.

    The bot exposes its serve-mode cert fingerprints as hex (the moq
    library's format); the browser's ``serverCertificateHashes`` expects
    base64 of the raw bytes.
    """
    try:
        import base64

        return base64.b64encode(bytes.fromhex(hex_str)).decode()
    except Exception as e:
        logger.warning(f"Could not convert cert hash {hex_str!r}: {e}")
        return None


def _build_moq_client_config(
    args: argparse.Namespace,
    namespace: str,
    cert_fingerprints: list[str] | None = None,
) -> dict[str, Any]:
    """Build the MoQ relay config the browser needs to construct a transport.

    Returned from POST /start (under the ``moq`` key) so the React UI can
    pipe it into ``MoqTransport``'s constructor without a separate fetch.

    In serve mode the bot just minted (or loaded) its own cert; we use
    the fingerprint it reported (passed via ``cert_fingerprints``).
    Otherwise we fall back to the PEM file at ``--moq-cert``.

    Track names aren't pinned here — the bot publishes a catalog at
    runtime and the browser reads whatever it advertises (codec, sample
    rate, channel count, track name). Lets us add tracks (video,
    screen-share) without a server-side config update.
    """
    # note this must be set ONLY if a self-signed certificate is used.
    # It purposely doesn't work with valid TLS certficates.
    cert_hash: str | None = None
    if args.moq_serve and cert_fingerprints:
        cert_hash = _hex_to_b64(cert_fingerprints[0])
    elif getattr(args, "moq_cert", None):
        cert_hash = _cert_hash_from_pem(args.moq_cert)

    # WebTransport always uses HTTPS — even for self-signed dev relays,
    # the cert is pinned via `certHash` below.
    return {
        "relayUrl": f"https://{args.moq_host}:{args.moq_port}{args.moq_path}",
        "certHash": cert_hash,
        "serve": args.moq_serve,
        "namespace": namespace,
        "clientId": args.moq_client_id,
        "botId": args.moq_bot_id,
        "transcriptTrack": "transcript",
    }
