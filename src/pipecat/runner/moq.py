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
from urllib.parse import urlparse

from loguru import logger

DEFAULT_MOQ_CONNECT = "https://localhost:4080/moq"
DEFAULT_MOQ_SERVE_BIND = "[::]:4080"
DEFAULT_MOQ_PATH = "/moq"


def _parse_bind_port(bind: str) -> int:
    """Parse the port from a bind address like ``[::]:4080`` or ``0.0.0.0:4080``."""
    _, sep, port_str = bind.rpartition(":")
    if not sep:
        raise ValueError(f"--moq-bind must include a port (got {bind!r})")
    try:
        return int(port_str)
    except ValueError as e:
        raise ValueError(f"--moq-bind has a non-numeric port: {bind!r}") from e


def _validate_moq_args(args: argparse.Namespace) -> bool:
    """Validate MoQ CLI args, warn on conflicts, and stash derived host/port/path on ``args``.

    Returns ``True`` if the args are usable, ``False`` if validation failed.

    Populates: ``args.moq_host``, ``args.moq_port``, ``args.moq_path``,
    ``args.moq_bind`` (defaulted in serve mode), ``args.moq_tls_host`` (the
    hostname presented to the browser).
    """
    # ------------------------------------------------------------------
    # TODO: MoQ client mode is not yet supported. The wiring exists but
    # hasn't been shaken out — the cert-fingerprint plumbing to the
    # browser is the missing piece for self-signed local relays, and we
    # haven't validated the flow against a public relay. Fail loudly at
    # arg-parse time so users get a clear message instead of a downstream
    # DNS or TLS error. Every --moq-connect / --moq-tls-insecure /
    # client-mode branch below is preserved so the switchover is a
    # one-line delete of this guard once client mode is ready.
    if not args.moq_serve:
        logger.error(
            "MoQ client mode is not yet supported. "
            "Pass --moq-serve to run the bot as its own MoQ server "
            "(with --moq-tls-generate <hostname> for a self-signed dev cert, or "
            "--moq-tls-cert/--moq-tls-key for a CA-signed one)."
        )
        return False
    # ------------------------------------------------------------------

    has_cert = bool(args.moq_tls_cert)
    has_key = bool(args.moq_tls_key)
    has_generate = bool(args.moq_tls_generate)

    if args.moq_serve:
        # Server mode.
        if args.moq_connect is not None:
            logger.warning(
                "--moq-connect is ignored in server mode (use --moq-bind to set the listen address)"
            )
        if args.moq_tls_insecure:
            logger.warning(
                "--moq-tls-insecure is ignored in server mode "
                "(server-side TLS is set via --moq-tls-cert or --moq-tls-generate)"
            )
        if has_cert != has_key:
            logger.error(
                "server mode requires both --moq-tls-cert AND --moq-tls-key "
                "(or use --moq-tls-generate <hostname> for a self-signed dev cert)"
            )
            return False
        if (has_cert and has_key) and has_generate:
            logger.warning(
                "--moq-tls-generate is ignored — using --moq-tls-cert/--moq-tls-key instead"
            )
        elif not (has_cert and has_key) and not has_generate:
            # No TLS config supplied at all — default to a self-signed dev
            # cert for localhost rather than requiring --moq-tls-generate.
            args.moq_tls_generate = "localhost"
            has_generate = True

        bind = args.moq_bind or DEFAULT_MOQ_SERVE_BIND
        try:
            bind_port = _parse_bind_port(bind)
        except ValueError as e:
            logger.error(str(e))
            return False

        # Hostname the browser uses to reach the bot. In dev with
        # --moq-tls-generate, that's the cert hostname. With a CA-signed
        # cert, fall back to localhost (operator can patch via env / code).
        tls_host = args.moq_tls_generate or "localhost"

        args.moq_bind = bind
        args.moq_host = tls_host
        args.moq_port = bind_port
        args.moq_path = DEFAULT_MOQ_PATH
        args.moq_tls_host = tls_host
    else:
        # Client mode.
        connect = args.moq_connect or DEFAULT_MOQ_CONNECT
        parsed = urlparse(connect)
        if not parsed.hostname:
            logger.error(
                f"--moq-connect must be a full URL with a host "
                f"(e.g. https://relay.example.com/moq); got {connect!r}"
            )
            return False
        # Default the port from the scheme so URLs on standard HTTPS/HTTP
        # ports (e.g. https://relay.quic.video/anon) don't have to spell
        # it out explicitly.
        default_ports = {"https": 443, "http": 80}
        client_port = parsed.port or default_ports.get(parsed.scheme.lower())
        if client_port is None:
            logger.error(
                f"--moq-connect needs an explicit port; scheme "
                f"{parsed.scheme!r} has no standard default (got {connect!r})"
            )
            return False
        client_host = parsed.hostname
        client_path = parsed.path or DEFAULT_MOQ_PATH

        if has_generate:
            logger.warning("--moq-tls-generate is ignored — only used in server mode")
        if has_key and not has_cert:
            logger.error("--moq-tls-key requires --moq-tls-cert")
            return False
        if has_key and has_cert:
            logger.warning(
                "--moq-tls-key is ignored in client mode (--moq-tls-cert is used for "
                "self-signed cert fingerprint pinning only)"
            )

        args.moq_host = client_host
        args.moq_port = client_port
        args.moq_path = client_path
        args.moq_tls_host = client_host

    return True


def _cert_hash_from_pem(path: str) -> str | None:
    """Compute the base64 SHA-256 of a PEM-encoded cert on disk.

    Used in client mode when ``--moq-tls-cert`` is set and we need the
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
    Otherwise we fall back to the PEM file at ``--moq-tls-cert``.

    Track names aren't pinned here — the bot publishes a catalog at
    runtime and the browser reads whatever it advertises (codec, sample
    rate, channel count, track name). Lets us add tracks (video,
    screen-share) without a server-side config update.
    """
    # certHash must be set ONLY for self-signed certs. It purposely
    # doesn't work with valid CA-signed TLS certificates.
    cert_hash: str | None = None
    if args.moq_serve and cert_fingerprints:
        cert_hash = _hex_to_b64(cert_fingerprints[0])
    elif getattr(args, "moq_tls_cert", None):
        cert_hash = _cert_hash_from_pem(args.moq_tls_cert)

    # WebTransport always uses HTTPS — even for self-signed dev relays,
    # the cert is pinned via `certHash` below.
    return {
        "relayUrl": f"https://{args.moq_host}:{args.moq_port}{args.moq_path}",
        "certHash": cert_hash,
        "serve": args.moq_serve,
        "namespace": namespace,
        "clientId": args.moq_client_id,
        "botId": args.moq_bot_id,
        "transcriptTrack": "transcript.json.z",
    }
