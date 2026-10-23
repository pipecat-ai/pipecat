#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Shared helpers for Krisp's streaming STT and TTS APIs.

Both WebSocket APIs authenticate with a short-lived session token created from
the account's master API key and scoped to one product (``stt`` or ``tts``),
and both open a session with a JSON handshake that the server answers with
``ready`` or ``error``.
"""

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp

KRISP_REST_BASE_URL = "https://api.developers.krisp.ai"
SESSION_TOKEN_PATH = "/v2/sdk/tokens/session"

HANDSHAKE_TIMEOUT = 10.0

# A session token can open any number of WebSocket sessions until it expires,
# so connections reuse one per (API key, product, REST base URL) instead of
# requesting a token for every connect, e.g. reconnects after an interruption.
_CACHED_TOKEN_TTL_MINUTES = 60
_CACHED_TOKEN_REUSE_SECS = (_CACHED_TOKEN_TTL_MINUTES - 1) * 60
_token_cache: dict[tuple[str, str, str], tuple[str, float]] = {}


class KrispServiceError(Exception):
    """An error reported by a Krisp REST or streaming API.

    ``status_code`` follows HTTP semantics, so pipecat classifies
    authentication (401/403) and invalid-request (400) errors as permanent.
    """

    def __init__(self, message: str, *, code: str | None = None, status_code: int | None = None):
        """Initialize the error.

        Args:
            message: Human-readable error message.
            code: Krisp error code (e.g. ``invalid_api_key``).
            status_code: HTTP-style status code reported by the server.
        """
        super().__init__(message)
        self.code = code
        self.status_code = status_code

    @property
    def is_permanent(self) -> bool:
        """Whether retrying can't help (rejected credentials or configuration)."""
        return self.status_code in (400, 401, 403)

    @classmethod
    def from_message(cls, msg: dict[str, Any]) -> "KrispServiceError":
        """Build an error from a server ``error`` message.

        Args:
            msg: The decoded ``{"type": "error", "error": {...}}`` message.

        Returns:
            The corresponding error.
        """
        error = msg.get("error") or {}
        hint = (error.get("details") or {}).get("hint")
        message = error.get("message") or "Unknown error"
        if hint:
            message = f"{message} ({hint})"
        return cls(
            f"{error.get('code', 'error')}: {message}",
            code=error.get("code"),
            status_code=error.get("status_code"),
        )


async def create_session_token(
    api_key: str,
    usage_type: str,
    *,
    expiration_ttl: int | None = None,
    session_id: str | None = None,
    base_url: str = KRISP_REST_BASE_URL,
    session: aiohttp.ClientSession | None = None,
) -> str:
    """Create a short-lived session token for a streaming WebSocket API.

    Args:
        api_key: Krisp master API key.
        usage_type: Product the token is scoped to: ``stt``, ``tts``, ``viva``
            or ``vt``.
        expiration_ttl: Token lifetime in minutes (5-300). None uses the
            server default.
        session_id: Optional caller-defined session identifier (up to 255
            characters), stored as ``metadata.session_id``.
        base_url: Krisp REST API base URL.
        session: Optional aiohttp session to reuse.

    Returns:
        The session token.

    Raises:
        KrispServiceError: If the server rejects the request or the response
            does not contain a token.
    """
    payload: dict[str, Any] = {"usage_type": usage_type}
    if expiration_ttl is not None:
        payload["expiration_ttl"] = expiration_ttl
    if session_id is not None:
        payload["metadata"] = {"session_id": session_id}
    body = await _request_json(
        "POST", f"{base_url}{SESSION_TOKEN_PATH}", api_key, json_body=payload, session=session
    )
    token = (body.get("data") or {}).get("session_key")
    if not token:
        raise KrispServiceError(f"Unexpected session token response: {body!r}")
    return token


async def _request_json(
    method: str,
    url: str,
    api_key: str,
    *,
    params: dict[str, str] | None = None,
    json_body: dict[str, Any] | None = None,
    session: aiohttp.ClientSession | None,
) -> dict[str, Any]:
    if session is None:
        async with aiohttp.ClientSession() as owned_session:
            return await _request_json(
                method, url, api_key, params=params, json_body=json_body, session=owned_session
            )
    headers = {"Authorization": f"api-key {api_key}"}
    async with session.request(method, url, headers=headers, params=params, json=json_body) as resp:
        if resp.status >= 400:
            try:
                body = await resp.json(content_type=None)
            except Exception:
                body = {}
            body = body if isinstance(body, dict) else {}
            code = body.get("error_code")
            message = body.get("message") or resp.reason or "request failed"
            raise KrispServiceError(
                f"{code or resp.status}: {message} ({method} {url}, req_id={body.get('req_id')})",
                code=code,
                status_code=resp.status,
            )
        return await resp.json(content_type=None)


def session_headers(session_token: str) -> dict[str, str]:
    """WebSocket headers authenticating with a session token."""
    return {"Authorization": f"api-key {session_token}"}


async def open_session(websocket, handshake: dict[str, Any]) -> dict[str, Any]:
    """Send the handshake and wait for the server's ``ready`` message.

    Args:
        websocket: The freshly opened WebSocket.
        handshake: The handshake message.

    Returns:
        The decoded ``ready`` message.

    Raises:
        KrispServiceError: If the server answers with an ``error`` message.
        TimeoutError: If ``ready`` doesn't arrive within ``HANDSHAKE_TIMEOUT``.
    """
    await websocket.send(json.dumps(handshake))
    # One deadline for the whole handshake, so other messages can't extend it.
    async with asyncio.timeout(HANDSHAKE_TIMEOUT):
        while True:
            raw = await websocket.recv()
            if isinstance(raw, bytes):
                continue
            msg = json.loads(raw)
            if msg.get("type") == "ready":
                return msg
            if msg.get("type") == "error":
                raise KrispServiceError.from_message(msg)


async def _session_token(
    api_key: str, usage_type: str, rest_base_url: str, *, refresh: bool = False
) -> tuple[str, bool]:
    """Return a session token, reusing a cached one until shortly before it expires.

    Returns:
        The token and whether it came from the cache.
    """
    key = (api_key, usage_type, rest_base_url)
    cached = _token_cache.get(key)
    if cached and not refresh and time.monotonic() < cached[1]:
        return cached[0], True
    token = await create_session_token(
        api_key, usage_type, expiration_ttl=_CACHED_TOKEN_TTL_MINUTES, base_url=rest_base_url
    )
    _token_cache[key] = (token, time.monotonic() + _CACHED_TOKEN_REUSE_SECS)
    return token, False


async def connect_session(
    connect: Callable[..., Awaitable[Any]],
    url: str,
    *,
    api_key: str,
    usage_type: str,
    rest_base_url: str,
    handshake: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Open the WebSocket with a session token and complete the handshake.

    The session token is cached and reused across connections. If the server
    rejects a cached token, a fresh one is requested and the connect retried
    once.

    Args:
        connect: Coroutine function opening the WebSocket, called as
            ``connect(url, additional_headers=...)``.
        url: WebSocket endpoint.
        api_key: Krisp API key used to create the session token.
        usage_type: Product the token is scoped to (``stt``, ``tts``, ``viva``).
        rest_base_url: Krisp REST API base URL.
        handshake: The handshake message.

    Returns:
        The WebSocket, ready for streaming, and the server's ``ready`` message.
        The WebSocket is closed if the handshake fails.
    """
    refresh = False
    while True:
        token, cached = await _session_token(api_key, usage_type, rest_base_url, refresh=refresh)
        websocket = await connect(url, additional_headers=session_headers(token))
        try:
            return websocket, await open_session(websocket, handshake)
        except KrispServiceError as e:
            await websocket.close()
            if cached and e.status_code in (401, 403):
                refresh = True
                continue
            raise
        except BaseException:
            await websocket.close()
            raise
