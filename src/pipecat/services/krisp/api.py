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
from typing import Any

import aiohttp

KRISP_REST_BASE_URL = "https://api.developers.krisp.ai"
SESSION_TOKEN_PATH = "/v2/sdk/tokens/session"
TTS_VOICES_PATH = "/v2/sdk/tts/voices"

HANDSHAKE_TIMEOUT = 10.0


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


async def list_tts_voices(
    api_key: str,
    *,
    model_name: str = "krisp-tts-v1",
    gender: str | None = None,
    language: str | None = None,
    search: str | None = None,
    base_url: str = KRISP_REST_BASE_URL,
    session: aiohttp.ClientSession | None = None,
) -> list[dict[str, Any]]:
    """Fetch the Krisp TTS voice catalog.

    Each voice's ``name`` is the value to use as the TTS ``voice`` setting.

    Args:
        api_key: Krisp master API key.
        model_name: TTS model to list voices for.
        gender: Optional filter: ``female``, ``male`` or ``neutral``.
        language: Optional filter to voices that can speak this language.
        search: Optional case-insensitive substring match on the voice name.
        base_url: Krisp REST API base URL.
        session: Optional aiohttp session to reuse.

    Returns:
        The catalog entries.
    """
    params = {"model_name": model_name, "gender": gender, "language": language, "search": search}
    body = await _request_json(
        "GET",
        f"{base_url}{TTS_VOICES_PATH}",
        api_key,
        params={k: v for k, v in params.items() if v is not None},
        session=session,
    )
    return body.get("data") or []


async def select_tts_voice(
    api_key: str,
    language: str,
    *,
    voice: str | None = None,
    gender: str | None = None,
    model_name: str = "krisp-tts-v1",
    base_url: str = KRISP_REST_BASE_URL,
    session: aiohttp.ClientSession | None = None,
) -> str:
    """Pick a catalog voice that can speak ``language``.

    Call it before creating :class:`~pipecat.services.krisp.tts.KrispTTSService`
    and pass the result as the ``voice`` setting together with the same
    ``language``, so the session handshake isn't rejected with
    ``invalid_voice`` or ``invalid_language``.

    Args:
        api_key: Krisp master API key.
        language: Language the bot will speak, e.g. ``de-DE``.
        voice: Preferred voice name. It is validated against the catalog and
            returned in its canonical spelling.
        gender: Optional filter used when no ``voice`` is given.
        model_name: TTS model to select a voice for.
        base_url: Krisp REST API base URL.
        session: Optional aiohttp session to reuse.

    Returns:
        The voice name to pass as the ``voice`` setting.

    Raises:
        KrispServiceError: If no voice (or not the preferred one) speaks
            ``language``.
    """
    voices = await list_tts_voices(
        api_key,
        model_name=model_name,
        language=language,
        gender=None if voice else gender,
        base_url=base_url,
        session=session,
    )
    names = [v["name"] for v in voices if v.get("name")]
    if voice:
        for name in names:
            if name.lower() == voice.lower():
                return name
        raise KrispServiceError(
            f"Voice {voice!r} does not speak {language}; available: {', '.join(names) or 'none'}"
        )
    if not names:
        raise KrispServiceError(f"No {model_name} voice speaks {language}")
    return names[0]


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
    """
    await websocket.send(json.dumps(handshake))
    while True:
        raw = await asyncio.wait_for(websocket.recv(), HANDSHAKE_TIMEOUT)
        if isinstance(raw, bytes):
            continue
        msg = json.loads(raw)
        if msg.get("type") == "ready":
            return msg
        if msg.get("type") == "error":
            raise KrispServiceError.from_message(msg)
