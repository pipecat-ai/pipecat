#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Resemble AI Signal client implementations."""

from __future__ import annotations

import inspect
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import aiohttp

RESEMBLE_SIGNAL_API_URL = "https://app.resemble.ai/api/v2"

SignalVerdict = Literal["safe", "suspicious", "fraud"]
SignalModality = Literal["text", "audio", "video", "image"]
SignalAction = Literal["allow", "review", "block"]


class SignalTransport(Protocol):
    """Transport used by :class:`ResembleSignal` to call Resemble Signal."""

    async def score_text(self, text: str, *, request_timeout: float) -> dict[str, Any]:
        """Score text and return the Signal item payload."""

    async def score_file(
        self,
        file: bytes,
        *,
        filename: str,
        media_type: SignalModality | None,
        content_type: str | None,
        request_timeout: float,
    ) -> dict[str, Any]:
        """Score a media file and return the Signal item payload."""

    async def list_submissions(
        self,
        *,
        page: int,
        per_page: int,
        request_timeout: float,
    ) -> dict[str, Any]:
        """List historical Signal submissions."""

    async def delete_submission(self, submission_id: str | int, *, request_timeout: float) -> None:
        """Delete a historical Signal submission."""

    async def list_custom_categories(self, *, request_timeout: float) -> dict[str, Any]:
        """List built-in and custom Signal categories."""

    async def create_custom_category(
        self,
        payload: Mapping[str, Any],
        *,
        request_timeout: float,
    ) -> dict[str, Any]:
        """Create a custom Signal category."""

    async def get_custom_category(
        self,
        category_id: str | int,
        *,
        request_timeout: float,
    ) -> dict[str, Any]:
        """Fetch one custom Signal category."""

    async def update_custom_category(
        self,
        category_id: str | int,
        payload: Mapping[str, Any],
        *,
        request_timeout: float,
    ) -> dict[str, Any]:
        """Update one custom Signal category."""

    async def delete_custom_category(
        self, category_id: str | int, *, request_timeout: float
    ) -> None:
        """Delete one custom Signal category."""

    async def update_settings(
        self,
        payload: Mapping[str, Any],
        *,
        request_timeout: float,
    ) -> dict[str, Any]:
        """Update team-level Signal settings."""


@dataclass
class SignalCategoryScore:
    """A category score returned by Resemble Signal."""

    name: str
    score: float
    icon: str | None = None
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable dictionary representation."""
        return {
            "name": self.name,
            "score": self.score,
            "icon": self.icon,
        }


@dataclass
class SignalResult:
    """Outcome of scoring content with Resemble Signal."""

    verdict: SignalVerdict
    input_modality: SignalModality
    id: str | int | None = None
    top_category: SignalCategoryScore | None = None
    category_scores: list[SignalCategoryScore] = field(default_factory=list)
    benign_score: float | None = None
    margin_over_second: float | None = None
    examples: list[str] = field(default_factory=list)
    top_matches: list[dict[str, Any]] = field(default_factory=list)
    duration_seconds: float | None = None
    created_at: str | None = None
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def score(self) -> float:
        """Return the best fraud/scam category score in ``[0, 1]`` when available."""
        if self.top_category is not None:
            return self.top_category.score
        if self.benign_score is not None:
            return max(0.0, min(1.0, 1.0 - self.benign_score))
        return 0.0

    @property
    def recommended_action(self) -> SignalAction:
        """Return a conservative app action for the Signal verdict."""
        if self.verdict == "fraud":
            return "block"
        if self.verdict == "suspicious":
            return "review"
        return "allow"

    @property
    def is_fraud(self) -> bool:
        """Return whether Signal classified the content as fraud."""
        return self.verdict == "fraud"

    def to_dict(self) -> dict[str, Any]:
        """Return a stable payload suitable for app events or dashboards."""
        return {
            "id": self.id,
            "verdict": self.verdict,
            "score": self.score,
            "recommended_action": self.recommended_action,
            "input_modality": self.input_modality,
            "top_category": self.top_category.to_dict() if self.top_category else None,
            "category_scores": [category.to_dict() for category in self.category_scores],
            "benign_score": self.benign_score,
            "margin_over_second": self.margin_over_second,
            "examples": self.examples,
            "top_matches": self.top_matches,
            "duration_seconds": self.duration_seconds,
            "created_at": self.created_at,
        }


class ResembleSignal:
    """Fraud and scam-intent scoring powered by Resemble Signal.

    Signal complements Resemble Detect: Detect answers "is this media synthetic?", while
    Signal answers "does this content match a fraud/scam pattern?".
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = RESEMBLE_SIGNAL_API_URL,
        http_session: aiohttp.ClientSession | None = None,
        transport: SignalTransport | None = None,
        request_timeout: float = 30.0,
    ) -> None:
        """Create a Signal client.

        Args:
            api_key: Resemble API key. If omitted, ``RESEMBLE_API_KEY`` is read from the
                environment. Not required when ``transport`` is provided. Pass a full
                ``"Bearer ..."`` value to override the default bearer header.
            base_url: Override the Resemble REST Signal API base URL.
            http_session: Existing session for the default REST transport.
            transport: Custom transport, useful for gateways or tests.
            request_timeout: Per-request timeout in seconds.
        """
        if request_timeout <= 0:
            raise ValueError("request_timeout must be > 0")

        if transport is None:
            api_key = api_key or os.environ.get("RESEMBLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "Resemble API key is required, either as argument or set "
                    "RESEMBLE_API_KEY environment variable"
                )
            transport = RestSignalTransport(
                api_key=api_key,
                base_url=base_url,
                http_session=http_session,
            )

        self._transport = transport
        self._request_timeout = request_timeout

    async def score_text(
        self,
        text: str,
        *,
        request_timeout: float | None = None,
    ) -> SignalResult:
        """Score text for fraud/scam intent."""
        text = text.strip()
        if not text:
            raise ValueError("text is required")
        item = await self._transport.score_text(
            text,
            request_timeout=request_timeout or self._request_timeout,
        )
        return _parse_signal_result(item)

    async def score_file(
        self,
        file: bytes,
        *,
        filename: str = "signal-upload",
        media_type: SignalModality | None = None,
        content_type: str | None = None,
        request_timeout: float | None = None,
    ) -> SignalResult:
        """Score an audio, video, or image file for fraud/scam intent."""
        if not file:
            raise ValueError("file is required")
        item = await self._transport.score_file(
            file,
            filename=filename,
            media_type=media_type,
            content_type=content_type,
            request_timeout=request_timeout or self._request_timeout,
        )
        return _parse_signal_result(item)

    async def list_submissions(
        self,
        *,
        page: int = 1,
        per_page: int = 10,
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        """List historical Signal submissions."""
        _validate_page(page, per_page)
        return await self._transport.list_submissions(
            page=page,
            per_page=per_page,
            request_timeout=request_timeout or self._request_timeout,
        )

    async def delete_submission(
        self,
        submission_id: str | int,
        *,
        request_timeout: float | None = None,
    ) -> None:
        """Delete one historical Signal submission."""
        await self._transport.delete_submission(
            submission_id,
            request_timeout=request_timeout or self._request_timeout,
        )

    async def list_custom_categories(
        self,
        *,
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        """List built-in categories, custom categories, and team settings."""
        return await self._transport.list_custom_categories(
            request_timeout=request_timeout or self._request_timeout
        )

    async def create_custom_category(
        self,
        *,
        name: str,
        scenarios: Sequence[str] | str,
        description: str | None = None,
        icon: str | None = None,
        enabled: bool | None = None,
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        """Create a custom category from example fraud/scam scenarios."""
        payload = _category_payload(
            name=name,
            scenarios=scenarios,
            description=description,
            icon=icon,
            enabled=enabled,
        )
        return await self._transport.create_custom_category(
            payload,
            request_timeout=request_timeout or self._request_timeout,
        )

    async def get_custom_category(
        self,
        category_id: str | int,
        *,
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        """Fetch one custom category."""
        return await self._transport.get_custom_category(
            category_id,
            request_timeout=request_timeout or self._request_timeout,
        )

    async def update_custom_category(
        self,
        category_id: str | int,
        *,
        name: str | None = None,
        scenarios: Sequence[str] | str | None = None,
        description: str | None = None,
        icon: str | None = None,
        enabled: bool | None = None,
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        """Update one custom category. Passing scenarios replaces its scenario set."""
        payload = _category_payload(
            name=name,
            scenarios=scenarios,
            description=description,
            icon=icon,
            enabled=enabled,
            allow_partial=True,
        )
        if not payload:
            raise ValueError("at least one custom category field is required")
        return await self._transport.update_custom_category(
            category_id,
            payload,
            request_timeout=request_timeout or self._request_timeout,
        )

    async def delete_custom_category(
        self,
        category_id: str | int,
        *,
        request_timeout: float | None = None,
    ) -> None:
        """Delete one custom category."""
        await self._transport.delete_custom_category(
            category_id,
            request_timeout=request_timeout or self._request_timeout,
        )

    async def update_settings(
        self,
        *,
        use_builtin_categories: bool,
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        """Toggle whether built-in categories are included during scoring."""
        return await self._transport.update_settings(
            {"use_builtin_categories": use_builtin_categories},
            request_timeout=request_timeout or self._request_timeout,
        )

    async def cleanup(self) -> None:
        """Close transport resources when the transport exposes a close method."""
        await self.aclose()

    async def aclose(self) -> None:
        """Close transport resources when the transport exposes a close method."""
        close = getattr(self._transport, "close", None)
        if callable(close):
            maybe_awaitable = close()
            if inspect.isawaitable(maybe_awaitable):
                await maybe_awaitable


class RestSignalTransport:
    """Default transport for Resemble Signal's REST API."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = RESEMBLE_SIGNAL_API_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a REST transport."""
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._session = http_session
        self._owns_session = http_session is None

    async def score_text(self, text: str, *, request_timeout: float) -> dict[str, Any]:
        """Score text and return the Signal item payload."""
        payload = await self._request(
            "POST",
            "/signal",
            json={"text": text},
            request_timeout=request_timeout,
        )
        return _item(payload, "signal")

    async def score_file(
        self,
        file: bytes,
        *,
        filename: str,
        media_type: SignalModality | None,
        content_type: str | None,
        request_timeout: float,
    ) -> dict[str, Any]:
        """Score a media file and return the Signal item payload."""
        form = aiohttp.FormData()
        form.add_field(
            "file",
            file,
            filename=filename,
            content_type=content_type or "application/octet-stream",
        )
        if media_type is not None:
            form.add_field("media_type", media_type)

        payload = await self._request(
            "POST",
            "/signal",
            data=form,
            request_timeout=request_timeout,
        )
        return _item(payload, "signal")

    async def list_submissions(
        self,
        *,
        page: int,
        per_page: int,
        request_timeout: float,
    ) -> dict[str, Any]:
        """List historical Signal submissions."""
        return await self._request(
            "GET",
            "/signal",
            params={"page": page, "per_page": per_page},
            request_timeout=request_timeout,
        )

    async def delete_submission(self, submission_id: str | int, *, request_timeout: float) -> None:
        """Delete one historical Signal submission."""
        await self._request(
            "DELETE",
            f"/signal/{submission_id}",
            request_timeout=request_timeout,
        )

    async def list_custom_categories(self, *, request_timeout: float) -> dict[str, Any]:
        """List built-in and custom Signal categories."""
        return await self._request(
            "GET",
            "/signal/custom_categories",
            request_timeout=request_timeout,
        )

    async def create_custom_category(
        self,
        payload: Mapping[str, Any],
        *,
        request_timeout: float,
    ) -> dict[str, Any]:
        """Create a custom Signal category."""
        response = await self._request(
            "POST",
            "/signal/custom_categories",
            json=dict(payload),
            request_timeout=request_timeout,
        )
        return _item(response, "custom category")

    async def get_custom_category(
        self,
        category_id: str | int,
        *,
        request_timeout: float,
    ) -> dict[str, Any]:
        """Fetch one custom Signal category."""
        response = await self._request(
            "GET",
            f"/signal/custom_categories/{category_id}",
            request_timeout=request_timeout,
        )
        return _item(response, "custom category")

    async def update_custom_category(
        self,
        category_id: str | int,
        payload: Mapping[str, Any],
        *,
        request_timeout: float,
    ) -> dict[str, Any]:
        """Update one custom Signal category."""
        response = await self._request(
            "PATCH",
            f"/signal/custom_categories/{category_id}",
            json=dict(payload),
            request_timeout=request_timeout,
        )
        return _item(response, "custom category")

    async def delete_custom_category(
        self,
        category_id: str | int,
        *,
        request_timeout: float,
    ) -> None:
        """Delete one custom Signal category."""
        await self._request(
            "DELETE",
            f"/signal/custom_categories/{category_id}",
            request_timeout=request_timeout,
        )

    async def update_settings(
        self,
        payload: Mapping[str, Any],
        *,
        request_timeout: float,
    ) -> dict[str, Any]:
        """Update team-level Signal settings."""
        return await self._request(
            "PATCH",
            "/signal/settings",
            json=dict(payload),
            request_timeout=request_timeout,
        )

    async def close(self) -> None:
        """Close the owned aiohttp session, if one was created by this transport."""
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        request_timeout: float,
        json: Mapping[str, Any] | None = None,
        data: aiohttp.FormData | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers = {"Authorization": _authorization_value(self._api_key)}
        if json is not None:
            headers["Content-Type"] = "application/json"

        async with self._ensure_session().request(
            method,
            f"{self._base_url}{path}",
            headers=headers,
            json=json,
            data=data,
            params=params,
            timeout=aiohttp.ClientTimeout(total=request_timeout),
        ) as resp:
            if resp.status < 200 or resp.status >= 300:
                body = await resp.text()
                raise RuntimeError(f"Resemble Signal request failed ({resp.status}): {body[:500]}")
            return await resp.json()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = aiohttp.ClientSession()

        return self._session


def _parse_signal_result(item: Mapping[str, Any]) -> SignalResult:
    verdict = item.get("verdict")
    if verdict not in ("safe", "suspicious", "fraud"):
        raise ValueError(f"Unexpected Resemble Signal verdict: {verdict!r}")

    input_modality = item.get("input_modality")
    if input_modality not in ("text", "audio", "video", "image"):
        raise ValueError(f"Unexpected Resemble Signal input_modality: {input_modality!r}")

    top_category = _parse_category(item.get("top_category"))
    category_scores = [
        category
        for category in (_parse_category(raw) for raw in item.get("category_scores") or [])
        if category is not None
    ]
    examples = [str(example) for example in item.get("examples") or []]
    top_matches = [
        dict(match) for match in item.get("top_matches") or [] if isinstance(match, Mapping)
    ]

    return SignalResult(
        id=item.get("id"),
        verdict=verdict,
        input_modality=input_modality,
        top_category=top_category,
        category_scores=category_scores,
        benign_score=_opt_float(item.get("benign_score")),
        margin_over_second=_opt_float(item.get("margin_over_second")),
        examples=examples,
        top_matches=top_matches,
        duration_seconds=_opt_float(item.get("duration_seconds")),
        created_at=item.get("created_at"),
        raw=dict(item),
    )


def _parse_category(raw: Any) -> SignalCategoryScore | None:
    if not isinstance(raw, Mapping):
        return None
    try:
        name = str(raw["name"])
        score = float(raw["score"])
    except (KeyError, TypeError, ValueError):
        return None
    return SignalCategoryScore(
        name=name,
        score=max(0.0, min(1.0, score)),
        icon=str(raw["icon"]) if raw.get("icon") is not None else None,
        raw=dict(raw),
    )


def _category_payload(
    *,
    name: str | None,
    scenarios: Sequence[str] | str | None,
    description: str | None,
    icon: str | None,
    enabled: bool | None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if name is not None:
        name = name.strip()
        if not name:
            raise ValueError("name is required")
        payload["name"] = name
    elif not allow_partial:
        raise ValueError("name is required")

    if scenarios is not None:
        if isinstance(scenarios, str):
            normalized_scenarios = [line.strip() for line in scenarios.splitlines() if line.strip()]
        else:
            normalized_scenarios = [str(scenario).strip() for scenario in scenarios if scenario]
        if not normalized_scenarios:
            raise ValueError("at least one scenario is required")
        payload["scenarios"] = normalized_scenarios
    elif not allow_partial:
        raise ValueError("at least one scenario is required")

    if description is not None:
        payload["description"] = description
    if icon is not None:
        payload["icon"] = icon
    if enabled is not None:
        payload["enabled"] = enabled
    return payload


def _validate_page(page: int, per_page: int) -> None:
    if page < 1:
        raise ValueError("page must be >= 1")
    if not 1 <= per_page <= 100:
        raise ValueError("per_page must be between 1 and 100")


def _item(payload: Mapping[str, Any], name: str) -> dict[str, Any]:
    item = payload.get("item")
    if not isinstance(item, Mapping):
        raise ValueError(f"Resemble Signal {name} response missing item: {payload}")
    return dict(item)


def _authorization_value(api_key: str) -> str:
    return api_key if api_key.lower().startswith("bearer ") else f"Bearer {api_key}"


def _opt_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
