#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Configuration models for the Blynt speech-to-text service."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class STTLanguages(StrEnum):
    """Supported languages for Blynt STT."""

    FR = "fr"
    EN = "en"
    PT = "pt"
    DE = "de"
    ES = "es"
    IT = "it"


@dataclass
class Fact:
    """A named fact that biases recognition for the whole session.

    Parameters:
        name: Name of the fact.
        value: Value of the fact.
        description: Optional description of the fact.
    """

    name: str
    value: str
    description: str | None = None


@dataclass
class DeclaredValues:
    """A list of expected values (hint) that biases recognition.

    Parameters:
        values: Values the speaker is likely to say.
        name: Optional name of the hint.
        description: Optional description of the hint.
        type: Hint type sent to the API.
    """

    values: list[str]
    name: str | None = None
    description: str | None = None
    type: str = "values"


def _omit_none(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


@dataclass
class BlyntSessionContext:
    """Session-level contextual biasing sent once on ``start_session``.

    Facts and hints apply to every turn in the session and combine with any
    per-turn ``turnContext``.
    """

    facts: list[Fact] = field(default_factory=list)
    hints: list[DeclaredValues] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any] | None:
        """Build the ``sessionContext`` payload for ``start_session``.

        Returns:
            The payload, or None when there are no facts and no hints.
        """
        if not self.facts and not self.hints:
            return None
        return {
            "facts": [_omit_none(asdict(fact)) for fact in self.facts],
            "hints": [_omit_none(asdict(hint)) for hint in self.hints],
        }


@dataclass
class BlyntSTTOptions:
    """Configuration options for Blynt STT.

    The realtime API is deployment-scoped and authenticated. A session connects to
    ``{base_url}/api/v1/deployments/{deployment_id}/ws`` with a Bearer token.

    Attributes:
        deployment_id: Blynt deployment id to run the session against. Falls back
            to the ``BLYNT_DEPLOYMENT_ID`` environment variable when not provided.
        api_key: Blynt API key for Bearer auth. Falls back to the ``BLYNT_API_KEY``
            environment variable when not provided.
        base_url: Base URL of the Blynt API (default: "wss://api.blynt.ai").
        language: Language code (use STTLanguages enum or string).
        session_context: Optional session-level facts and hints for biasing.

    Example:
        >>> options = BlyntSTTOptions(
        ...     deployment_id="699db8281984ab5b86867207",  # or set BLYNT_DEPLOYMENT_ID
        ...     api_key="sk-...",  # or set BLYNT_API_KEY
        ...     language=STTLanguages.FR,
        ...     session_context=BlyntSessionContext(
        ...         facts=[Fact(name="domaine", value="immatriculation")],
        ...         hints=[DeclaredValues(values=["plaque", "AA-123-BB"])],
        ...     ),
        ... )
    """

    deployment_id: str | None = None
    api_key: str | None = field(default=None, repr=False)
    base_url: str = "wss://api.blynt.ai"
    language: STTLanguages | str = STTLanguages.FR
    session_context: BlyntSessionContext | None = None

    def __post_init__(self) -> None:
        if self.deployment_id is None:
            self.deployment_id = os.environ.get("BLYNT_DEPLOYMENT_ID")
        if not self.deployment_id:
            raise ValueError(
                "A Blynt deployment id is required. Pass `deployment_id` or set "
                "the BLYNT_DEPLOYMENT_ID environment variable."
            )
        if self.api_key is None:
            self.api_key = os.environ.get("BLYNT_API_KEY")
        if not self.api_key:
            raise ValueError(
                "A Blynt API key is required. Pass `api_key` or set the "
                "BLYNT_API_KEY environment variable."
            )
        if isinstance(self.language, str) and self.language not in {
            lang.value for lang in STTLanguages
        }:
            raise ValueError(
                "When passed as a string, language must be one of "
                f"{', '.join(lang.value for lang in STTLanguages)}. "
                f"Got language={self.language!r}."
            )

    def get_ws_url(self) -> str:
        """Build the deployment WebSocket URL."""
        base = self.base_url.rstrip("/")
        if base.startswith("http://"):
            base = base.replace("http://", "ws://", 1)
        elif base.startswith("https://"):
            base = base.replace("https://", "wss://", 1)
        elif not base.startswith(("ws://", "wss://")):
            base = f"wss://{base}"

        return f"{base}/api/v1/deployments/{self.deployment_id}/ws"

    def get_headers(self) -> dict[str, str]:
        """Build the Bearer authentication headers."""
        return {"Authorization": f"Bearer {self.api_key}"}

    @property
    def language_code(self) -> str:
        """Language code sent to the API."""
        if isinstance(self.language, STTLanguages):
            return self.language.value
        return str(self.language)
