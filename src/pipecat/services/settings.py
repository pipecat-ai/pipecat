#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Settings infrastructure for Pipecat AI services.

This module provides dataclass-based settings objects for service configuration.
Each service type has a corresponding settings class (e.g. ``TTSSettings``,
``LLMSettings``) whose fields use the ``NOT_GIVEN`` sentinel to distinguish
"leave unchanged" from an explicit ``None``.

Key concepts:

- **NOT_GIVEN sentinel**: A value meaning "this field was not provided in the
  update". Distinct from ``None`` (which may be a valid value for a setting).
- **Settings as both state and delta**: The same class is used for the
  service's current settings *and* for update objects.  Fields set to
  ``NOT_GIVEN`` are simply skipped when applying an update.
- **apply_update**: Applies a delta onto a target settings object and returns
  a dict mapping each changed field name to its previous value.
- **from_mapping**: Constructs a settings object from a plain dict,
  supporting field aliases (e.g. ``"voice_id"`` → ``"voice"``).
- **Extras**: Unknown keys land in the ``extra`` dict so services that have
  non-standard settings don't lose data.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Mapping, Optional, Type, TypeVar

from loguru import logger

from pipecat.transcriptions.language import Language

if TYPE_CHECKING:
    from pipecat.turns.user_turn_completion_mixin import UserTurnCompletionConfig

# ---------------------------------------------------------------------------
# NOT_GIVEN sentinel
# ---------------------------------------------------------------------------


class _NotGiven:
    """Sentinel indicating a settings field was not provided.

    ``NOT_GIVEN`` means "the caller did not supply this value" — distinct from
    ``None``, which may be a legitimate setting value.  It is used as the
    default for every settings field so that ``apply_update`` can tell which
    fields the caller actually wants to change.
    """

    _instance: Optional[_NotGiven] = None

    def __new__(cls) -> _NotGiven:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "NOT_GIVEN"

    def __bool__(self) -> bool:
        return False


NOT_GIVEN: _NotGiven = _NotGiven()
"""Singleton sentinel meaning "this field was not included in the update"."""


def is_given(value: Any) -> bool:
    """Check whether a value was explicitly provided (i.e. is not ``NOT_GIVEN``).

    Args:
        value: The value to check.

    Returns:
        ``True`` if *value* is anything other than ``NOT_GIVEN``.
    """
    return not isinstance(value, _NotGiven)


# ---------------------------------------------------------------------------
# Base ServiceSettings
# ---------------------------------------------------------------------------

_S = TypeVar("_S", bound="ServiceSettings")


@dataclass
class ServiceSettings:
    """Base class for service settings.

    Every AI service type (LLM, TTS, STT) extends this with its own fields.
    Fields default to ``NOT_GIVEN`` so that an instance can represent either
    the full current state **or** a sparse update delta.

    Parameters:
        model: The model identifier used by the service.
        extra: Overflow dict for service-specific keys that don't map to a
            declared field.
    """

    # -- common fields -------------------------------------------------------

    model: str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    """AI model identifier (e.g. ``"gpt-4o"``, ``"eleven_turbo_v2_5"``)."""

    extra: Dict[str, Any] = field(default_factory=dict)
    """Catch-all for service-specific keys that have no declared field."""

    # -- class-level configuration -------------------------------------------

    _aliases: ClassVar[Dict[str, str]] = {}
    """Map of alternative key names to canonical field names.

    For example ``{"voice_id": "voice"}`` lets callers use either spelling.
    Subclasses should override this as needed.
    """

    # -- public API ----------------------------------------------------------

    def given_fields(self) -> Dict[str, Any]:
        """Return a dict of only the fields that were explicitly provided.

        Skips ``NOT_GIVEN`` values and the ``extra`` field itself.  Entries
        from ``extra`` are included at the top level.

        Returns:
            Dictionary mapping field names to their provided values.
        """
        result: Dict[str, Any] = {}
        for f in fields(self):
            if f.name == "extra":
                continue
            val = getattr(self, f.name)
            if is_given(val):
                result[f.name] = val
        result.update(self.extra)
        return result

    def apply_update(self: _S, update: _S) -> Dict[str, Any]:
        """Apply *update* onto this settings object, returning changed fields.

        Only fields in *update* that are **given** (i.e. not ``NOT_GIVEN``)
        are considered.  A field is "changed" if its new value differs from
        the current value.

        The ``extra`` dicts are merged: keys present in the update overwrite
        keys in the target.

        Args:
            update: A settings object of the same type containing the delta.

        Returns:
            A dict mapping each changed field name to its **pre-update** value.
            Use ``changed.keys()`` for the set of names, or index with
            ``changed["field"]`` to inspect the old value.

        Examples::

            current = TTSSettings(voice="alice", language="en")
            delta = TTSSettings(voice="bob")
            changed = current.apply_update(delta)
            # changed == {"voice": "alice"}
            # current.voice == "bob", current.language == "en"
        """
        changed: Dict[str, Any] = {}
        for f in fields(self):
            if f.name == "extra":
                continue
            new_val = getattr(update, f.name)
            if not is_given(new_val):
                continue
            old_val = getattr(self, f.name)
            if old_val != new_val:
                setattr(self, f.name, new_val)
                changed[f.name] = old_val

        # Merge extra
        for key, new_val in update.extra.items():
            old_val = self.extra.get(key, NOT_GIVEN)
            if old_val != new_val:
                self.extra[key] = new_val
                changed[key] = old_val

        return changed

    @classmethod
    def from_mapping(cls: Type[_S], settings: Mapping[str, Any]) -> _S:
        """Construct a settings object from a plain dictionary.

        This exists for backward compatibility with code that passes plain
        dicts via ``*UpdateSettingsFrame(settings={...})``.

        Keys are matched to dataclass fields by name.  Keys listed in
        ``_aliases`` are translated to their canonical name first.  Any
        remaining unrecognized keys are placed into ``extra``.

        Args:
            settings: A dictionary of setting names to values.

        Returns:
            A new settings instance with the corresponding fields populated.

        Examples::

            update = TTSSettings.from_mapping({"voice_id": "alice", "speed": 1.2})
            # update.voice == "alice"  (via alias)
            # update.extra == {"speed": 1.2}
        """
        field_names = {f.name for f in fields(cls)} - {"extra"}
        kwargs: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}

        for key, value in settings.items():
            # Resolve aliases first
            canonical = cls._aliases.get(key, key)
            if canonical in field_names:
                kwargs[canonical] = value
            else:
                extra[key] = value

        instance = cls(**kwargs)
        instance.extra = extra
        return instance

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a flat dictionary, including extra.

        Only given (non-``NOT_GIVEN``) values are included.  This is the
        inverse of ``from_mapping`` and useful for passing settings to APIs
        that expect plain dicts.

        Returns:
            A flat dictionary of all given settings.
        """
        return self.given_fields()

    def copy(self: _S) -> _S:
        """Return a deep copy of this settings instance.

        Returns:
            A new settings object with the same field values.
        """
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# Service-specific settings
# ---------------------------------------------------------------------------


@dataclass
class LLMSettings(ServiceSettings):
    """Settings for LLM services.

    Parameters:
        model: LLM model identifier.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        top_p: Nucleus sampling probability.
        top_k: Top-k sampling parameter.
        frequency_penalty: Frequency penalty.
        presence_penalty: Presence penalty.
        seed: Random seed for reproducibility.
        filter_incomplete_user_turns: Enable LLM-based turn completion detection
            to suppress bot responses when the user was cut off mid-thought.
            See ``examples/foundational/22-filter-incomplete-turns.py`` and
            ``UserTurnCompletionLLMServiceMixin``.
        user_turn_completion_config: Configuration for turn completion behavior
            when ``filter_incomplete_user_turns`` is enabled. Controls timeouts
            and prompts for incomplete turns.
    """

    temperature: float | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    max_tokens: int | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    top_p: float | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    top_k: int | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    frequency_penalty: float | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    presence_penalty: float | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    seed: int | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    filter_incomplete_user_turns: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    user_turn_completion_config: UserTurnCompletionConfig | _NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )


@dataclass
class TTSSettings(ServiceSettings):
    """Settings for TTS services.

    Parameters:
        model: TTS model identifier.
        voice: Voice identifier or name.
        language: Language for speech synthesis. Accepts a ``Language`` enum
            (converted to a service-specific string) or a raw string (stored
            as-is).
    """

    voice: str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    language: Language | str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)

    _aliases: ClassVar[Dict[str, str]] = {"voice_id": "voice"}


@dataclass
class STTSettings(ServiceSettings):
    """Settings for STT services.

    Parameters:
        model: STT model identifier.
        language: Language for speech recognition. Accepts a ``Language`` enum
            (converted to a service-specific string) or a raw string (stored
            as-is).
    """

    language: Language | str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
