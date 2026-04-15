#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Settings infrastructure for Pipecat AI services.

Each service type has a settings dataclass (``LLMSettings``, ``TTSSettings``,
``STTSettings``, or a service-specific subclass).  The same class is used in
two distinct modes:

**Store mode** — the service's ``self._settings`` object that holds the full
current state.  Every field must have a real value; ``NOT_GIVEN`` is never
valid here.  Services that don't support an inherited field should set it to
``None``.  ``validate_complete()`` (called automatically in
``AIService.start()``) enforces this invariant.

**Delta mode** — a sparse update object carried by an
``*UpdateSettingsFrame``.  Only the fields the caller wants to change are set;
all others remain at their default of ``NOT_GIVEN``.  ``apply_update()``
merges a delta into a store, skipping any ``NOT_GIVEN`` fields.

Key helpers:

- ``NOT_GIVEN`` / ``is_given()`` — sentinel and check for "field not provided
  in this delta".
- ``apply_update(delta)`` — merge a delta into a store, returning changed
  fields.
- ``from_mapping(dict)`` — build a delta from a plain dict (for backward
  compatibility with dict-based ``*UpdateSettingsFrame``).
- ``validate_complete()`` — assert that a store has no ``NOT_GIVEN`` fields.
- ``extra`` dict — overflow for service-specific keys that don't map to a
  declared field.
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
    """Sentinel meaning "this field was not included in the delta".

    ``NOT_GIVEN`` is distinct from ``None`` (which is a valid stored value,
    typically meaning "this service doesn't support this field").  Every
    settings field defaults to ``NOT_GIVEN`` so that delta-mode objects are
    sparse by default and ``apply_update`` can skip untouched fields.

    ``NOT_GIVEN`` must never appear in a store-mode object — see
    ``validate_complete()``.
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
"""Singleton sentinel meaning "this field was not included in the delta".

Valid only in delta-mode settings objects.  Must never appear in a service's
``self._settings`` (store mode) — use ``None`` instead for unsupported fields.
"""


def is_given(value: Any) -> bool:
    """Check whether a delta field was explicitly provided.

    Typically used when processing a delta to decide whether a field
    should be applied::

        if is_given(delta.voice):
            # caller wants to change the voice
            ...

    For store-mode objects this always returns ``True`` (since
    ``validate_complete`` ensures no ``NOT_GIVEN`` fields remain).

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
    """Base class for runtime-updatable service settings.

    These settings capture the subset of a service's configuration that can
    be changed **while the pipeline is running** (e.g. switching the model or
    changing the voice).  They are *not* meant to capture every constructor
    parameter — only those that support live updates via
    ``*UpdateSettingsFrame``.

    Every AI service type (LLM, TTS, STT) extends this with its own fields.
    Each instance operates in one of two modes (see module docstring):

    - **Store mode** (``self._settings``): holds the full current state.
      Every field must be a real value — ``NOT_GIVEN`` is never valid.
      Use ``None`` for inherited fields the service doesn't support.
      Enforced at runtime by ``validate_complete()``.
    - **Delta mode** (``*UpdateSettingsFrame``): a sparse update.
      Only fields the caller wants to change are set; all others stay at
      the default ``NOT_GIVEN`` and are skipped by ``apply_update()``.

    Parameters:
        model: The model identifier used by the service.  Set to ``None``
            in store mode if the service has no model concept.
        extra: Overflow dict for service-specific keys that don't map to a
            declared field.
    """

    # -- common fields -------------------------------------------------------

    model: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    """AI model identifier (e.g. ``"gpt-4o"``, ``"eleven_turbo_v2_5"``).

    Defaults to ``NOT_GIVEN`` for delta mode.  In store mode, set to a
    model string or ``None`` if the service has no model concept.
    """

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
        """Return a dict of only the fields that are not ``NOT_GIVEN``.

        Primarily useful for delta-mode objects to inspect which fields were
        set.  For a store-mode object this returns all declared fields (since
        none should be ``NOT_GIVEN``).

        Skips the ``extra`` field itself but merges its entries into the
        returned dict at the top level.

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

    def apply_update(self: _S, delta: _S) -> Dict[str, Any]:
        """Merge a delta-mode object into this store-mode object.

        Only fields in *delta* that are **given** (i.e. not ``NOT_GIVEN``)
        are considered.  A field is "changed" if its new value differs from
        the current value.

        The ``extra`` dicts are merged: keys present in the delta overwrite
        keys in the target.

        Args:
            delta: A delta-mode settings object of the same type.

        Returns:
            A dict mapping each changed field name to its **pre-update** value.
            Use ``changed.keys()`` for the set of names, or index with
            ``changed["field"]`` to inspect the old value.

        Examples::

            # store-mode object (all fields given)
            current = TTSSettings(voice="alice", language="en")
            # delta-mode object (only voice is set)
            delta = TTSSettings(voice="bob")
            changed = current.apply_update(delta)
            # changed == {"voice": "alice"}
            # current.voice == "bob", current.language == "en"
        """
        changed: Dict[str, Any] = {}
        for f in fields(self):
            if f.name == "extra":
                continue
            new_val = getattr(delta, f.name, NOT_GIVEN)
            if not is_given(new_val):
                continue
            old_val = getattr(self, f.name)
            if old_val != new_val:
                setattr(self, f.name, new_val)
                changed[f.name] = old_val

        # Merge extra
        for key, new_val in delta.extra.items():
            old_val = self.extra.get(key, NOT_GIVEN)
            if old_val != new_val:
                self.extra[key] = new_val
                changed[key] = old_val

        return changed

    @classmethod
    def from_mapping(cls: Type[_S], settings: Mapping[str, Any]) -> _S:
        """Build a **delta-mode** settings object from a plain dictionary.

        This exists for backward compatibility with code that passes plain
        dicts via ``*UpdateSettingsFrame(settings={...})``.  The returned
        object is a delta: only the keys present in *settings* are set;
        all other fields remain ``NOT_GIVEN``.

        Keys are matched to dataclass fields by name.  Keys listed in
        ``_aliases`` are translated to their canonical name first.  Any
        remaining unrecognized keys are placed into ``extra``.

        Args:
            settings: A dictionary of setting names to values.

        Returns:
            A new delta-mode settings instance.

        Examples::

            delta = TTSSettings.from_mapping({"voice_id": "alice", "speed": 1.2})
            # delta.voice == "alice"  (via alias)
            # delta.language is NOT_GIVEN  (not in the dict)
            # delta.extra == {"speed": 1.2}
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

    def validate_complete(self) -> None:
        """Check that this is a valid store-mode object (no ``NOT_GIVEN`` fields).

        Called automatically by ``AIService.start()`` to catch fields that a
        service forgot to initialize in its ``__init__``.  Can also be called
        manually after constructing a store-mode settings object.

        Logs a warning for each uninitialized field.  Failure to initialize
        all fields may or may not cause runtime issues — it depends on
        whether and how the service actually reads the field — but it indicates
        a deviation from expectations and should be fixed.
        """
        missing = [
            f.name
            for f in fields(self)
            if f.name != "extra" and isinstance(getattr(self, f.name), _NotGiven)
        ]
        if missing:
            names = ", ".join(missing)
            logger.error(
                f"{type(self).__name__}: the following fields are NOT_GIVEN: {names}. "
                f"All settings fields should be initialized in the service's "
                f"__init__ (use None for unsupported fields)."
            )

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
class ImageGenSettings(ServiceSettings):
    """Runtime-updatable settings for image generation services.

    Used in both store and delta mode — see ``ServiceSettings``.

    Parameters:
        model: Image generation model identifier.
    """


@dataclass
class VisionSettings(ServiceSettings):
    """Runtime-updatable settings for vision services.

    Used in both store and delta mode — see ``ServiceSettings``.

    Parameters:
        model: Vision model identifier.
    """


@dataclass
class LLMSettings(ServiceSettings):
    """Runtime-updatable settings for LLM services.

    Used in both store and delta mode — see ``ServiceSettings``.

    These fields are common across LLM providers.  Not every provider supports
    every field; in store mode, set unsupported fields to ``None`` (e.g. a
    service that doesn't support ``seed`` should initialize it as
    ``seed=None``).

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

    temperature: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    max_tokens: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    top_p: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    top_k: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    frequency_penalty: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    presence_penalty: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    seed: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    filter_incomplete_user_turns: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    user_turn_completion_config: UserTurnCompletionConfig | None | _NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )


@dataclass
class TTSSettings(ServiceSettings):
    """Runtime-updatable settings for TTS services.

    Used in both store and delta mode — see ``ServiceSettings``.

    In store mode, set unsupported fields to ``None`` (e.g. ``language=None``
    if the service doesn't expose a language setting).

    Parameters:
        model: TTS model identifier.
        voice: Voice identifier or name.
        language: Language for speech synthesis.  The union type reflects the
            *input* side: callers may pass a ``Language`` enum or a raw string
            in a delta.  However, the **stored** value (in store mode) is
            always a service-specific string or ``None`` —
            ``TTSService._update_settings`` converts ``Language`` enums via
            ``language_to_service_language()`` before writing, and
            ``__init__`` methods do the same at construction time.
    """

    voice: str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    language: Language | str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)

    _aliases: ClassVar[Dict[str, str]] = {"voice_id": "voice"}


@dataclass
class STTSettings(ServiceSettings):
    """Runtime-updatable settings for STT services.

    Used in both store and delta mode — see ``ServiceSettings``.

    In store mode, set unsupported fields to ``None`` (e.g. ``language=None``
    if the service auto-detects language).

    Parameters:
        model: STT model identifier.
        language: Language for speech recognition.  The union type reflects the
            *input* side: callers may pass a ``Language`` enum or a raw string
            in a delta.  However, the **stored** value (in store mode) is
            always a service-specific string or ``None`` —
            ``STTService._update_settings`` converts ``Language`` enums via
            ``language_to_service_language()`` before writing, and
            ``__init__`` methods do the same at construction time.
    """

    language: Language | str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
