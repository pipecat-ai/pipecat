# Copyright 2026 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom pronunciations for the Rime TTS plugin.

Rime's Mist-family models (``mistv2``, ``mistv3``) parse ``{phoneme}`` blocks
inline in the synthesis text: whatever is inside the curly braces is pronounced
verbatim using Rime's phoneme alphabet, bypassing the grapheme-to-phoneme
lexicon for that span.

This module provides a ``CustomPronunciations`` object that owns a mapping of
``{input_text: phonemes}`` pairs, optionally kept in sync with the Rime
customer API via a background refresh loop. Any number of ``rime.TTS``
instances can reference the same object; each call to ``tts.synthesize(...)``
consults ``prons.rewrite(text)`` by reference, so a refresh mid-session is
picked up by the next utterance with no TTS reconfiguration.

Keys are stored exactly as the API delivers them (no case folding, no
whitespace stripping). Matching rules are derived from key content:

- Keys containing any uppercase character are matched **case-sensitively**
  (these are literals; canonical forms are lowercased by the server).
- All-lowercase keys are matched **case-insensitively** (could be canonicals
  or lowercase literals — case-insensitive is the safe superset).

Matching uses ``\\w`` lookarounds (``(?<!\\w)...(?!\\w)``) rather than ``\\b``,
so keys with trailing punctuation (``"Dr."``) and keys with internal spaces
(``"Lisinopril 10mg"``) both match cleanly.

``{phoneme}`` bracket syntax is Mist-only. Attaching a non-empty pronunciations
object to a ``rime.TTS(model="arcana")`` logs a warning once per TTS instance
and suppresses rewriting for that TTS — Arcana would synthesize the brace
characters literally in the rendered audio, so the plugin sends unrewritten
text rather than producing broken output.
"""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Mapping
from typing import Any, Callable

import aiohttp
from loguru import logger

_DEFAULT_CUSTOMER_API_BASE_URL = "https://users.rime.ai"
_MIN_REFRESH_INTERVAL = 30.0
_INITIAL_FETCH_RETRIES = 2
_INITIAL_FETCH_BACKOFFS = (1.0, 3.0)


# ---------------------------------------------------------------------------
# Response projection
# ---------------------------------------------------------------------------


def _project_response(body: Mapping[str, Any]) -> dict[str, str]:
    """Project a ``GET /speech-qa/custom-pronunciations`` response into a flat map.

    Expected response shape (API-010)::

        {"default": {"<input_text_or_canonical>": "<pronunciation>", ...},
         "vocabs": {}}

    Keys and values must both be non-empty strings. ``vocabs`` is read and
    discarded. Keys are not normalized — they are stored exactly as delivered.
    """
    out: dict[str, str] = {}
    default = body.get("default")
    if isinstance(default, Mapping):
        for k, v in default.items():
            if isinstance(k, str) and isinstance(v, str) and k and v:
                out[k] = v
    return out


# ---------------------------------------------------------------------------
# Regex compilation
# ---------------------------------------------------------------------------


def _compile_patterns(
    entries: Mapping[str, str],
) -> tuple[re.Pattern[str] | None, re.Pattern[str] | None]:
    """Build two alternation regexes over ``entries.keys()``.

    Returns ``(case_sensitive_pattern, case_insensitive_pattern)``. Keys
    containing any uppercase character go into the case-sensitive pattern;
    all-lowercase keys go into the case-insensitive pattern. Within each
    pattern, keys are sorted descending by length so longer literals match
    before shorter ones that would shadow them (``"Lisinopril 10mg"`` before
    ``"Lisinopril"``). Bounds use ``\\w`` lookarounds, not ``\\b``.
    """
    cased_keys: list[str] = []
    lower_keys: list[str] = []
    for k in entries:
        (cased_keys if any(c.isupper() for c in k) else lower_keys).append(k)

    def _build(keys: list[str], flags: int) -> re.Pattern[str] | None:
        if not keys:
            return None
        keys_sorted = sorted(keys, key=len, reverse=True)
        escaped = [re.escape(k) for k in keys_sorted]
        return re.compile(r"(?<!\w)(" + "|".join(escaped) + r")(?!\w)", flags)

    return _build(cased_keys, 0), _build(lower_keys, re.IGNORECASE)


# ---------------------------------------------------------------------------
# CustomPronunciations
# ---------------------------------------------------------------------------


class CustomPronunciations:
    """A live mapping of ``{input_text: phonemes}`` used at synthesis time.

    Instances are typically produced by :func:`load_custom_pronunciations`,
    which performs the initial fetch and optionally runs a background refresh
    loop. The object is safe to share across many ``rime.TTS`` instances:
    each synth consults ``rewrite(text)`` by reference, so refreshes are
    picked up by the next utterance without any TTS reconfiguration.

    The object must be closed when no longer needed (``await prons.aclose()``
    or ``async with prons: ...``). When used inside a LiveKit ``JobContext``,
    the recommended pattern is ``ctx.add_shutdown_callback(prons.aclose)``.
    """

    def __init__(self) -> None:
        # Public attribute initialized to empty; populated by fetch or by the
        # internal factory used in :func:`load_custom_pronunciations`.
        self._entries: dict[str, str] = {}
        self._cs_pattern: re.Pattern[str] | None = None
        self._ci_pattern: re.Pattern[str] | None = None
        self._refresh_lock = asyncio.Lock()
        self._in_flight: asyncio.Task[None] | None = None
        self._refresh_task: asyncio.Task[None] | None = None
        self._closed = False

        # Fetch configuration, set when load_custom_pronunciations builds the
        # object. Direct construction leaves these as None, which is fine:
        # rewrite() still works over the empty mapping, and refresh() raises
        # cleanly if invoked without configuration.
        self._api_key: str | None = None
        self._base_url: str = _DEFAULT_CUSTOMER_API_BASE_URL
        self._http_session: aiohttp.ClientSession | None = None
        self._refresh_interval: float | None = None

    # ------------------------------------------------------------------
    # Public read surface
    # ------------------------------------------------------------------

    @property
    def entries(self) -> Mapping[str, str]:
        """Read-only snapshot of the current mapping."""
        return dict(self._entries)

    def rewrite(self, text: str) -> str:
        """Wrap every known literal in ``{phonemes}``.

        Pure, synchronous, no IO. Case-sensitive for keys containing any
        uppercase character; case-insensitive for all-lowercase keys.
        """
        cs = self._cs_pattern
        ci = self._ci_pattern
        if cs is None and ci is None:
            return text
        entries = self._entries

        def _replace(match: re.Match[str]) -> str:
            key = match.group(0)
            # Case-sensitive hits: exact lookup.
            if key in entries:
                return "{" + entries[key] + "}"
            # Case-insensitive hits: lookup by lowered form.
            lowered = key.lower()
            if lowered in entries:
                return "{" + entries[lowered] + "}"
            # Shouldn't happen (both patterns are built from `entries`), but
            # if it does, leave the span untouched.
            return key

        if cs is not None:
            text = cs.sub(_replace, text)
        if ci is not None:
            text = ci.sub(_replace, text)
        return text

    def __call__(self, text: str) -> str:
        """Alias for :meth:`rewrite`, so instances satisfy ``Callable[[str], str]``."""
        return self.rewrite(text)

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    async def refresh(self) -> None:
        """Force a fetch now. Coalesces with any in-flight refresh.

        If another ``refresh()`` is already in flight, awaits that one
        instead of issuing a second request. Safe to call concurrently from
        many tasks.
        """
        if self._closed:
            return
        task = self._in_flight
        if task is not None and not task.done():
            await asyncio.shield(task)
            return
        task = asyncio.create_task(self._fetch_and_swap(), name="rime-prons-refresh-fetch")
        self._in_flight = task
        try:
            await task
        finally:
            # Always clear so a future refresh starts a new fetch. If the
            # task raised, the exception is consumed by the `await` above;
            # callers waiting on `asyncio.shield(task)` also see it, but
            # the task's result is consumed by every awaiter, so no
            # unretrieved-exception warning.
            self._in_flight = None

    async def _fetch_and_swap(self) -> None:
        body = await _http_get_pronunciations(
            api_key=self._api_key,
            base_url=self._base_url,
            http_session=self._http_session,
        )
        new_entries = _project_response(body)
        async with self._refresh_lock:
            cs, ci = _compile_patterns(new_entries)
            # Atomic swap. Reads don't take the lock; Python reference
            # assignment is atomic under the GIL, so a concurrent rewrite()
            # sees either old or new, never a torn state.
            self._entries = new_entries
            self._cs_pattern = cs
            self._ci_pattern = ci

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def aclose(self) -> None:
        """Cancel the refresh task. Idempotent."""
        if self._closed:
            return
        self._closed = True
        task = self._refresh_task
        self._refresh_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    async def __aenter__(self) -> CustomPronunciations:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    def __del__(self) -> None:
        # Best-effort warning for leaked refresh loops. We can't await here,
        # so we can't cancel the task cleanly; just make the mistake noisy.
        try:
            if not self._closed and self._refresh_task is not None:
                logger.warning(
                    "CustomPronunciations was garbage-collected without aclose(); "
                    "the refresh loop may leak until event loop shutdown. "
                    "Call `await prons.aclose()` or use `async with prons:`."
                )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


async def _http_get_pronunciations(
    *,
    api_key: str | None,
    base_url: str,
    http_session: aiohttp.ClientSession | None,
) -> Mapping[str, Any]:
    """One HTTP GET against ``{base_url}/speech-qa/custom-pronunciations``.

    Raises on HTTP errors and connection errors. Callers decide whether to
    retry or swallow.
    """
    if not api_key:
        raise ValueError(
            "Rime API key is required, either as argument or set RIME_API_KEY "
            "environmental variable"
        )
    url = f"{base_url.rstrip('/')}/speech-qa/custom-pronunciations"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    if http_session is not None:
        async with http_session.get(url, headers=headers) as resp:
            resp.raise_for_status()
            return await resp.json()
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as resp:
            resp.raise_for_status()
            return await resp.json()


def _is_transient_error(exc: BaseException) -> bool:
    """Return True for errors worth retrying at initial-fetch time.

    Transient: 5xx responses, timeouts, connection resets. Permanent: 4xx
    responses (auth, bad request) — those mean config bugs that should
    surface immediately at app startup.
    """
    if isinstance(exc, asyncio.TimeoutError):
        return True
    if isinstance(exc, aiohttp.ClientResponseError):
        return exc.status >= 500
    if isinstance(exc, aiohttp.ClientConnectionError):
        return True
    return False


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


async def load_custom_pronunciations(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    refresh_interval: float | None = 300.0,
    http_session: aiohttp.ClientSession | None = None,
) -> CustomPronunciations:
    """Fetch custom pronunciations and return a ready-to-use object.

    Performs the initial fetch before returning. Transient errors (5xx,
    timeouts, connection resets) are retried twice with 1s/3s backoff;
    permanent errors (4xx, including auth failures) raise immediately.

    If ``refresh_interval`` is set (default 300s, clamped to a minimum of
    30s), a background task re-fetches on that interval. Refresh-loop
    failures are logged and swallowed — the previously loaded mapping stays
    in effect. Pass ``refresh_interval=None`` for a one-shot snapshot.

    The returned object must be closed when no longer needed::

        async def entrypoint(ctx: JobContext):
            prons = await rime.load_custom_pronunciations(refresh_interval=300)
            ctx.add_shutdown_callback(prons.aclose)
            tts = rime.TTS(model="mistv2", custom_pronunciations=prons)
    """
    resolved_key = api_key or os.environ.get("RIME_API_KEY")
    if not resolved_key:
        raise ValueError(
            "Rime API key is required, either as argument or set RIME_API_KEY "
            "environmental variable"
        )

    resolved_base = (base_url or _DEFAULT_CUSTOMER_API_BASE_URL).rstrip("/")

    resolved_interval: float | None
    if refresh_interval is None:
        resolved_interval = None
    else:
        if refresh_interval < _MIN_REFRESH_INTERVAL:
            logger.warning(
                "refresh_interval=%s is below the minimum of %ss; clamping.",
                refresh_interval,
                _MIN_REFRESH_INTERVAL,
            )
            resolved_interval = _MIN_REFRESH_INTERVAL
        else:
            resolved_interval = float(refresh_interval)

    prons = CustomPronunciations()
    prons._api_key = resolved_key
    prons._base_url = resolved_base
    prons._http_session = http_session
    prons._refresh_interval = resolved_interval

    # Initial fetch with limited retry on transient errors.
    last_exc: BaseException | None = None
    attempts = _INITIAL_FETCH_RETRIES + 1
    for attempt in range(attempts):
        try:
            await prons._fetch_and_swap()
            last_exc = None
            break
        except BaseException as e:
            last_exc = e
            if attempt == attempts - 1 or not _is_transient_error(e):
                raise
            backoff = _INITIAL_FETCH_BACKOFFS[min(attempt, len(_INITIAL_FETCH_BACKOFFS) - 1)]
            logger.warning(
                "initial custom_pronunciations fetch failed (%s); "
                "retrying in %ss (attempt %d/%d)",
                type(e).__name__,
                backoff,
                attempt + 1,
                attempts,
            )
            await asyncio.sleep(backoff)
    if last_exc is not None:
        raise last_exc

    # Start the refresh loop if requested.
    if resolved_interval is not None:
        prons._refresh_task = asyncio.create_task(
            _refresh_loop(prons, resolved_interval),
            name="rime-custom-pronunciations-refresh",
        )

    return prons


async def _refresh_loop(prons: CustomPronunciations, interval: float) -> None:
    """Background refresh task. Logs and swallows all errors but cancellation."""
    while True:
        try:
            await asyncio.sleep(interval)
            await prons.refresh()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "custom_pronunciations refresh failed; "
                "keeping previously loaded mapping in effect"
            )


# ---------------------------------------------------------------------------
# Text-rewriter protocol
# ---------------------------------------------------------------------------

# A text-rewriter is any callable that takes a string and returns a string.
# Both `CustomPronunciations` and user-supplied lambdas satisfy this, so the
# TTS constructor accepts either shape in the `custom_pronunciations=` slot.
TextRewriter = Callable[[str], str]


def _rewriter_is_nonempty(rewriter: TextRewriter | None) -> bool:
    """Cheap check for "would this rewriter produce any substitutions?"

    For a ``CustomPronunciations`` instance we can inspect its mapping; for
    arbitrary callables we conservatively answer True (we can't introspect a
    user function). Used to decide whether Arcana's warn-and-drop applies.
    """
    if rewriter is None:
        return False
    if isinstance(rewriter, CustomPronunciations):
        return bool(rewriter._entries)
    return True


__all__ = [
    "CustomPronunciations",
    "TextRewriter",
    "load_custom_pronunciations",
]
