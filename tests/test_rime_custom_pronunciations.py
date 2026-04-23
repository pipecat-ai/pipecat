from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import aiohttp
import pytest

from pipecat.services.rime import custom_pronunciations as cp_mod
from pipecat.services.rime.custom_pronunciations import (
    CustomPronunciations,
    _compile_patterns,
    _project_response,
    _rewriter_is_nonempty,
    load_custom_pronunciations,
)
from pipecat.services.rime.tts import (
    _ARCANA_SUPPRESSION_WARNING,
    RimeHttpTTSService,
    RimeTTSService,
    _apply_rewriter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prons(entries: dict[str, str]) -> CustomPronunciations:
    """Build a CustomPronunciations with the given entries, no HTTP."""
    prons = CustomPronunciations()
    prons._entries = dict(entries)
    prons._cs_pattern, prons._ci_pattern = _compile_patterns(entries)
    return prons


class _FetchScript:
    """Scriptable replacement for `_http_get_pronunciations`.

    Each call returns the next scripted response (body dict or an exception
    to raise). Tracks call count. Use to drive fetch-success, fetch-error,
    refresh-change scenarios.
    """

    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.calls = 0

    async def __call__(self, **_: Any) -> Any:
        self.calls += 1
        idx = min(self.calls - 1, len(self.responses) - 1)
        value = self.responses[idx]
        if isinstance(value, BaseException):
            raise value
        return value


def _patch_fetch(monkeypatch: pytest.MonkeyPatch, fetch: _FetchScript) -> None:
    monkeypatch.setattr(cp_mod, "_http_get_pronunciations", fetch)


def _patch_zero_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace asyncio.sleep inside custom_pronunciations with an instant sleep.

    Keeps retry/refresh-loop timing out of the way of tests.
    """
    orig_sleep = asyncio.sleep

    async def _instant(_: float) -> None:
        # Yield control once so other tasks can progress, without real delay.
        await orig_sleep(0)

    monkeypatch.setattr(cp_mod.asyncio, "sleep", _instant)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def test_project_response_extracts_default_map() -> None:
    body = {
        "default": {"Lisinopril": "l0Is1Inxpr0Il", "dr.": "d1aktx0r"},
        "vocabs": {},
    }
    assert _project_response(body) == {
        "Lisinopril": "l0Is1Inxpr0Il",
        "dr.": "d1aktx0r",
    }


def test_project_response_ignores_vocabs() -> None:
    body = {
        "default": {"foo": "f0U"},
        "vocabs": {"pediatrics": {"amoxicillin": "@m0ks0Is1Il0In"}},
    }
    assert _project_response(body) == {"foo": "f0U"}


def test_project_response_drops_invalid_entries() -> None:
    body = {
        "default": {
            "": "x",
            "good": "",
            "ok": "1kA",
            123: "bad",
            "also_ok": "ok",
        }
    }
    assert _project_response(body) == {"ok": "1kA", "also_ok": "ok"}


def test_project_response_handles_missing_default() -> None:
    assert _project_response({}) == {}
    assert _project_response({"vocabs": {}}) == {}
    assert _project_response({"default": None}) == {}
    assert _project_response({"default": "not a map"}) == {}


def test_project_response_preserves_key_case() -> None:
    body = {"default": {"AT&T": "@t and t", "at&t": "@t @mprs@nd t"}}
    projected = _project_response(body)
    assert "AT&T" in projected
    assert "at&t" in projected


# ---------------------------------------------------------------------------
# Rewrite — compile + apply
# ---------------------------------------------------------------------------


def test_rewrite_empty_mapping_returns_input_unchanged() -> None:
    prons = _make_prons({})
    assert prons.rewrite("hello world") == "hello world"


def test_rewrite_lowercase_key_matches_case_insensitively() -> None:
    prons = _make_prons({"lisinopril": "l0Is1Inxpr0Il"})
    assert prons.rewrite("take your Lisinopril") == "take your {l0Is1Inxpr0Il}"
    assert prons.rewrite("LISINOPRIL is a drug") == "{l0Is1Inxpr0Il} is a drug"
    assert prons.rewrite("lisinopril") == "{l0Is1Inxpr0Il}"


def test_rewrite_uppercase_key_matches_case_sensitively() -> None:
    prons = _make_prons({"AT&T": "1e t1i 1e t1i"})
    assert prons.rewrite("AT&T is a carrier") == "{1e t1i 1e t1i} is a carrier"
    # Case-sensitive: "at&t" should NOT match the cased key.
    assert prons.rewrite("at&t is lowercase") == "at&t is lowercase"


def test_rewrite_mixed_case_and_lowercase_keys_coexist() -> None:
    prons = _make_prons({"AT&T": "upper_pron", "at&t": "lower_pron"})
    # Cased literal wins for "AT&T"; lowercased canonical matches "at&t".
    assert prons.rewrite("AT&T and at&t") == "{upper_pron} and {lower_pron}"


def test_rewrite_longest_match_wins() -> None:
    prons = _make_prons(
        {
            "lisinopril": "L",
            "lisinopril 10mg": "L_10",
        }
    )
    assert prons.rewrite("take lisinopril 10mg now") == "take {L_10} now"
    # Without the 10mg qualifier, the shorter one fires.
    assert prons.rewrite("take lisinopril now") == "take {L} now"


def test_rewrite_trailing_punctuation_keys_match() -> None:
    prons = _make_prons({"Dr.": "d1aktx0r"})
    assert prons.rewrite("Hi Dr. Smith") == "Hi {d1aktx0r} Smith"


def test_rewrite_boundary_prevents_partial_word_matches() -> None:
    prons = _make_prons({"dr": "d1aktx0r"})
    # "dream" starts with "dr" but shouldn't match — \w lookahead blocks it.
    assert prons.rewrite("I dream of dr quick") == "I dream of {d1aktx0r} quick"


def test_rewrite_metachars_in_keys_escaped() -> None:
    prons = _make_prons({"c++": "si p1Lxs p1Lxs"})
    assert prons.rewrite("write c++ fast") == "write {si p1Lxs p1Lxs} fast"


def test_rewrite_is_callable_via_dunder_call() -> None:
    prons = _make_prons({"foo": "f0U"})
    assert prons("foo bar") == "{f0U} bar"


def test_rewriter_is_nonempty_introspects_custom_pronunciations() -> None:
    assert _rewriter_is_nonempty(_make_prons({})) is False
    assert _rewriter_is_nonempty(_make_prons({"a": "1"})) is True


def test_rewriter_is_nonempty_assumes_callable_nonempty() -> None:
    assert _rewriter_is_nonempty(None) is False
    assert _rewriter_is_nonempty(lambda s: s) is True


# ---------------------------------------------------------------------------
# load_custom_pronunciations — happy path + errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_fetches_and_populates(monkeypatch: pytest.MonkeyPatch) -> None:
    fetch = _FetchScript({"default": {"hello": "h1El0U"}, "vocabs": {}})
    _patch_fetch(monkeypatch, fetch)

    prons = await load_custom_pronunciations(api_key="test", refresh_interval=None)
    try:
        assert fetch.calls == 1
        assert prons.entries == {"hello": "h1El0U"}
        assert prons.rewrite("hello world") == "{h1El0U} world"
    finally:
        await prons.aclose()


@pytest.mark.asyncio
async def test_load_without_refresh_interval_spawns_no_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch = _FetchScript({"default": {}, "vocabs": {}})
    _patch_fetch(monkeypatch, fetch)

    prons = await load_custom_pronunciations(api_key="k", refresh_interval=None)
    try:
        assert prons._refresh_task is None
    finally:
        await prons.aclose()


@pytest.mark.asyncio
async def test_load_with_refresh_interval_spawns_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch = _FetchScript({"default": {}, "vocabs": {}})
    _patch_fetch(monkeypatch, fetch)

    prons = await load_custom_pronunciations(api_key="k", refresh_interval=300)
    try:
        assert prons._refresh_task is not None
        assert not prons._refresh_task.done()
    finally:
        await prons.aclose()


@pytest.mark.asyncio
async def test_load_clamps_below_minimum_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch = _FetchScript({"default": {}, "vocabs": {}})
    _patch_fetch(monkeypatch, fetch)

    prons = await load_custom_pronunciations(api_key="k", refresh_interval=5)
    try:
        # Clamped to _MIN_REFRESH_INTERVAL (30s); we observe clamping via the
        # task being spawned (would not spawn at all if interval were treated
        # as invalid rather than clamped).
        assert prons._refresh_task is not None
    finally:
        await prons.aclose()


@pytest.mark.asyncio
async def test_load_raises_immediately_on_auth_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    req_info = aiohttp.RequestInfo(
        url=aiohttp.client.URL("https://users.rime.ai/speech-qa/custom-pronunciations"),
        method="GET",
        headers=aiohttp.typedefs.CIMultiDict(),
        real_url=aiohttp.client.URL("https://users.rime.ai/speech-qa/custom-pronunciations"),
    )
    auth_err = aiohttp.ClientResponseError(
        request_info=req_info,
        history=(),
        status=401,
        message="Unauthorized",
    )
    fetch = _FetchScript(auth_err)
    _patch_fetch(monkeypatch, fetch)

    with pytest.raises(aiohttp.ClientResponseError) as excinfo:
        await load_custom_pronunciations(api_key="k", refresh_interval=None)
    assert excinfo.value.status == 401
    # No retries on 4xx.
    assert fetch.calls == 1


@pytest.mark.asyncio
async def test_load_retries_transient_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_zero_sleep(monkeypatch)
    req_info = aiohttp.RequestInfo(
        url=aiohttp.client.URL("https://users.rime.ai/speech-qa/custom-pronunciations"),
        method="GET",
        headers=aiohttp.typedefs.CIMultiDict(),
        real_url=aiohttp.client.URL("https://users.rime.ai/speech-qa/custom-pronunciations"),
    )
    transient = aiohttp.ClientResponseError(
        request_info=req_info, history=(), status=503, message="Unavailable"
    )
    fetch = _FetchScript(
        transient,
        {"default": {"foo": "f0U"}, "vocabs": {}},
    )
    _patch_fetch(monkeypatch, fetch)

    prons = await load_custom_pronunciations(api_key="k", refresh_interval=None)
    try:
        assert fetch.calls == 2
        assert prons.entries == {"foo": "f0U"}
    finally:
        await prons.aclose()


@pytest.mark.asyncio
async def test_load_raises_after_exhausting_transient_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_zero_sleep(monkeypatch)
    req_info = aiohttp.RequestInfo(
        url=aiohttp.client.URL("https://users.rime.ai/speech-qa/custom-pronunciations"),
        method="GET",
        headers=aiohttp.typedefs.CIMultiDict(),
        real_url=aiohttp.client.URL("https://users.rime.ai/speech-qa/custom-pronunciations"),
    )
    transient = aiohttp.ClientResponseError(
        request_info=req_info, history=(), status=503, message="Unavailable"
    )
    fetch = _FetchScript(transient, transient, transient, transient)
    _patch_fetch(monkeypatch, fetch)

    with pytest.raises(aiohttp.ClientResponseError):
        await load_custom_pronunciations(api_key="k", refresh_interval=None)
    # Initial + 2 retries.
    assert fetch.calls == 3


@pytest.mark.asyncio
async def test_load_without_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RIME_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Rime API key"):
        await load_custom_pronunciations(refresh_interval=None)


# ---------------------------------------------------------------------------
# Refresh — manual trigger
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refresh_updates_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    fetch = _FetchScript(
        {"default": {"one": "1"}, "vocabs": {}},
        {"default": {"one": "1", "two": "2"}, "vocabs": {}},
    )
    _patch_fetch(monkeypatch, fetch)

    prons = await load_custom_pronunciations(api_key="k", refresh_interval=None)
    try:
        assert prons.entries == {"one": "1"}
        await prons.refresh()
        assert prons.entries == {"one": "1", "two": "2"}
        assert fetch.calls == 2
    finally:
        await prons.aclose()


@pytest.mark.asyncio
async def test_refresh_coalesces_concurrent_callers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A fetch that yields control so the coalescing window is observable.
    call_count = {"n": 0}

    async def slow_fetch(**_: Any) -> Any:
        call_count["n"] += 1
        # Yield a few times to keep the fetch "in flight" while callers pile up.
        for _i in range(3):
            await asyncio.sleep(0)
        return {"default": {"x": "X"}, "vocabs": {}}

    monkeypatch.setattr(cp_mod, "_http_get_pronunciations", slow_fetch)

    prons = await load_custom_pronunciations(api_key="k", refresh_interval=None)
    # After load, one call has already happened.
    try:
        assert call_count["n"] == 1
        # Fire five concurrent refreshes; they should coalesce.
        await asyncio.gather(*(prons.refresh() for _ in range(5)))
        # Exactly one additional fetch for the five concurrent calls.
        assert call_count["n"] == 2
    finally:
        await prons.aclose()


@pytest.mark.asyncio
async def test_refresh_after_close_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    fetch = _FetchScript({"default": {}, "vocabs": {}})
    _patch_fetch(monkeypatch, fetch)
    prons = await load_custom_pronunciations(api_key="k", refresh_interval=None)
    await prons.aclose()

    # No call is issued; refresh returns cleanly.
    await prons.refresh()
    assert fetch.calls == 1


# ---------------------------------------------------------------------------
# Refresh loop — automatic ticks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refresh_loop_picks_up_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    # Relax the 30s floor so we can exercise the loop in test-time.
    monkeypatch.setattr(cp_mod, "_MIN_REFRESH_INTERVAL", 0.01)

    fetch = _FetchScript(
        {"default": {"one": "1"}, "vocabs": {}},
        {"default": {"two": "2"}, "vocabs": {}},
        {"default": {"two": "2", "three": "3"}, "vocabs": {}},
    )
    _patch_fetch(monkeypatch, fetch)

    prons = await load_custom_pronunciations(api_key="k", refresh_interval=0.01)
    try:
        # Initial state from load().
        assert prons.entries == {"one": "1"}
        # Allow the refresh loop to tick at least twice.
        for _ in range(50):
            await asyncio.sleep(0.01)
            if fetch.calls >= 3:
                break
        assert fetch.calls >= 3
        assert "three" in prons.entries
    finally:
        await prons.aclose()


@pytest.mark.asyncio
async def test_refresh_loop_swallows_fetch_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cp_mod, "_MIN_REFRESH_INTERVAL", 0.01)

    req_info = aiohttp.RequestInfo(
        url=aiohttp.client.URL("https://users.rime.ai/speech-qa/custom-pronunciations"),
        method="GET",
        headers=aiohttp.typedefs.CIMultiDict(),
        real_url=aiohttp.client.URL("https://users.rime.ai/speech-qa/custom-pronunciations"),
    )
    transient = aiohttp.ClientResponseError(
        request_info=req_info, history=(), status=503, message="Unavailable"
    )
    fetch = _FetchScript(
        {"default": {"stable": "s"}, "vocabs": {}},
        transient,
        transient,
        {"default": {"stable": "s", "new": "n"}, "vocabs": {}},
    )
    _patch_fetch(monkeypatch, fetch)

    prons = await load_custom_pronunciations(api_key="k", refresh_interval=0.01)
    try:
        assert prons.entries == {"stable": "s"}
        # Wait until the recovery fetch lands.
        for _ in range(100):
            await asyncio.sleep(0.01)
            if "new" in prons.entries:
                break
        assert prons.entries == {"stable": "s", "new": "n"}
    finally:
        await prons.aclose()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_with_closes_refresh_task(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cp_mod, "_MIN_REFRESH_INTERVAL", 0.01)
    fetch = _FetchScript({"default": {}, "vocabs": {}})
    _patch_fetch(monkeypatch, fetch)

    async with await load_custom_pronunciations(api_key="k", refresh_interval=0.01) as prons:
        task = prons._refresh_task
        assert task is not None
        assert not task.done()
    assert task.done()
    assert prons._closed is True


@pytest.mark.asyncio
async def test_double_aclose_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    fetch = _FetchScript({"default": {}, "vocabs": {}})
    _patch_fetch(monkeypatch, fetch)

    prons = await load_custom_pronunciations(api_key="k", refresh_interval=None)
    await prons.aclose()
    await prons.aclose()  # should not raise


# ---------------------------------------------------------------------------
# _apply_rewriter — direct tests of the module-level helper
# ---------------------------------------------------------------------------


def test_apply_rewriter_none_returns_text_unchanged() -> None:
    flag = [False]
    assert _apply_rewriter(None, "hello foo", "mistv2", flag, "svc") == "hello foo"
    assert flag == [False]


def test_apply_rewriter_mist_applies() -> None:
    prons = _make_prons({"foo": "f0U"})
    flag = [False]
    assert _apply_rewriter(prons, "foo bar", "mistv2", flag, "svc") == "{f0U} bar"
    assert flag == [False]


def test_apply_rewriter_callable_applies() -> None:
    flag = [False]
    out = _apply_rewriter(lambda s: s.upper(), "foo", "mistv2", flag, "svc")
    assert out == "FOO"
    assert flag == [False]


def test_apply_rewriter_arcana_nonempty_warns_and_passes_through() -> None:
    prons = _make_prons({"foo": "f0U"})
    flag = [False]
    assert _apply_rewriter(prons, "foo bar", "arcana", flag, "svc") == "foo bar"
    assert flag == [True]


def test_apply_rewriter_arcana_empty_does_not_warn() -> None:
    prons = _make_prons({})
    flag = [False]
    assert _apply_rewriter(prons, "foo bar", "arcana", flag, "svc") == "foo bar"
    assert flag == [False]


def test_apply_rewriter_arcana_callable_warns_and_passes_through() -> None:
    flag = [False]
    out = _apply_rewriter(lambda s: s.upper(), "foo", "arcana", flag, "svc")
    assert out == "foo"
    assert flag == [True]


def test_apply_rewriter_arcana_warns_once_across_calls() -> None:
    prons = _make_prons({"foo": "f0U"})
    flag = [False]
    _apply_rewriter(prons, "foo", "arcana", flag, "svc")
    assert flag == [True]
    # Subsequent Arcana calls with a still-nonempty rewriter do not re-warn.
    # The flag is the side-effect; we can only observe it flipped once.
    _apply_rewriter(prons, "foo", "arcana", flag, "svc")
    _apply_rewriter(prons, "foo", "arcana", flag, "svc")
    assert flag == [True]


def test_apply_rewriter_swallows_rewriter_exception() -> None:
    def broken(_: str) -> str:
        raise RuntimeError("bad rewriter")

    flag = [False]
    assert _apply_rewriter(broken, "foo", "mistv2", flag, "svc") == "foo"


# ---------------------------------------------------------------------------
# RimeHttpTTSService integration
# ---------------------------------------------------------------------------

# RimeHttpTTSService is used for construction-level assertions because its
# __init__ does not open any network connection — unlike RimeTTSService which
# connects only in start(), but still performs extra framework wiring we can
# sidestep with the HTTP variant.


def _make_http_service(
    *, model: str = "mistv2", custom_pronunciations: Any = None
) -> RimeHttpTTSService:
    return RimeHttpTTSService(
        api_key="k",
        aiohttp_session=MagicMock(spec=aiohttp.ClientSession),
        settings=RimeHttpTTSService.Settings(model=model, voice="luna"),
        custom_pronunciations=custom_pronunciations,
    )


def _make_ws_service(
    *, model: str = "mistv2", custom_pronunciations: Any = None
) -> RimeTTSService:
    return RimeTTSService(
        api_key="k",
        settings=RimeTTSService.Settings(model=model, voice="luna"),
        custom_pronunciations=custom_pronunciations,
    )


def test_http_service_accepts_custom_pronunciations_object() -> None:
    prons = _make_prons({"foo": "f0U"})
    svc = _make_http_service(custom_pronunciations=prons)
    assert svc._custom_pronunciations is prons
    assert svc._arcana_warning_logged == [False]


def test_http_service_accepts_callable_rewriter() -> None:
    def fn(s: str) -> str:
        return s.replace("a", "{@}")

    svc = _make_http_service(custom_pronunciations=fn)
    assert svc._custom_pronunciations is fn


def test_http_service_default_custom_pronunciations_is_none() -> None:
    svc = _make_http_service()
    assert svc._custom_pronunciations is None


def test_ws_service_accepts_custom_pronunciations_object() -> None:
    prons = _make_prons({"foo": "f0U"})
    svc = _make_ws_service(custom_pronunciations=prons)
    assert svc._custom_pronunciations is prons
    assert svc._arcana_warning_logged == [False]


def test_ws_service_set_custom_pronunciations_swaps() -> None:
    p1 = _make_prons({"foo": "f0U"})
    p2 = _make_prons({"bar": "b1A"})
    svc = _make_ws_service(custom_pronunciations=p1)
    assert svc._custom_pronunciations is p1
    svc.set_custom_pronunciations(p2)
    assert svc._custom_pronunciations is p2
    svc.set_custom_pronunciations(None)
    assert svc._custom_pronunciations is None


def test_http_service_arcana_nonempty_prons_warns_and_passes_through() -> None:
    prons = _make_prons({"foo": "f0U"})
    svc = _make_http_service(model="arcana", custom_pronunciations=prons)
    # Exercise the same helper run_tts uses, with the service's state.
    out = _apply_rewriter(
        svc._custom_pronunciations,
        "foo bar",
        svc._settings.model,
        svc._arcana_warning_logged,
        str(svc),
    )
    assert out == "foo bar"
    assert svc._arcana_warning_logged == [True]


def test_http_service_model_swap_activates_and_clears_suppression() -> None:
    # Cross-model behavior via direct _settings manipulation — we don't need
    # update_settings' reconnect machinery to exercise the rewrite logic.
    prons = _make_prons({"foo": "f0U"})
    svc = _make_http_service(model="mistv2", custom_pronunciations=prons)

    # Mist: rewrites.
    assert (
        _apply_rewriter(
            svc._custom_pronunciations,
            "foo",
            svc._settings.model,
            svc._arcana_warning_logged,
            str(svc),
        )
        == "{f0U}"
    )
    assert svc._arcana_warning_logged == [False]

    # Swap to Arcana: warns once, passes through.
    svc._settings.model = "arcana"
    out = _apply_rewriter(
        svc._custom_pronunciations,
        "foo",
        svc._settings.model,
        svc._arcana_warning_logged,
        str(svc),
    )
    assert out == "foo"
    assert svc._arcana_warning_logged == [True]

    # Back to Mist: rewriter applies again, no suppression state to clear.
    svc._settings.model = "mistv2"
    assert (
        _apply_rewriter(
            svc._custom_pronunciations,
            "foo",
            svc._settings.model,
            svc._arcana_warning_logged,
            str(svc),
        )
        == "{f0U}"
    )

    # Arcana again: still warns-once flag stays True; no second warning.
    svc._settings.model = "arcana"
    out = _apply_rewriter(
        svc._custom_pronunciations,
        "foo",
        svc._settings.model,
        svc._arcana_warning_logged,
        str(svc),
    )
    assert out == "foo"
    assert svc._arcana_warning_logged == [True]


@pytest.mark.asyncio
async def test_shared_prons_across_services_not_closed_by_service_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch = _FetchScript({"default": {"foo": "f0U"}, "vocabs": {}})
    _patch_fetch(monkeypatch, fetch)
    prons = await load_custom_pronunciations(api_key="k", refresh_interval=None)
    try:
        svc1 = _make_http_service(custom_pronunciations=prons)
        svc2 = _make_ws_service(custom_pronunciations=prons)

        # The service holds a reference, but its lifecycle is independent.
        del svc1
        del svc2

        assert prons._closed is False
        assert prons.rewrite("foo") == "{f0U}"
    finally:
        await prons.aclose()


def test_shared_prons_rewrite_picks_up_changes_across_services() -> None:
    prons = _make_prons({"foo": "f0U"})
    svc1 = _make_http_service(custom_pronunciations=prons)
    svc2 = _make_ws_service(custom_pronunciations=prons)

    # Simulate a refresh that replaces entries.
    new_entries = {"bar": "b1A"}
    prons._entries = new_entries
    prons._cs_pattern, prons._ci_pattern = _compile_patterns(new_entries)

    assert svc1._custom_pronunciations.rewrite("bar") == "{b1A}"
    assert svc2._custom_pronunciations.rewrite("bar") == "{b1A}"


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def test_module_exports_expected_surface() -> None:
    """Customers import from these two submodules — verify the shape."""
    from pipecat.services.rime import custom_pronunciations, tts

    assert custom_pronunciations.CustomPronunciations is CustomPronunciations
    assert custom_pronunciations.load_custom_pronunciations is load_custom_pronunciations
    assert hasattr(custom_pronunciations, "TextRewriter")
    assert tts.RimeTTSService is RimeTTSService
    assert tts.RimeHttpTTSService is RimeHttpTTSService


def test_arcana_suppression_warning_is_informative() -> None:
    """The module-level warning string should mention the key facts."""
    assert "arcana" in _ARCANA_SUPPRESSION_WARNING.lower()
    assert "{phoneme}" in _ARCANA_SUPPRESSION_WARNING or "Mist" in _ARCANA_SUPPRESSION_WARNING
