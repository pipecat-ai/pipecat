#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Scenario file format for Pipecat behavioral evaluations.

A scenario is a YAML file describing a scripted conversation and the semantic
events expected to flow back from the bot. Simple example::

    name: simple_user_input
    turns:
      - user: "hello world"
        expect:
          - event: user_started_speaking
          - event: user_transcription
            text_contains: "hello world"

The runner (see :mod:`pipecat.evals.harness`) loads the scenario, connects to
the bot's eval transport over RTVI, drives each turn, collects the RTVI events
the bot emits, and asserts on them in order.

Event names are the friendly names the harness maps RTVI server messages onto:
``user_started_speaking``, ``user_stopped_speaking``, ``user_transcription``,
``llm_started``, ``response``, ``llm_response``, ``tts_response``,
``function_call``.

The bot's reply can be asserted three ways:
    response       the transcription of the bot's *actual synthesized audio* (a
                   local Whisper model run by the harness) in audio modality, or
                   the LLM text in text modality. The real end-to-end check —
                   prefer this.
    llm_response   the LLM's text output (``bot-llm-text``). Available in both
                   modalities.
    tts_response   the text the TTS reports speaking (``bot-tts-text``, with
                   word timing). Audio modality only.

Supported expectation fields (per event):
    event: <name>              required — event type name
    within_ms: <int>           latency budget from the most recent anchor
                               (optional; defaults to 60s when omitted)
    text_contains: <str>       substring check on the event's text content
    name: <str>                function_call.name equality
    args: <object>             function_call.args equality
    eval: <str>                natural-language criterion the event's text content
                               must satisfy, evaluated by a judge LLM (see
                               :mod:`pipecat.evals.judge`).

A turn may also include ``send_after:`` to schedule its user send relative to a
prior event (used for interruption / barge-in tests).

Top-level optional fields:
    reset:  list of LLM messages the harness sends as a reset before driving
            turns (default empty list, which clears the bot's context).
    user:   how user turns are delivered::

                user:
                  modality: audio          # audio | text (default text)
                  speech:                  # required when modality is audio
                    service: cartesia      # TTS that synthesizes the user turns
                    voice: <voice-id>
                    model: sonic-2         # optional
                    sample_rate: 16000     # optional

            ``audio`` streams synthesized user audio to the bot (exercising its
            STT for real); ``text`` (the default) sends RTVI ``send-text``.
    judge:  what the judge evaluates, and with which LLM::

                judge:
                  modality: audio          # audio | text (default text)
                  eval:                    # the judge LLM (default ollama)
                    service: openai
                    model: gpt-4o-mini
                  transcription:           # required when modality is audio
                    service: whisper       # STT for the bot's audio
                    model: base            # optional

            ``audio`` makes the bot speak and judges the transcription of its
            actual audio (``tts_response``); ``text`` (the default) skips TTS and
            judges the LLM text (``llm_response``), which is faster and silent.
    fixtures: free-form mapping (e.g. ``bot_url:``).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

try:
    import yaml
except ModuleNotFoundError as e:
    logger.error("PyYAML is required for the scenario runner. Install with: pip install pyyaml")
    raise ImportError(f"Missing module: {e}") from e

# Events whose payloads carry bot-generated text the judge can sensibly
# evaluate. Asserting ``eval:`` on anything else (user transcripts, tool
# calls, interruption signals) produces a parser warning — the test controls
# user input deterministically, so judging it adds cost without signal.
# ``response`` is the modality-agnostic alias, resolved to one of the others
# after parsing (see _resolve_response_events).
JUDGEABLE_EVENTS = frozenset({"response", "llm_response", "tts_response"})


@dataclass
class Expectation:
    """A single expected event in a scenario turn.

    Parameters:
        event: Required — the semantic event name (e.g. ``user_stopped_speaking``).
        within_ms: Optional latency budget, measured from the most recent anchor
            (typically the preceding ``user_input`` send for the first event of
            a turn, or the previous matched event otherwise). Defaults to 60s
            when omitted, so timing isn't asserted unless set explicitly.
        text_contains: Optional substring check on the event's text content
            (``llm_response.text`` or ``user_transcription.transcript``).
        name: Optional equality check for ``function_call.name``.
        args: Optional equality check for ``function_call.args``.
        eval: Optional natural-language criterion the event's text content
            must satisfy. Evaluated by a judge LLM. Only meaningful on
            ``llm_response`` (the text the bot produced for this turn).
        raw: The original parsed dict, for forward compatibility.
    """

    event: str
    within_ms: int | None = None
    text_contains: str | None = None
    name: str | None = None
    args: dict | None = None
    eval: str | None = None
    raw: dict = field(default_factory=dict)


@dataclass
class SendAfter:
    """Event-driven scheduling for a turn's ``user_input`` send.

    When set on a :class:`Turn`, the harness waits for ``event`` to have been
    seen (either earlier in the run or arriving now), then waits an additional
    ``delay_ms`` before sending the turn's ``user`` text. Used for barge-in
    tests: ``send_after: {event: llm_started, delay_ms: 500}`` means
    "interrupt 500ms after the bot started responding."

    Parameters:
        event: Name of the event to schedule from.
        delay_ms: Additional delay in milliseconds after the event was received.
    """

    event: str
    delay_ms: int


@dataclass
class Turn:
    """One turn in a scenario.

    A turn is either driven by the harness sending a ``user`` utterance, or it
    is observation-only (no ``user`` field — useful for bot-first scenarios
    like opening greetings).

    Parameters:
        user: Optional text to send as ``{"type": "user_input", "text": ...}``.
            If absent, the turn just waits for and asserts on expected events.
        expect: Expected events, in the order they should arrive.
        send_after: Optional event-driven schedule for when the ``user`` send
            should fire. Only meaningful when ``user`` is set.
    """

    user: str | None
    expect: list[Expectation]
    send_after: SendAfter | None = None


@dataclass
class Scenario:
    """A parsed scenario file.

    Parameters:
        name: The eval name (from ``name:``).
        turns: Ordered list of turns.
        reset: Messages to seed the bot's LLM context with before this eval
            runs. Sent by the harness as a ``{"type": "reset", "messages":
            ...}`` message right after the initial ``ready`` handshake. Empty
            list (the default) clears the context entirely; bots without an
            LLM context aggregator ignore the resulting frame.
        judge: Judge LLM configuration dict with keys ``service``, ``model``,
            and optional ``endpoint``. Defaults to
            ``{"service": "ollama", "model": "qwen2.5:3b"}``.
        bot_audio: Whether the bot produces speech (parsed from a bool or a
            mapping). Default False: the bot skips TTS, the harness configures
            skip-TTS at connect, and even an on-connect greeting is silent. True
            (or a mapping) makes the bot speak. A mapping additionally enables
            ``tts_response`` and configures the STT (``service`` / ``model``)
            used to transcribe the bot's audio (see :attr:`transcriber`).
        transcriber: Parsed from a mapping ``bot_audio``; the STT config
            (``service`` defaults to ``whisper``, plus ``model``) used to
            transcribe the bot's audio for ``tts_response`` (``None`` when
            ``bot_audio`` is a plain bool).
        user_audio: TTS config the harness uses to generate user audio. When
            present, the harness streams RTVI ``raw-audio`` (not ``send-text``)
            to the bot, exercising its STT for real. Mapping with ``service``,
            ``voice``, and optional ``model`` / ``sample_rate`` /
            ``api_key``. Omit for text-only evals (default).
        fixtures: Optional fixtures dict (e.g. ``bot_url:``).
        source_path: Path the scenario was loaded from, for error messages.
    """

    name: str
    turns: list[Turn]
    reset: list[dict] = field(default_factory=list)
    judge: dict = field(default_factory=lambda: {"service": "ollama", "model": "qwen2.5:3b"})
    bot_audio: bool = False
    transcriber: dict | None = None
    user_audio: dict | None = None
    fixtures: dict = field(default_factory=dict)
    source_path: Path | None = None


def load_scenario(path: str | Path) -> Scenario:
    """Parse a scenario YAML file into a :class:`Scenario`.

    Args:
        path: Path to a YAML file with the scenario schema.

    Returns:
        The parsed scenario.

    Raises:
        ValueError: If the file structure is invalid.
        FileNotFoundError: If the path doesn't exist.
    """
    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")

    name = data.get("name")
    if not name or not isinstance(name, str):
        raise ValueError(f"{path}: missing or invalid 'name:' field")

    raw_turns = data.get("turns")
    if not isinstance(raw_turns, list):
        raise ValueError(f"{path}: 'turns:' must be a list")

    turns = [_parse_turn(t, path, idx) for idx, t in enumerate(raw_turns)]

    raw_reset = data.get("reset")
    if raw_reset is None:
        reset: list[dict] = []
    elif isinstance(raw_reset, list):
        reset = raw_reset
    else:
        raise ValueError(f"{path}: 'reset:' must be a list of message dicts")

    # user: { modality: audio|text, speech: {...} }. Audio synthesizes each user
    # turn via TTS (exercising the bot's STT); text sends it as text. Stored
    # internally as user_audio (the speech config when audio, else None).
    user_audio = _parse_user_block(data.get("user"), path)

    # judge: { modality: audio|text, eval: {...}, transcription: {...} }. Audio
    # means the bot speaks and the judge evaluates the transcription of its
    # actual audio (tts_response); text means the bot's LLM text directly
    # (llm_response, bot skips TTS). Stored as bot_audio/transcriber/judge.
    bot_audio, transcriber, judge = _parse_judge_block(data.get("judge"), path)

    # Resolve the modality-agnostic `response` event and check event/modality
    # consistency now that the judge modality is known.
    _resolve_response_events(turns, bot_audio, path)

    return Scenario(
        name=name,
        turns=turns,
        reset=reset,
        judge=judge,
        bot_audio=bot_audio,
        transcriber=transcriber,
        user_audio=user_audio,
        fixtures=data.get("fixtures") or {},
        source_path=path,
    )


_DEFAULT_JUDGE = {"service": "ollama", "model": "qwen2.5:3b"}


def _parse_user_block(user: Any, path: Path) -> dict | None:
    """Parse the ``user:`` block into the internal user_audio (speech config or None)."""
    if user is None:
        return None  # default: text modality
    if not isinstance(user, dict):
        raise ValueError(f"{path}: 'user:' must be a mapping")
    modality = user.get("modality", "text")
    if modality not in ("audio", "text"):
        raise ValueError(f"{path}: 'user.modality:' must be 'audio' or 'text', got {modality!r}")
    if modality == "text":
        return None
    speech = user.get("speech")
    if not isinstance(speech, dict):
        raise ValueError(
            f"{path}: 'user.modality: audio' requires a 'user.speech:' block "
            "(TTS service + voice to synthesize the user's turns)"
        )
    return speech


def _parse_judge_block(judge: Any, path: Path) -> tuple[bool, dict | None, dict]:
    """Parse the ``judge:`` block into (bot_audio, transcriber, eval-config)."""
    if judge is None:
        judge = {}
    if not isinstance(judge, dict):
        raise ValueError(f"{path}: 'judge:' must be a mapping")
    modality = judge.get("modality", "text")
    if modality not in ("audio", "text"):
        raise ValueError(f"{path}: 'judge.modality:' must be 'audio' or 'text', got {modality!r}")
    eval_cfg = judge.get("eval") or dict(_DEFAULT_JUDGE)
    if not isinstance(eval_cfg, dict):
        raise ValueError(f"{path}: 'judge.eval:' must be a mapping (the judge LLM service)")
    if modality == "text":
        return False, None, eval_cfg
    transcription = judge.get("transcription")
    if not isinstance(transcription, dict):
        raise ValueError(
            f"{path}: 'judge.modality: audio' requires a 'judge.transcription:' block "
            "(STT service to transcribe the bot's audio)"
        )
    return True, transcription, eval_cfg


def _resolve_response_events(turns: list[Turn], bot_audio: bool, path: Path) -> None:
    """Resolve the modality-agnostic ``response`` event and validate consistency.

    In audio modality ``response`` is the transcription of the bot's actual
    audio, so it stays as ``response``. In text modality there is no audio, so it
    falls back to ``llm_response``. ``tts_response`` (the TTS's spoken text) needs
    the bot to speak, so asserting it in text modality is an error.
    """
    for ti, turn in enumerate(turns):
        for exp in turn.expect:
            if exp.event == "response" and not bot_audio:
                exp.event = "llm_response"
            elif exp.event == "tts_response" and not bot_audio:
                raise ValueError(
                    f"{path}: turn #{ti} asserts 'tts_response' but 'judge.modality' is text "
                    "(the bot doesn't speak). Use 'response'/'llm_response', or set "
                    "'judge.modality: audio'."
                )


def describe_config(scenario: Scenario) -> str:
    """One-line summary of a scenario's modalities and services, for pre-run logs."""
    if scenario.user_audio:
        user = f"audio (speech: {scenario.user_audio.get('service', '?')})"
    else:
        user = "text"
    eval_cfg = scenario.judge or {}
    eval_svc = f"{eval_cfg.get('service', '?')}/{eval_cfg.get('model', '?')}"
    if scenario.bot_audio:
        transcription = (scenario.transcriber or {}).get("service", "whisper")
        judge = f"audio (eval: {eval_svc}, transcription: {transcription})"
    else:
        judge = f"text (eval: {eval_svc})"
    return f"user: {user}  |  judge: {judge}"


def _parse_turn(t: Any, path: Path, idx: int) -> Turn:
    """Parse one entry from the ``turns:`` list."""
    if not isinstance(t, dict):
        raise ValueError(f"{path}: turn #{idx} must be a mapping")

    user = t.get("user")
    if user is not None and not isinstance(user, str):
        raise ValueError(f"{path}: turn #{idx} 'user:' must be a string if present")

    raw_expect = t.get("expect")
    if not isinstance(raw_expect, list):
        raise ValueError(f"{path}: turn #{idx} missing or invalid 'expect:' list")

    expect = [_parse_expectation(e, path, idx, ei) for ei, e in enumerate(raw_expect)]

    send_after = _parse_send_after(t.get("send_after"), path, idx) if "send_after" in t else None
    if send_after is not None and user is None:
        raise ValueError(
            f"{path}: turn #{idx} has 'send_after:' but no 'user:' — "
            "send_after only schedules when the user message gets sent"
        )

    return Turn(user=user, expect=expect, send_after=send_after)


def _parse_send_after(s: Any, path: Path, turn_idx: int) -> SendAfter:
    """Parse a ``send_after:`` block."""
    if not isinstance(s, dict):
        raise ValueError(f"{path}: turn #{turn_idx} 'send_after:' must be a mapping")

    event = s.get("event")
    if not event or not isinstance(event, str):
        raise ValueError(f"{path}: turn #{turn_idx} 'send_after:' missing or invalid 'event:'")

    delay_ms = s.get("delay_ms", 0)
    if not isinstance(delay_ms, int) or delay_ms < 0:
        raise ValueError(
            f"{path}: turn #{turn_idx} 'send_after.delay_ms' must be a non-negative int"
        )

    return SendAfter(event=event, delay_ms=delay_ms)


def _parse_expectation(e: Any, path: Path, turn_idx: int, exp_idx: int) -> Expectation:
    """Parse one entry from a turn's ``expect:`` list."""
    if not isinstance(e, dict):
        raise ValueError(f"{path}: turn #{turn_idx} expectation #{exp_idx} must be a mapping")

    event = e.get("event")
    if not event or not isinstance(event, str):
        raise ValueError(
            f"{path}: turn #{turn_idx} expectation #{exp_idx} missing or invalid 'event:'"
        )

    criterion = e.get("eval")
    if criterion is not None and event not in JUDGEABLE_EVENTS:
        logger.warning(
            f"{path}: turn #{turn_idx} expectation #{exp_idx}: 'eval:' on "
            f"event {event!r} — judge only makes sense on bot-generated text "
            f"events ({', '.join(sorted(JUDGEABLE_EVENTS))}). Will run but is "
            "unlikely to be meaningful."
        )

    return Expectation(
        event=event,
        within_ms=e.get("within_ms"),
        text_contains=e.get("text_contains"),
        name=e.get("name"),
        args=e.get("args"),
        eval=criterion,
        raw=e,
    )
