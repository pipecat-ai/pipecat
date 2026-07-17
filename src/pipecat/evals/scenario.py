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
``user_started_speaking``, ``user_stopped_speaking``, ``vad_user_started_speaking``,
``vad_user_stopped_speaking``, ``user_transcription``, ``llm_started``, ``response``,
``llm_response``, ``tts_response``, ``function_call``. The ``vad_*`` events are the raw
VAD signal, useful as a timing anchor when a turn-detection strategy gates or defers the
turn-level ``user_stopped_speaking`` (e.g. filtering incomplete turns).

The bot's reply can be asserted three ways:
    response       the transcription of the bot's *actual synthesized audio* (a
                   local STT — Moonshine or Whisper — run by the harness) in
                   audio modality, or the LLM text in text modality. The real
                   end-to-end check — prefer this.
    llm_response   the LLM's text output (``bot-llm-text``). Available in both
                   modalities.
    tts_response   the text the TTS reports speaking (``bot-tts-text``, with
                   word timing). Audio modality only.

Supported expectation fields (per event):
    event: <name>              required — event type name
    within_ms: <int>           latency budget from the most recent anchor
                               (optional; defaults to 60s when omitted)
    text_contains: <str>       substring check on the event's text content
    calls:                     for ``function_call`` — the set of calls the turn
                               should make, matched by name in any order; the
                               expectation passes only when all are found::

                                   - event: function_call
                                     calls:
                                       - name: get_current_weather
                                         args: { location: San Francisco }
                                       - name: get_restaurant_recommendation

    eval: <str>                natural-language criterion the event's text content
                               must satisfy, evaluated by a judge LLM (see
                               :mod:`pipecat.evals.judge`).

    absent: true               invert the expectation: assert that NO event of
                               this type arrives before the ``within_ms`` budget
                               expires (default 60s — set ``within_ms`` explicitly
                               to keep the quiet-window wait short). Matches on
                               event type only, so it cannot be combined with
                               ``text_contains``, ``eval:``, or ``calls:``. Used
                               for duplicate-output regressions::

                                   - event: response
                                     eval: "answers the question"
                                   - event: response
                                     absent: true
                                     within_ms: 30000

Instead of ``user:``, a turn may press DTMF keys with ``dtmf:`` (the two are
mutually exclusive — you press keys or you talk)::

    turns:
      - dtmf: "123#"            # quote it: an unquoted # starts a YAML comment
        expect:
          - event: user_transcription
            text_contains: "DTMF: 123#"
          - event: response
            eval: "confirms the entered digits"

Each character is sent as one ``InputDTMFFrame`` (``0``-``9``, ``*``, ``#``),
regardless of the scenario's user/judge modality. A bot running a
``DTMFAggregator`` accumulates them and flushes — on the ``#`` terminator or its
idle timeout — into a ``DTMF: ...`` transcription it reacts to.

A turn may also include ``send_after:`` to schedule its ``user``/``dtmf`` send
relative to a prior event (used for interruption / barge-in tests), or
``image:`` (a path, relative to the scenario file) to register an image for the
turn — when a function-calling-video bot requests a user image, the eval
transport serves it. ``send_after`` with only ``delay_ms`` (no ``event``) is a
pure time delay relative to the previous send — handy for pacing keypresses
across ``dtmf`` turns to exercise the aggregator's idle-timeout flush.

``expect:`` is optional; omit it for a turn that only sends input or only waits.

Top-level optional fields:
    context: LLM messages the bot's context should start from. When given, the
            harness sends them before driving turns (replacing the bot's
            context); omit to leave the bot's own context untouched.
    user:   how user turns are delivered::

                user:
                  modality: audio          # audio | text (default text)
                  speech:                  # required when modality is audio
                    service: kokoro        # local TTS that synthesizes the user turns
                    voice: af_heart
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
                    service: moonshine     # STT for the bot's audio (or whisper)
                    model: small-streaming # optional
                    padding_secs: 0        # optional; silence padded around the
                                           # segment (default: 2)

            ``audio`` makes the bot speak and judges the transcription of its
            actual audio (``tts_response``); ``text`` (the default) skips TTS and
            judges the LLM text (``llm_response``), which is faster and silent.

Any value can be pulled from a separate file with ``!include``, resolved
relative to the scenario file's directory. This is handy for sharing the
``judge:`` and ``user:`` blocks across scenarios::

    user: !include user_audio.yaml
    judge: !include judge_audio.yaml
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from pipecat.audio.dtmf.types import KeypadEntry


class _ScenarioLoader(yaml.SafeLoader):
    """SafeLoader that resolves only plain-decimal numeric scalars as ints.

    PyYAML's SafeLoader follows YAML 1.1, which reinterprets unquoted numeric
    scalars as octal (``010`` -> 8), hex (``0x10`` -> 16), or binary before
    application code sees them. For DTMF that silently rewrites the digit
    sequence the user typed (``dtmf: 012`` would load as ``10``). Dropping those
    resolvers and keeping only plain decimal means ``dtmf: 123`` still loads as
    an int (so the unquoted-digits convenience works), while leading-zero, hex,
    and binary tokens stay strings and reach DTMF validation with their digits
    intact. No scenario field wants an octal/hex literal, so this is safe
    document-wide.
    """


# Strip the inherited int resolvers (which match octal/hex/binary/sexagesimal)
# and register a decimal-only replacement. Underscores stay allowed to match
# YAML's grouping syntax (e.g. ``1_000``); a leading zero (``012``) no longer
# matches, so such tokens load as strings.
_ScenarioLoader.yaml_implicit_resolvers = {
    ch: [(tag, rx) for tag, rx in resolvers if tag != "tag:yaml.org,2002:int"]
    for ch, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
yaml.add_implicit_resolver(
    "tag:yaml.org,2002:int",
    re.compile(r"^[-+]?(?:0|[1-9][0-9_]*)$"),
    list("-+0123456789"),
    Loader=_ScenarioLoader,
)


def _add_include_constructor(loader_class: type[yaml.SafeLoader], base_dir: Path) -> None:
    """Register an ``!include <relative-path>`` constructor on ``loader_class``.

    Included files load with the same loader class, so nested includes work and
    scalars get the same resolver treatment as the top-level document. Paths
    resolve against ``base_dir`` (the scenario file's directory).
    """

    def _include(loader: yaml.SafeLoader, node: yaml.Node) -> Any:
        if not isinstance(node, yaml.ScalarNode):
            raise yaml.constructor.ConstructorError(
                None, None, "!include expects a file path", node.start_mark
            )
        include_path = base_dir / str(loader.construct_scalar(node))
        with include_path.open() as f:
            return yaml.load(f, loader_class)

    loader_class.add_constructor("!include", _include)


# Events whose payloads carry bot-generated text the judge can sensibly
# evaluate. Asserting ``eval:`` on anything else (user transcripts, tool
# calls, interruption signals) produces a parser warning — the test controls
# user input deterministically, so judging it adds cost without signal.
# ``response`` is the modality-agnostic alias, resolved to one of the others
# after parsing (see _resolve_response_events).
JUDGEABLE_EVENTS = frozenset({"response", "llm_response", "tts_response"})


@dataclass
class EvalFunctionCall:
    """One expected function call within a ``function_call`` expectation.

    Parameters:
        name: The function name to match. ``None`` matches any call (used by a
            bare ``function_call`` expectation that just asserts a call happened).
        args: Optional subset check on the call's arguments (every listed
            key/value must be present; extra arguments are ignored).
    """

    name: str | None = None
    args: dict | None = None


@dataclass
class EvalExpectation:
    """A single expected event in a scenario turn.

    Parameters:
        event: Required — the semantic event name (e.g. ``user_stopped_speaking``).
        within_ms: Optional latency budget, measured from the turn's user send —
            all of a turn's expectations share that one anchor, so a stalled turn
            fails within a single budget rather than one per expectation. For audio
            turns the anchor is when the utterance was *sent*, not when it finishes
            playing out of the transport's virtual mic. Defaults to 60s when
            omitted, so timing isn't asserted unless set explicitly.
        text_contains: Optional substring check on the event's text content
            (``llm_response.text`` or ``user_transcription.transcript``).
        calls: For a ``function_call`` event, the set of calls expected in the
            turn. They are matched by name in any order and the expectation passes
            only when all of them are found. Built from ``calls:`` in the YAML, or
            from the single ``name:``/``args:`` shorthand.
        eval: Optional natural-language criterion the event's text content
            must satisfy. Evaluated by a judge LLM. Only meaningful on the
            bot-generated text events: ``response``, ``llm_response``, and
            ``tts_response``.
        absent: When True, the expectation is inverted: it passes only when NO
            event of this type arrives before the ``within_ms`` budget expires,
            and fails as soon as one does. Matches on event type only;
            ``text_contains``, ``eval``, and ``calls`` are not allowed alongside
            it.
    """

    event: str
    within_ms: int | None = None
    text_contains: str | None = None
    calls: list[EvalFunctionCall] | None = None
    eval: str | None = None
    absent: bool = False


@dataclass
class EvalSendAfter:
    """Scheduling for when a turn's input (``user`` or ``dtmf``) is sent.

    When set on a :class:`EvalTurn`, the harness waits for ``event`` to have been
    seen (either earlier in the run or arriving now), then waits an additional
    ``delay_ms`` before sending the turn's input. Used for barge-in tests:
    ``send_after: {event: llm_started, delay_ms: 500}`` means "interrupt 500ms
    after the bot started responding."

    ``event`` is optional: a bare ``send_after: {delay_ms: 500}`` is a pure time
    delay with no event anchor (500ms after the previous turn's send). Handy for
    pacing keypresses across ``dtmf`` turns, where there is no per-key event to
    anchor on.

    Parameters:
        event: Name of the event to schedule from, or ``None`` for a pure
            ``delay_ms`` time delay with no event anchor.
        delay_ms: Additional delay in milliseconds after the event was received
            (or, when ``event`` is ``None``, after the previous turn's send).
    """

    event: str | None
    delay_ms: int


@dataclass
class EvalTurn:
    """One turn in a scenario.

    A turn drives the bot one of three ways: the harness sends a ``user``
    utterance (the person speaks), it sends a ``dtmf`` keypress sequence (the
    person presses keys), or it is observation-only (neither field — useful for
    bot-first scenarios like opening greetings). ``user`` and ``dtmf`` are
    mutually exclusive: a turn is one or the other.

    Parameters:
        user: Optional text the harness sends as the user's turn — an RTVI
            ``send-text`` in text modality, or synthesized speech (``raw-audio``)
            in audio modality. If absent, the turn just waits for and asserts on
            expected events.
        dtmf: Optional DTMF keypad sequence the harness sends, one
            :class:`~pipecat.frames.frames.InputDTMFFrame` per character (e.g.
            ``"123#"``). Each character must be a valid
            :class:`~pipecat.audio.dtmf.types.KeypadEntry` (``0``-``9``, ``*``,
            ``#``). Mutually exclusive with ``user``. The keys are injected the
            same way regardless of the scenario's user/judge modality; a bot with
            a ``DTMFAggregator`` turns them into a transcription it reacts to.
            Quote the value in YAML (``dtmf: "123#"``) — an unquoted ``#`` starts
            a comment.
        expect: Expected events, in the order they should arrive. Optional —
            omit it for a pure pacing/observation turn (e.g. a ``dtmf`` turn that
            only presses keys, with the assertion on a later turn).
        send_after: Optional schedule for when the turn's input should fire. Only
            meaningful when ``user`` or ``dtmf`` is set.
        image: Optional path to an image to register for this turn (resolved
            relative to the scenario file). When a function-calling-video bot
            requests a user image during the turn, the eval transport serves this
            one. Stays registered until a later turn provides a different image.
    """

    user: str | None
    expect: list[EvalExpectation] = field(default_factory=list)
    dtmf: str | None = None
    send_after: EvalSendAfter | None = None
    image: str | None = None


@dataclass
class EvalScenario:
    """A parsed scenario file.

    Parameters:
        name: The eval name (from ``name:``).
        turns: Ordered list of turns.
        context: LLM messages the bot's context should start from for this eval.
            When non-empty, the harness sends them as an ``eval-context`` client
            message right after the bot-ready handshake (the eval serializer
            turns it into an ``LLMMessagesUpdateFrame``, which replaces the
            context); bots without an LLM context aggregator ignore the frame.
            Omitted or empty (the default): the harness sends nothing and the
            bot keeps the context it set up itself.
        judge: Judge LLM configuration dict with keys ``service``, ``model``,
            and optional ``endpoint``. Defaults to
            ``{"service": "ollama", "model": "gemma2:9b"}``.
        bot_audio: Whether the bot produces speech, derived from
            ``judge.modality``. False (text, the default): the bot skips TTS —
            the harness configures skip-TTS at connect, so even an on-connect
            greeting is silent. True (audio): the bot speaks, and the judge
            evaluates the transcription of its actual audio.
        transcriber: Parsed from the ``judge.transcription:`` block; the STT
            config (``service`` defaults to ``moonshine``, plus ``model``) used to
            transcribe the bot's audio for the ``response`` event (``None`` in
            text modality).
        user_audio: TTS config the harness uses to generate user audio. When
            present, the harness streams RTVI ``raw-audio`` (not ``send-text``)
            to the bot, exercising its STT for real. Mapping with ``service``,
            ``voice``, and optional ``model`` / ``sample_rate`` /
            ``api_key``. Omit for text-only evals (default).
        trigger_disconnect: Whether the harness fires the bot's
            ``on_client_disconnected`` handler when this scenario's connection
            ends. Bots often cancel their pipeline there, so this is False by
            default to avoid that between scenarios; set True to exercise the
            bot's disconnect path. Independent of ``--stop-bot``, which tears the
            bot down via ``eval-cancel`` regardless of the handler.
        source_path: Path the scenario was loaded from, for error messages.
    """

    name: str
    turns: list[EvalTurn]
    context: list[dict] = field(default_factory=list)
    judge: dict = field(default_factory=lambda: {"service": "ollama", "model": "gemma2:9b"})
    bot_audio: bool = False
    transcriber: dict | None = None
    user_audio: dict | None = None
    trigger_disconnect: bool = False
    source_path: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> "EvalScenario":
        """Parse a scenario YAML file into an :class:`EvalScenario`.

        Args:
            path: Path to a YAML file with the scenario schema.

        Returns:
            The parsed scenario.

        Raises:
            ValueError: If the file structure is invalid.
            FileNotFoundError: If the path doesn't exist.
        """
        path = Path(path)

        # Support `judge: !include judge_audio.yaml` (and `user:`, etc.) so
        # scenarios can share judge/user config. Includes resolve relative to the
        # scenario file's directory. Register the constructor on a private loader
        # subclass (not the global SafeLoader) so it has no global side effects.
        class _Loader(_ScenarioLoader):
            pass

        _add_include_constructor(_Loader, path.parent)

        with path.open() as f:
            data = yaml.load(f, _Loader)

        if not isinstance(data, dict):
            raise ValueError(f"{path}: top level must be a mapping")

        name = data.get("name")
        if not name or not isinstance(name, str):
            raise ValueError(f"{path}: missing or invalid 'name:' field")

        raw_turns = data.get("turns")
        if not isinstance(raw_turns, list):
            raise ValueError(f"{path}: 'turns:' must be a list")

        turns = [_parse_turn(t, path, idx) for idx, t in enumerate(raw_turns)]

        raw_context = data.get("context")
        if raw_context is None:
            context: list[dict] = []
        elif isinstance(raw_context, list):
            context = raw_context
        else:
            raise ValueError(f"{path}: 'context:' must be a list of message dicts")

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

        return cls(
            name=name,
            turns=turns,
            context=context,
            judge=judge,
            bot_audio=bot_audio,
            transcriber=transcriber,
            user_audio=user_audio,
            trigger_disconnect=bool(data.get("trigger_disconnect", False)),
            source_path=path,
        )


_DEFAULT_JUDGE = {"service": "ollama", "model": "gemma2:9b"}


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


def _resolve_response_events(turns: list[EvalTurn], bot_audio: bool, path: Path) -> None:
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


# ANSI codes for the colored config summary (applied only when color=True). The
# section labels are bold, the separators dim, and each segment's keyword gets one
# hue per category so modality / service / judge LLM are easy to tell apart; the
# values are left uncolored.
_CFG_LABEL = "1"  # bold — the "user" / "judge" section labels
_CFG_SEP = "2"  # dim — the "->" arrow and "|" separators
_CFG_MODALITY = "33"  # yellow — modality keyword
_CFG_SERVICE = "32"  # green — speech (TTS) / transcription (STT) keywords
_CFG_EVAL = "35"  # magenta — eval keyword (judge LLM)


def describe_config(scenario: EvalScenario, *, color: bool = False) -> str:
    """Two-line summary of a scenario's user + judge config, for pre-run logs.

    Args:
        scenario: The parsed scenario to summarize.
        color: When True, ANSI-color each segment's keyword by category (modality,
            service, judge LLM) so they're easy to tell apart.

    Returns:
        A ``user`` line and a ``judge`` line, each a set of ``key: value`` segments
        separated by ``|``, e.g.::

            user  -> modality: audio | speech: kokoro/af_heart
            judge -> modality: audio | transcription: moonshine/small-streaming | eval: ollama/gemma2:9b
    """

    def paint(text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if color else text

    def seg(key: str, value: str, code: str) -> str:
        return f"{paint(key + ':', code)} {value}"

    arrow = paint(" -> ", _CFG_SEP)
    sep = paint(" | ", _CFG_SEP)

    def svc_model(cfg: dict, default_service: str, model_key: str) -> str:
        service = cfg.get("service", default_service)
        model = cfg.get(model_key)
        return f"{service}/{model}" if model else str(service)

    user_segs = [seg("modality", "audio" if scenario.user_audio else "text", _CFG_MODALITY)]
    if scenario.user_audio:
        # The TTS "voice" is the speech config's model-equivalent.
        user_segs.append(seg("speech", svc_model(scenario.user_audio, "?", "voice"), _CFG_SERVICE))

    eval_svc = f"{scenario.judge.get('service', '?')}/{scenario.judge.get('model', '?')}"
    judge_segs = [seg("modality", "audio" if scenario.bot_audio else "text", _CFG_MODALITY)]
    if scenario.bot_audio:
        judge_segs.append(
            seg(
                "transcription",
                svc_model(scenario.transcriber or {}, "whisper", "model"),
                _CFG_SERVICE,
            )
        )
    judge_segs.append(seg("eval", eval_svc, _CFG_EVAL))

    return (
        f"{paint('user'.ljust(5), _CFG_LABEL)}{arrow}{sep.join(user_segs)}\n"
        f"{paint('judge'.ljust(5), _CFG_LABEL)}{arrow}{sep.join(judge_segs)}"
    )


def _parse_turn(t: Any, path: Path, idx: int) -> EvalTurn:
    """Parse one entry from the ``turns:`` list."""
    if not isinstance(t, dict):
        raise ValueError(f"{path}: turn #{idx} must be a mapping")

    user = t.get("user")
    if user is not None and not isinstance(user, str):
        raise ValueError(f"{path}: turn #{idx} 'user:' must be a string if present")

    dtmf = _parse_dtmf(t.get("dtmf"), path, idx)
    if user is not None and dtmf is not None:
        raise ValueError(
            f"{path}: turn #{idx} has both 'user:' and 'dtmf:' — a turn is one or the other"
        )

    # `expect:` is optional: a turn may just send input (e.g. paced keypresses)
    # or just wait, with the assertion living on another turn.
    raw_expect = t.get("expect", [])
    if not isinstance(raw_expect, list):
        raise ValueError(f"{path}: turn #{idx} 'expect:' must be a list if present")

    expect = [_parse_expectation(e, path, idx, ei) for ei, e in enumerate(raw_expect)]

    send_after = _parse_send_after(t.get("send_after"), path, idx) if "send_after" in t else None
    if send_after is not None and user is None and dtmf is None:
        raise ValueError(
            f"{path}: turn #{idx} has 'send_after:' but no 'user:' or 'dtmf:' — "
            "send_after only schedules when the turn's input gets sent"
        )

    # Image paths resolve relative to the scenario file, so a scenario is portable.
    image = t.get("image")
    if image is not None:
        if not isinstance(image, str):
            raise ValueError(f"{path}: turn #{idx} 'image:' must be a path string")
        image = str((path.parent / image).resolve())

    return EvalTurn(user=user, dtmf=dtmf, expect=expect, send_after=send_after, image=image)


def _parse_dtmf(dtmf: Any, path: Path, turn_idx: int) -> str | None:
    """Parse and validate a turn's ``dtmf:`` keypad sequence."""
    if dtmf is None:
        return None
    # YAML parses an unquoted digit sequence as an int (`dtmf: 123`); normalize so
    # both `dtmf: 123` and `dtmf: "123#"` work the same.
    if isinstance(dtmf, int):
        dtmf = str(dtmf)
    if not isinstance(dtmf, str) or not dtmf:
        raise ValueError(
            f"{path}: turn #{turn_idx} 'dtmf:' must be a non-empty string of keypad entries"
        )
    for ch in dtmf:
        try:
            KeypadEntry(ch)
        except ValueError:
            valid = ", ".join(e.value for e in KeypadEntry)
            raise ValueError(
                f"{path}: turn #{turn_idx} 'dtmf:' has invalid keypad entry {ch!r} "
                f"(valid entries: {valid})"
            )
    return dtmf


def _parse_send_after(s: Any, path: Path, turn_idx: int) -> EvalSendAfter:
    """Parse a ``send_after:`` block."""
    if not isinstance(s, dict):
        raise ValueError(f"{path}: turn #{turn_idx} 'send_after:' must be a mapping")

    event = s.get("event")
    if event is not None and not isinstance(event, str):
        raise ValueError(f"{path}: turn #{turn_idx} 'send_after.event' must be a string if present")

    delay_ms = s.get("delay_ms", 0)
    if not isinstance(delay_ms, int) or delay_ms < 0:
        raise ValueError(
            f"{path}: turn #{turn_idx} 'send_after.delay_ms' must be a non-negative int"
        )

    # With no event to anchor on, a zero delay would be a no-op send_after.
    if event is None and delay_ms == 0:
        raise ValueError(
            f"{path}: turn #{turn_idx} 'send_after:' needs an 'event:' or a positive 'delay_ms:'"
        )

    return EvalSendAfter(event=event, delay_ms=delay_ms)


def _parse_expectation(e: Any, path: Path, turn_idx: int, exp_idx: int) -> EvalExpectation:
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

    absent = e.get("absent", False)
    if not isinstance(absent, bool):
        raise ValueError(
            f"{path}: turn #{turn_idx} expectation #{exp_idx} 'absent:' must be a boolean"
        )
    if absent:
        # An absent expectation matches on event type only: content and call
        # checks describe an event that must arrive, which contradicts absence.
        conflicting = [
            key for key in ("text_contains", "eval", "calls", "name", "args") if key in e
        ]
        if conflicting:
            raise ValueError(
                f"{path}: turn #{turn_idx} expectation #{exp_idx} 'absent: true' "
                f"cannot be combined with {', '.join(repr(k) for k in conflicting)}"
            )

    calls = _parse_function_calls(e, event, path, turn_idx, exp_idx) if not absent else None

    return EvalExpectation(
        event=event,
        within_ms=e.get("within_ms"),
        text_contains=e.get("text_contains"),
        calls=calls,
        eval=criterion,
        absent=absent,
    )


def _parse_function_calls(
    e: dict, event: str, path: Path, turn_idx: int, exp_idx: int
) -> list[EvalFunctionCall] | None:
    """Normalize a ``function_call`` expectation's expected calls into a list.

    Accepts a ``calls:`` list (each entry a bare name string or a ``{name, args}``
    mapping) for the multi-call case, or the single ``name:``/``args:`` shorthand.
    A bare ``function_call`` (neither) becomes one ``EvalFunctionCall(name=None)``
    that matches any single call. Returns None for non-function_call events.
    """
    if event != "function_call":
        return None

    where = f"{path}: turn #{turn_idx} expectation #{exp_idx}"
    raw_calls = e.get("calls")
    if raw_calls is None:
        return [EvalFunctionCall(name=e.get("name"), args=e.get("args"))]

    if not isinstance(raw_calls, list) or not raw_calls:
        raise ValueError(f"{where}: 'calls:' must be a non-empty list")
    out: list[EvalFunctionCall] = []
    for c in raw_calls:
        if isinstance(c, str):
            out.append(EvalFunctionCall(name=c))
        elif isinstance(c, dict):
            out.append(EvalFunctionCall(name=c.get("name"), args=c.get("args")))
        else:
            raise ValueError(f"{where}: each 'calls:' entry must be a name or a mapping")
    return out
