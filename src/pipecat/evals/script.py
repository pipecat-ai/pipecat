#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Scripted scenario file format for Pipecat behavioral evaluations.

A scenario is a YAML file describing a scripted conversation and the semantic
events expected to flow back from the bot. Simple example::

    name: simple_user_input
    turns:
      - user: "hello world"
        expect:
          - event: user_started_speaking
          - event: user_transcription
            text_contains: "hello world"

The harness plays each turn and checks the events the bot emits back, in
order. A file with a ``persona:`` instead of ``turns:`` is the other kind, a
simulation, where an LLM plays the user (:mod:`pipecat.evals.simulation`).

Event names are the friendly names the harness maps RTVI server messages onto:
``user_started_speaking``, ``user_stopped_speaking``, ``vad_user_started_speaking``,
``vad_user_stopped_speaking``, ``user_transcription``, ``bot_started_speaking``,
``bot_stopped_speaking``, ``llm_started``, ``response``, ``llm_response``,
``tts_response``, ``function_call``, ``function_call_stopped``. The
``vad_*`` events are the raw
VAD signal, useful as a timing anchor when a turn-detection strategy gates or defers the
turn-level ``user_stopped_speaking`` (e.g. filtering incomplete turns).

The bot's reply can be asserted three ways:

``response``
    the transcription of the bot's *actual synthesized audio* (a local STT —
    Moonshine or Whisper — run by the harness) in audio modality, or the LLM
    text in text modality. The real end-to-end check — prefer this.

``llm_response``
    the LLM's text output (``bot-llm-text``). Available in both modalities.

``tts_response``
    the text the TTS reports speaking (``bot-tts-text``, with word timing).
    Audio modality only.

Supported expectation fields (per event):

``event: <name>``
    required — event type name

``within_ms: <int>``
    latency budget from the most recent anchor (optional; defaults to 60s when
    omitted)

``text_contains: <str>``
    substring check on the event's text content, ignoring whitespace differences

``calls:``
    for ``function_call`` — the set of calls the turn should make, matched by
    name in any order; the expectation passes only when all are found::

        - event: function_call
          calls:
            - name: get_current_weather
              args: { location: San Francisco }
            - name: get_restaurant_recommendation

    ``function_call_stopped`` takes the same ``calls:`` shape, and its ``args``
    say how the call ended — which is how a scenario tells work that was stopped
    from work that finished on its own::

        - event: function_call_stopped
          calls:
            - name: write_report
              args: { cancelled: true }

``eval: <str>``
    natural-language criterion the event's text content must satisfy, evaluated
    by a judge LLM (see :mod:`pipecat.evals.judge`).

    On ``function_call`` and ``function_call_stopped`` the criterion is about
    the call instead: each call ``calls:`` (or the ``name:``/``args:``
    shorthand) matches is put to the judge by name and arguments, over the
    conversation so far, which is how a scenario checks what ``args:`` cannot
    match verbatim::

        - event: function_call
          calls: [{name: submit_session_suggestion}]
          eval: "the suggestion is for a session about OpenTelemetry tracing, submitted for Jennifer Smith"

    The judge also sees every call the harness matched as a ``[tool call]``
    line in the bot's reply, so a later ``response`` criterion can check the
    bot's words against what it actually submitted.

``absent: true``
    invert the expectation: assert that NO event of this type arrives before the
    ``within_ms`` budget expires (default 60s — set ``within_ms`` explicitly to
    keep the quiet-window wait short). Matches on event type only, so it cannot
    be combined with ``text_contains``, ``eval:``, or ``calls:``. Used for
    duplicate-output regressions::

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

A turn is sent once the bot has finished speaking, like a caller who waits for
the end of the sentence, so a reply the previous turn was satisfied with early
is never talked over. A turn may also include ``send_after:`` to schedule its
``user``/``dtmf`` send
relative to a prior event (used for interruption / barge-in tests), or
``image:`` (a path, relative to the scenario file) to register an image for the
turn — when a function-calling-video bot requests a user image, the eval
transport serves it. ``send_after`` with only ``delay_ms`` (no ``event``) is a
pure time delay relative to the previous send — handy for pacing keypresses
across ``dtmf`` turns to exercise the aggregator's idle-timeout flush.

In audio modality a turn may name a recording with ``audio:`` (a path, relative
to the scenario file) that is played to the bot in place of synthesizing its
``user`` text: a real caller's voice, or a clip that reproduces a bug. The file
is sent at its own sample rate, so it need not match the bot's input rate, and
``user`` still gives what the recording says, since that is what the judge and
``text_contains`` see as the turn's input::

    turns:
      - user: "What is the capital of Germany?"
        audio: ../assets/capital_question.wav
        expect:
          - event: response
            eval: "the response says the capital of Germany is Berlin"

``expect:`` is optional; omit it for a turn that only sends input or only waits.

Top-level optional fields:

``context:``
    LLM messages the bot's context should start from. When given, the harness
    sends them before driving turns (replacing the bot's context); omit to leave
    the bot's own context untouched.

``stop_on_failure:``
    whether the first failed turn ends the scenario (default true). A failure
    leaves the conversation in an unknown state, so the remaining turns usually
    just burn a timeout each. Set it false for a scenario that scores every turn
    independently — a benchmark that reports a per-turn pass rate needs all of
    its turns driven, not just the ones before the first miss::

        stop_on_failure: false

    Give those turns an explicit ``within_ms``: with the 60s default, a silent
    bot costs one full budget per remaining turn.

``user:``
    how user turns are delivered::

        user:
          modality: audio          # audio | text (default text)
          speech:                  # needed unless every spoken turn has audio:
            service: kokoro        # local TTS that synthesizes the user turns
            voice: af_heart        # voices are language-specific
            language: en           # optional; must match the voice
            sample_rate: 16000     # optional
            # or, for any other TTS: factory: my_evals.voice (a callable
            # taking this mapping and returning a local or HTTP TTSService)

    ``audio`` streams synthesized user audio to the bot (exercising its STT for
    real); ``text`` (the default) sends RTVI ``send-text``. A scenario whose
    spoken turns all name an ``audio:`` recording needs no ``speech:`` block.

``judge:``
    what the judge evaluates, and with which LLM::

        judge:
          modality: audio          # audio | text (default text)
          eval:                    # the judge LLM (default ollama)
            service: ollama
            model: gemma4:12b
            # or, for any other LLM: factory: my_evals.judge (a callable
            # taking this mapping and returning an OpenAI-compatible service)
          transcription:           # required when modality is audio
            service: moonshine     # STT for the bot's audio (or whisper, or a factory)
            model: small-streaming # optional
            language: en           # optional; the language the bot speaks
            padding_secs: 0        # optional; silence padded around the
                                   # segment (default: 2)

    ``audio`` makes the bot speak and judges the transcription of its actual
    audio (``tts_response``); ``text`` (the default) skips TTS and judges the
    LLM text (``llm_response``), which is faster and silent.

Any value can be pulled from a separate file with ``!include``, resolved
relative to the scenario file's directory. This is handy for sharing the
``judge:`` and ``user:`` blocks across scenarios::

    user: !include user_audio.yaml
    judge: !include judge_audio.yaml
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.evals.scenario_config import _DEFAULT_JUDGE, _parse_judge_block, _parse_user_block
from pipecat.evals.scenario_loader import _load_mapping
from pipecat.utils.deprecation import deprecated

# Events whose payloads carry bot-generated text the judge can sensibly
# evaluate. Asserting ``eval:`` on anything else but a function call (user
# transcripts, interruption signals) produces a parser warning — the test
# controls user input deterministically, so judging it adds cost without
# signal. ``response`` is the modality-agnostic alias, resolved to one of the
# others after parsing (see _resolve_response_events).
JUDGEABLE_EVENTS = frozenset({"response", "llm_response", "tts_response"})

# Events carrying a function call, matched by name and arguments rather than by
# text: ``function_call`` when one starts, ``function_call_stopped`` when it ends
# (its ``args`` say whether it was cancelled or ran to completion).
FUNCTION_CALL_EVENTS = ("function_call", "function_call_stopped")


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

    @property
    def signature(self) -> str:
        """A short label for the call: ``name(arg=value, ...)``."""
        name = self.name or "any function"
        if not self.args:
            return name
        args = ", ".join(f"{k}={v!r}" for k, v in self.args.items())
        return f"{name}({args})"


@dataclass
class EvalExpectation:
    """A single expected event in a scenario turn.

    Parameters:
        event: Required — the semantic event name (e.g. ``user_stopped_speaking``).
        within_ms: Optional latency budget, measured from the turn's user send —
            all of a turn's expectations share that one anchor, so a stalled turn
            fails within a single budget rather than one per expectation. For audio
            turns the anchor is when the utterance was *sent*, not when it finishes
            streaming to the bot. Defaults to 60s when omitted, so timing isn't
            asserted unless set explicitly.
        text_contains: Optional substring check on the event's text content
            (``llm_response.text`` or ``user_transcription.transcript``).
        calls: For a ``function_call`` event, the set of calls expected in the
            turn. They are matched by name in any order and the expectation passes
            only when all of them are found. Built from ``calls:`` in the YAML, or
            from the single ``name:``/``args:`` shorthand.
        eval: Optional natural-language criterion the event's text content
            must satisfy. Evaluated by a judge LLM. Meaningful on the
            bot-generated text events (``response``, ``llm_response``, and
            ``tts_response``) and on the function-call events, where it is
            about each matched call's name and arguments rather than text.
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

    @property
    def aggregates(self) -> bool:
        """Whether the check accumulates text across events rather than matching one.

        A reply with a content check accumulates the bot's segments; a
        ``user_transcription`` with ``text_contains`` accumulates an STT's pieces.
        """
        if self.event in ("response", "llm_response", "tts_response"):
            return self.text_contains is not None or self.eval is not None
        if self.event == "user_transcription":
            return self.text_contains is not None and self.eval is None
        return False


@dataclass
class EvalSendAfter:
    """Scheduling for when a turn's input (``user`` or ``dtmf``) is sent.

    When set on a :class:`EvalScriptTurn`, the harness waits for ``event`` to have been
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
class EvalScriptTurn:
    """One turn in a scenario.

    A turn drives the bot one of three ways: the harness sends a ``user``
    utterance (the person speaks), it sends a ``dtmf`` keypress sequence (the
    person presses keys), or it is observation-only (neither field — useful for
    bot-first scenarios like opening greetings). ``user`` and ``dtmf`` are
    mutually exclusive: a turn is one or the other.

    A ``user`` turn is spoken by the harness's TTS unless the turn names an
    ``audio`` file to play instead.

    Parameters:
        user: Optional text the harness sends as the user's turn — an RTVI
            ``send-text`` in text modality, or synthesized speech (``raw-audio``)
            in audio modality. If absent, the turn just waits for and asserts on
            expected events.
        audio: Optional path to an audio file (resolved relative to the scenario
            file) played as this turn instead of synthesizing ``user``. Any
            format ``soundfile`` reads works (WAV, MP3, FLAC, OGG, ...);
            multi-channel audio is downmixed to mono. ``user`` is required
            alongside it and is what the recording says — the judge reads it as
            the turn's input and ``text_contains`` matches against it, neither of
            which can be recovered from the audio. Requires
            ``user.modality: audio``, and is mutually exclusive with ``dtmf``.
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
    audio: str | None = None
    dtmf: str | None = None
    send_after: EvalSendAfter | None = None
    image: str | None = None


@deprecated(
    "`EvalTurn` is deprecated since 1.9.0 and will be removed in 2.0.0. "
    "Use `EvalScriptTurn` instead."
)
@dataclass
class EvalTurn(EvalScriptTurn):
    """Deprecated alias for :class:`EvalScriptTurn`.

    .. deprecated:: 1.9.0
        Use :class:`EvalScriptTurn` instead. Will be removed in 2.0.0.
    """


@dataclass
class EvalScriptScenario:
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
            optional ``endpoint``, and an optional ``extra`` mapping forwarded to
            the model as top-level request parameters. Defaults to
            ``{"service": "ollama", "model": "gemma4:12b",
            "extra": {"reasoning_effort": "none"}}``.
        bot_audio: Whether the bot produces speech, derived from
            ``judge.modality``. False (text, the default): the bot skips TTS —
            the harness configures skip-TTS at connect, so even an on-connect
            greeting is silent. True (audio): the bot speaks, and the judge
            evaluates the transcription of its actual audio.
        transcriber: Parsed from the ``judge.transcription:`` block; the STT
            config (``service`` defaults to ``moonshine``, plus ``model`` and an
            optional ``language`` code) used to transcribe the bot's audio for the
            ``response`` event (``None`` in text modality). Set ``language`` when
            the bot speaks a non-English language so the STT doesn't default to
            English.
        user_audio: Whether the user's turns reach the bot as speech, derived
            from ``user.modality``. False (text, the default): each turn is sent
            as an RTVI ``send-text``. True (audio): the harness streams RTVI
            ``raw-audio``, exercising the bot's STT for real.
        user_speech: Parsed from the ``user.speech:`` block; the TTS config the
            harness synthesizes user turns with (``None`` in text modality).
            Mapping with ``service``, ``voice``, and optional ``model`` /
            ``language`` / ``sample_rate`` / ``api_key``. Set ``language`` (a
            code like ``zh``) to synthesize non-English user turns.
        trigger_disconnect: Whether the harness fires the bot's
            ``on_client_disconnected`` handler when this scenario's connection
            ends. Bots often cancel their pipeline there, so this is False by
            default to avoid that between scenarios; set True to exercise the
            bot's disconnect path. Independent of ``--stop-bot``, which tears the
            bot down via ``eval-cancel`` regardless of the handler.
        stop_on_failure: Whether the first failed turn ends the scenario
            (default True). A failed turn leaves the conversation in an unknown
            state, so continuing usually costs one timeout per remaining turn.
            Set False for a scenario whose turns are scored independently, where
            the turns after a failure are still worth driving; each turn's
            outcome is reported in
            :attr:`~pipecat.evals.results.EvalScriptResult.turns`. This governs
            turn-to-turn progression only: within a turn, an expectation that
            times out still ends that turn's matching, because a turn's
            expectations share one deadline anchored at the send.
        source_path: Path the scenario was loaded from, for error messages.
    """

    name: str
    turns: list[EvalScriptTurn]
    context: list[dict] = field(default_factory=list)
    judge: dict = field(default_factory=lambda: dict(_DEFAULT_JUDGE))
    bot_audio: bool = False
    transcriber: dict | None = None
    user_audio: bool = False
    user_speech: dict | None = None
    trigger_disconnect: bool = False
    stop_on_failure: bool = True
    source_path: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> "EvalScriptScenario":
        """Parse a scenario YAML file into an :class:`EvalScriptScenario`.

        Args:
            path: Path to a YAML file with the scenario schema.

        Returns:
            The parsed scenario.

        Raises:
            ValueError: If the file structure is invalid.
            FileNotFoundError: If the path doesn't exist.
        """
        path = Path(path)
        data = _load_mapping(path)

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
        # turn via TTS (exercising the bot's STT); text sends it as text.
        user_audio, user_speech = _parse_user_block(data.get("user"), path)
        _check_user_audio(turns, user_audio, user_speech, path)

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
            user_speech=user_speech,
            trigger_disconnect=bool(data.get("trigger_disconnect", False)),
            stop_on_failure=bool(data.get("stop_on_failure", True)),
            source_path=path,
        )

    def wants_response(self) -> bool:
        """Whether any expectation asserts on the transcription of the bot's audio."""
        return any(exp.event == "response" for turn in self.turns for exp in turn.expect)

    def required_report_level(self) -> str | None:
        """The function-call report level the scenario's assertions need: ``full`` for args, ``name`` for names, else ``None``.

        A judged scenario that asserts on calls needs ``full`` too: the judge
        reads the calls the harness matches, with their arguments, whether the
        ``eval:`` is on the call itself or on the reply after it.
        """
        needs_name = False
        judged = any(exp.eval is not None for turn in self.turns for exp in turn.expect)
        for turn in self.turns:
            for exp in turn.expect:
                if exp.event not in FUNCTION_CALL_EVENTS:
                    continue
                if judged:
                    return "full"
                # name/args live in exp.calls (the parser normalizes the single
                # name:/args: shorthand into it too).
                for call in exp.calls or []:
                    if call.args is not None:
                        return "full"
                    if call.name is not None:
                        needs_name = True
        return "name" if needs_name else None

    def needs_vad_events(self) -> bool:
        """Whether the scenario uses the raw VAD speaking events, which the bot emits only on request."""
        vad_events = {"vad_user_started_speaking", "vad_user_stopped_speaking"}
        for turn in self.turns:
            if turn.send_after is not None and turn.send_after.event in vad_events:
                return True
            if any(exp.event in vad_events for exp in turn.expect):
                return True
        return False


@deprecated(
    "`EvalScenario` is deprecated since 1.9.0 and will be removed in 2.0.0. "
    "Use `EvalScriptScenario` instead."
)
@dataclass
class EvalScenario(EvalScriptScenario):
    """Deprecated alias for :class:`EvalScriptScenario`.

    .. deprecated:: 1.9.0
        Use :class:`EvalScriptScenario` instead. Will be removed in 2.0.0.
    """


def _check_user_audio(
    turns: list[EvalScriptTurn], user_audio: bool, speech: dict | None, path: Path
) -> None:
    """Check the turns against the ``user:`` block they are delivered by."""
    for idx, turn in enumerate(turns):
        if turn.audio is not None and not user_audio:
            raise ValueError(
                f"{path}: turn #{idx} has 'audio:' but the scenario is text modality — "
                "set 'user.modality: audio' to play it to the bot"
            )

    # A turn naming a file is played as-is; only the rest need a voice to speak them.
    if user_audio and speech is None:
        spoken = [i for i, t in enumerate(turns) if t.user is not None and t.audio is None]
        if spoken:
            raise ValueError(
                f"{path}: 'user.modality: audio' requires a 'user.speech:' block "
                f"(TTS service + voice) to synthesize turn(s) {spoken} — "
                "or give each of them an 'audio:' file"
            )


def _resolve_response_events(turns: list[EvalScriptTurn], bot_audio: bool, path: Path) -> None:
    """Resolve ``response`` for the modality and check consistency.

    In audio modality ``response`` is the transcription of the bot's audio; in
    text modality it becomes ``llm_response``. ``tts_response`` needs the bot
    to speak, so it is an error in text modality.
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


def _parse_turn(t: Any, path: Path, idx: int) -> EvalScriptTurn:
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

    # Audio paths resolve relative to the scenario file, so a scenario is portable.
    audio = t.get("audio")
    if audio is not None:
        if not isinstance(audio, str):
            raise ValueError(f"{path}: turn #{idx} 'audio:' must be a path string")
        if dtmf is not None:
            raise ValueError(
                f"{path}: turn #{idx} has both 'audio:' and 'dtmf:' — a turn is one or the other"
            )
        if user is None:
            raise ValueError(
                f"{path}: turn #{idx} has 'audio:' but no 'user:' — give the text the "
                "recording says, so the judge and 'text_contains' have the turn's input"
            )
        audio = str((path.parent / audio).resolve())

    # Image paths resolve relative to the scenario file, so a scenario is portable.
    image = t.get("image")
    if image is not None:
        if not isinstance(image, str):
            raise ValueError(f"{path}: turn #{idx} 'image:' must be a path string")
        image = str((path.parent / image).resolve())

    return EvalScriptTurn(
        user=user, dtmf=dtmf, expect=expect, send_after=send_after, image=image, audio=audio
    )


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
    if (
        criterion is not None
        and event not in JUDGEABLE_EVENTS
        and event not in FUNCTION_CALL_EVENTS
    ):
        logger.warning(
            f"{path}: turn #{turn_idx} expectation #{exp_idx}: 'eval:' on "
            f"event {event!r} — judge only makes sense on bot-generated text "
            f"events ({', '.join(sorted(JUDGEABLE_EVENTS))}) and function calls "
            f"({', '.join(FUNCTION_CALL_EVENTS)}). Will run but is unlikely to be "
            "meaningful."
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
    """Normalize a ``function_call`` expectation's calls into a list.

    A ``calls:`` list (names, or ``{name, args}`` mappings), the single
    ``name:``/``args:`` shorthand, or nothing, which matches any one call.
    ``None`` for other events.
    """
    if event not in FUNCTION_CALL_EVENTS:
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
