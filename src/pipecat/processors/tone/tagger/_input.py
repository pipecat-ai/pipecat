#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""What Pipecat Tone's tagger model reads: the call so far, as text it was trained on.

For each sentence the model reads a text pair. The first text is the end of the call.
The second holds what the TTS service performs, the policy's presets, how the user
sounded, the tags of the bot's previous reply, state hints from the whole call, the
reply so far and the sentence, last. State hints carry what the end of the call leaves
out.

Training wrote the same text, so it must not change for a released model.
"""

import re
from dataclasses import dataclass

from pipecat.expressive.perception import PerceptionSignal
from pipecat.expressive.policy import ExpressionGuidance, ExpressionPolicy
from pipecat.processors.tone.tagger._labels import LabelSpace, SentenceLabels

# The presets the model reads, by the names it was trained on.
PRESET_NAMES: dict[str, str] = {
    ExpressionGuidance.MATCH_STYLE: "match style",
    ExpressionGuidance.SLOW_FOR_DETAILS: "slow for details",
    ExpressionGuidance.BREATHE: "breathe",
}

# The model was trained on a ``tone`` signal; Pipecat Tone's reader reads an
# ``emotion``, folded in by adding its confidences.
_EMOTION_TO_TONE = {
    "neutral": "neutral",
    "happy": "positive",
    "amused": "positive",
    "surprised": "neutral",
    "sad": "sad",
    "afraid": "sad",
    "angry": "frustrated",
    "disgusted": "frustrated",
    "contemptuous": "frustrated",
}
_CLEAR = 0.7
_LIKELY = 0.5

# Details someone might write down: a code with letters and digits, five or more
# digits (with dashes or spaces), characters spelled out one by one, an email.
_DETAILS = re.compile(
    r"\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{4,}\b"
    r"|(?<![\d$.,])\d(?:[\d\- ]*\d){4,}(?![\d.,]\d)"
    r"|\b(?:[A-Za-z0-9][ -]){3,}[A-Za-z0-9]\b"
    r"|[\w.+-]+@[\w-]+(?:\.[\w-]+)+"
)
_APOLOGY = re.compile(r"\b(sorry|apologi[sz]e)", re.IGNORECASE)
_SLOW_OR_REPEAT = re.compile(
    r"\b(slow(er)? down|slower|say (that|it) again|repeat|didn't (catch|get) (that|it)"
    r"|one more time|come again)\b",
    re.IGNORECASE,
)
_DIGIT_WORDS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}  # fmt: skip


@dataclass
class Message:
    """A turn of the call, as plain words.

    Parameters:
        role: ``"user"`` or ``"assistant"``.
        text: The words.
        heard: How the user sounded, folded into the model's tones, or None.
        speculation: Whether it's the turn an early reply answers, not yet
            confirmed.
    """

    role: str
    text: str
    heard: list[PerceptionSignal] | None = None
    speculation: bool = False


@dataclass
class TurnContext:
    """What the model knows when a reply starts.

    Parameters:
        messages: The call so far.
        space: The labels the call can use.
        policy: The pipeline's expression policy.
        previous: The labels of the bot's previous reply, or None before the
            first.
    """

    messages: list[Message]
    space: LabelSpace
    policy: ExpressionPolicy
    previous: list[SentenceLabels] | None = None

    @property
    def heard(self) -> list[PerceptionSignal] | None:
        """How the user sounded in the turn this reply answers."""
        user = next((m for m in reversed(self.messages) if m.role == "user"), None)
        return user.heard if user else None


@dataclass(frozen=True)
class Layout:
    """What a model reads, as its config records it was trained.

    Parameters:
        context_turns: The messages of the call.
        earlier: The reply's earlier sentences.
        custom_guidance: Whether the policy's own lines are read too.
        state: Whether state hints are read.
        max_tokens: The token budget.
        cut_call_first: Over budget, whether the call's oldest tokens go first.
        previous_tags: Whether the labels of the bot's previous reply are read.
    """

    context_turns: int = 1
    earlier: int = 1
    custom_guidance: bool = False
    state: bool = True
    max_tokens: int = 160
    cut_call_first: bool = False
    previous_tags: bool = False


def fold(heard: list[PerceptionSignal] | None) -> list[PerceptionSignal] | None:
    """Fold an ``emotion`` signal into a ``tone`` signal, leaving the others as they are."""
    if not heard:
        return heard
    folded = []
    for signal in heard:
        if signal.name == "emotion" and signal.labels:
            totals: dict[str, float] = {}
            for label, confidence in signal.labels:
                if tone := _EMOTION_TO_TONE.get(label):
                    totals[tone] = totals.get(tone, 0.0) + confidence
            labels = sorted(
                ((tone, round(min(c, 1.0), 2)) for tone, c in totals.items()), key=lambda p: -p[1]
            )
            folded.append(PerceptionSignal("tone", labels=labels[:2]))
        else:
            folded.append(signal)
    return folded


def encode(
    turn: TurnContext, earlier: list[str], sentence: str, layout: Layout
) -> tuple[str, str, int]:
    """Write one sentence's text pair.

    Args:
        turn: What the model knows of the call.
        earlier: The reply's earlier sentences.
        sentence: The sentence to label.
        layout: What the model reads.

    Returns:
        The two texts, and where the sentence starts in the second.
    """
    lines = []
    for m in turn.messages[-layout.context_turns :] if layout.context_turns else []:
        lines.append(
            f"caller ({_describe(m.heard)}): {m.text}" if m.role == "user" else f"agent: {m.text}"
        )

    guidance = turn.policy.guidance
    presets = [PRESET_NAMES[g] for g in guidance if g in PRESET_NAMES]
    custom = [g for g in guidance if g not in PRESET_NAMES] if layout.custom_guidance else []
    space = turn.space
    header = [
        f"tts {space.tts}",
        "can " + (" ".join(space.deliveries) or "nothing"),
        "sounds " + (" ".join(space.sounds) or "none"),
        f"stress {'yes' if space.emphasis else 'no'}",
        "policy " + (", ".join(presets) or "none") + ("; " + " ".join(custom) if custom else ""),
        f"caller sounded {_describe(turn.heard)}",
    ]
    if layout.previous_tags:
        header.append("last reply " + _last_reply(turn.previous))
    if layout.state:
        header.append("state " + (", ".join(_state_hints(turn.messages, sentence)) or "none"))
    before = " ".join(earlier[-layout.earlier :]) if layout.earlier else ""
    prefix = " | ".join(header) + f" | reply so far: {before or '-'} | sentence: "
    return "\n".join(lines), prefix + sentence, len(prefix)


def _state_hints(messages: list[Message], sentence: str) -> list[str]:
    """Facts from the whole call that the model's short window misses."""
    agent = [m.text for m in messages if m.role != "user"]
    user = [m.text for m in messages if m.role == "user"]
    hints = []
    if any(_APOLOGY.search(text) for text in agent):
        hints.append("apologized before")
    if any(_SLOW_OR_REPEAT.search(text) for text in user):
        hints.append("caller asked to slow down or repeat")
    details = [_normalize(m[0]) for m in _DETAILS.finditer(sentence)]
    if details:
        said = _normalize(" ".join(agent))
        last = _normalize(user[-1]) if user else ""
        if any(d in last for d in details):
            hints.append("caller just read this detail back")
        elif any(d in said for d in details):
            hints.append("detail given before")
        else:
            hints.append("new detail")
    return hints


def _last_reply(labels: list[SentenceLabels] | None) -> str:
    """The deliveries and sounds the previous reply used, in order: "none" before the first."""
    if labels is None:
        return "none"
    used = []
    for sentence in labels:
        used += [
            sentence.sound_before,
            sentence.delivery,
            *(span.delivery for span in sentence.spans),
            sentence.sound_after,
        ]
    return " ".join(dict.fromkeys(u for u in used if u)) or "plain"


def _describe(heard: list[PerceptionSignal] | None) -> str:
    """A reading in bands: ``"tone sad clear, neutral guess; sarcasm low"``."""
    if not heard:
        return "no reading"
    parts = []
    for signal in heard:
        if signal.labels:
            bands = ", ".join(f"{label} {_band(c)}" for label, c in signal.labels)
            parts.append(f"{signal.name} {bands}")
        elif signal.value is not None:
            parts.append(f"{signal.name} {_level(signal.value)}")
    return "; ".join(parts)


def _band(confidence: float) -> str:
    return "clear" if confidence >= _CLEAR else "likely" if confidence >= _LIKELY else "guess"


def _level(value: float) -> str:
    return "high" if value >= _CLEAR else "some" if value >= _LIKELY else "low"


def _normalize(text: str) -> str:
    """Letters and digits only, spoken digits as digits: "seven Q two" is "7q2"."""
    words: list[str] = re.findall(r"[a-z0-9]+", text.lower())
    return "".join(_DIGIT_WORDS.get(w, w) for w in words)
