#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A line of a group conversation, and the inputs Pipecat Sidechat's models read for it.

A model reads two things for a line. One is text: the agent's card, the
earlier turns tagged by who said them, and the line itself. The base model
reads them as one text, each turn tagged relative to the line's speaker::

    agent: Atlas | also: at last | AI assistant on the call
    [other1] Morning.
    [speaker] Atlas, how far is Leeds from York?
    [agent] About twenty-five miles.
    >> [speaker] And by train?

The small model reads the earlier turns and the line apart, so the turns can
be encoded before the line is said. Their people are numbered in the order
they first spoke, and the line carries its speaker's number::

    agent: Atlas | also: at last | AI assistant on the call
    [p1] Morning.
    [p2] Atlas, how far is Leeds from York?
    [agent] About twenty-five miles.

    agent: Atlas | also: at last | AI assistant on the call
    >> [p2] And by train?

The other is a short vector of numbers: the rhythm of those turns (whether
the agent spoke last, whether the speaker is the person it was answering) and
a few cues in the line's words. All are built exactly as the models were
trained on them, so this module changes only with a model's input version.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

#: The versions of the inputs built here, for a model that reads the turns
#: and the line as one text and for one that reads them apart. A model release
#: names the version it reads.
INPUT_VERSION = "sidechat-input-0.2"
TWO_TOWER_INPUT_VERSION = "sidechat-tt-0.1"

#: The most earlier turns the model reads before a line.
MAX_HISTORY = 8

_LINE_MARK = ">> "
_DEVICE_WORDS = ("siri", "ok google", "hey google", "alexa", "cortana", "hey tv", "ok car")
_ROLE_RE = re.compile(r"\b(assistant|ai|bot)\b", re.IGNORECASE)
_OPEN_RE = re.compile(
    r"\b(anyone|anybody|someone|somebody|everyone|everybody|you guys|you all|y'all)\b",
    re.IGNORECASE,
)
_YOU_RE = re.compile(r"\byou(r|rs|'re|'ll|'ve|'d)?\b", re.IGNORECASE)
_VOCATIVE_RE = re.compile(r"^\W*(?:hey |ok |okay |oh |so |right |well )?([A-Z][a-z]+)\s*[,!?]")
_TRAILING_VOCATIVE_RE = re.compile(r",\s*([A-Z][a-z]+)\s*[?.!]*\s*$")


@dataclass(frozen=True)
class SidechatAgent:
    """The agent, as people in the conversation call it.

    Parameters:
        name: The agent's name, as people say it. Matched with its case.
        aliases: Other ways its name comes out of speech recognition, such as
            ``"at last"`` for ``"Atlas"``. Matched without case.
        role: One line on what the agent is, such as ``"AI assistant on the call"``.
    """

    name: str
    aliases: tuple[str, ...] = ()
    role: str | None = None


@dataclass(frozen=True)
class SidechatTurn:
    """One earlier turn in the conversation.

    Parameters:
        text: What was said.
        speaker: Who said it: a name or a diarization label, or None when
            the transcript doesn't say. Only compared with other speakers.
        agent: Whether the agent said it.
    """

    text: str
    speaker: str | None = None
    agent: bool = False


@dataclass(frozen=True)
class SidechatLine:
    """A line to judge, with the turns before it.

    Parameters:
        text: The line.
        speaker: Who said it, compared with each earlier turn's speaker, or
            None when the transcript doesn't say.
        history: The turns before it, oldest first. The model reads the last
            :data:`MAX_HISTORY`.
        people: The names of the people present, when known. A name of one
            of them in the line is a sign it is for them.
        over_agent: True when the line was said while the agent was speaking.
            Leave it None otherwise: the model was trained with no value there.
    """

    text: str
    speaker: str | None = None
    history: Sequence[SidechatTurn] = ()
    people: Sequence[str] | None = None
    over_agent: bool | None = None


@dataclass(frozen=True)
class _Rhythm:
    """The turns before a line, as numbers."""

    turns: int
    agent_turns: int
    # 0 when the agent spoke just before the line, None when it hasn't spoken.
    turns_since_agent: int | None
    agent_last_asked: bool
    # Whether the agent's latest turn followed this speaker's, None when unknown.
    speaker_was_agent_partner: bool | None
    speaker_spoke_last: bool
    speaker_turns: int
    others: int


def _history(line: SidechatLine) -> list[SidechatTurn]:
    return list(line.history)[-MAX_HISTORY:]


def _roles(line: SidechatLine, *, slots: bool = False) -> list[str]:
    """Each earlier turn's speaker relative to the line's, oldest first.

    ``agent``, ``speaker`` (whoever says the line), ``other`` or, with
    ``slots``, ``other1``, ``other2``... in order of appearance, and
    ``unknown`` when either speaker isn't known.
    """
    order: dict[str, int] = {}
    roles = []
    for turn in _history(line):
        if turn.agent:
            roles.append("agent")
        elif turn.speaker is None or line.speaker is None:
            roles.append("unknown")
        elif turn.speaker == line.speaker:
            roles.append("speaker")
        elif slots:
            order.setdefault(turn.speaker, len(order) + 1)
            roles.append(f"other{order[turn.speaker]}")
        else:
            roles.append("other")
    return roles


def _rhythm(line: SidechatLine) -> _Rhythm:
    history = _history(line)
    roles = _roles(line)
    last_agent = max((i for i, role in enumerate(roles) if role == "agent"), default=None)
    partner = None
    if last_agent is not None and line.speaker is not None:
        before = [role for role in roles[:last_agent] if role != "agent"]
        partner = (before[-1] == "speaker") if before and before[-1] != "unknown" else None
    return _Rhythm(
        turns=len(roles),
        agent_turns=roles.count("agent"),
        turns_since_agent=None if last_agent is None else len(roles) - 1 - last_agent,
        agent_last_asked=bool(
            last_agent is not None and history[last_agent].text.rstrip().endswith("?")
        ),
        speaker_was_agent_partner=partner,
        speaker_spoke_last=bool(roles and roles[-1] == "speaker"),
        speaker_turns=roles.count("speaker"),
        others=len(
            {turn.speaker for turn, role in zip(history, roles) if role == "other"},
        ),
    )


def _card(agent: SidechatAgent) -> str:
    card = f"agent: {agent.name}"
    if agent.aliases:
        card += f" | also: {', '.join(agent.aliases)}"
    if agent.role:
        card += f" | {agent.role}"
    return card


def encoder_text(agent: SidechatAgent, line: SidechatLine) -> tuple[str, int]:
    """The base model's text for a line, and where the line starts in it.

    Args:
        agent: The agent.
        line: The line, with the turns before it.

    Returns:
        The text, and the character offset at which the line itself starts.
    """
    card = _card(agent)
    parts = [
        f"[{role}] {turn.text}" for role, turn in zip(_roles(line, slots=True), _history(line))
    ]
    who = "speaker" if line.speaker is not None else "unknown"
    over = " (over the agent)" if line.over_agent else ""
    parts.append(f"{_LINE_MARK}[{who}]{over} {line.text}")
    text = card + "\n" + "\n".join(parts)
    return text, text.rfind(_LINE_MARK)


def _people(history: Sequence[SidechatTurn]) -> dict[str, int]:
    """Each person in the turns, numbered in the order they first spoke."""
    order: dict[str, int] = {}
    for turn in history:
        if not turn.agent and turn.speaker is not None:
            order.setdefault(turn.speaker, len(order) + 1)
    return order


def _turns_text(agent: SidechatAgent, history: Sequence[SidechatTurn]) -> str:
    order = _people(history)
    parts = []
    for turn in history:
        tag = (
            "agent"
            if turn.agent
            else "unknown"
            if turn.speaker is None
            else f"p{order[turn.speaker]}"
        )
        parts.append(f"[{tag}] {turn.text}")
    return _card(agent) + "\n" + ("\n".join(parts) if parts else "(start of the conversation)")


def turns_text(
    agent: SidechatAgent, line: SidechatLine, max_tokens: int, count: Callable[[str], int]
) -> str:
    """The small model's text for the turns before a line.

    It doesn't depend on the line or on who says it, so it can be built, and
    encoded, as soon as the turn before the line ends. The oldest turns are
    left out until the text fits.

    Args:
        agent: The agent.
        line: The line, of which only the turns before it are read.
        max_tokens: The most tokens the model reads of the turns.
        count: Counts a text's tokens.

    Returns:
        The text.
    """
    history = _history(line)
    while True:
        text = _turns_text(agent, history)
        if not history or count(text) <= max_tokens:
            return text
        history = history[1:]


def line_text(agent: SidechatAgent, line: SidechatLine) -> str:
    """The small model's text for a line, tagged with its speaker's number in the turns.

    Args:
        agent: The agent.
        line: The line, with the turns before it.

    Returns:
        The text.
    """
    order = _people(_history(line))
    if line.speaker is None:
        tag = "unknown"
    else:
        tag = f"p{order[line.speaker]}" if line.speaker in order else "new"
    over = " (over the agent)" if line.over_agent else ""
    return f"{_card(agent)}\n{_LINE_MARK}[{tag}]{over} {line.text}"


def _calls(agent: SidechatAgent, text: str) -> tuple[bool, bool, bool]:
    """Whether the line names the agent, names it first and names it last."""
    patterns = [re.compile(rf"(?<!\w){re.escape(agent.name)}(?!\w)")] + [
        re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", re.IGNORECASE) for alias in agent.aliases
    ]
    hits = [match for pattern in patterns for match in pattern.finditer(text)]
    if not hits:
        return False, False, False
    first = min(match.start() for match in hits) <= 12
    last = max(match.end() for match in hits) >= len(text.rstrip(" ?.!,")) - 2
    return True, first, last


def numbers(agent: SidechatAgent, line: SidechatLine) -> list[float]:
    """The model's numeric inputs for a line: the turns' rhythm and cues in its words.

    Args:
        agent: The agent.
        line: The line, with the turns before it.

    Returns:
        The numbers, in the order the model reads them.
    """
    rhythm = _rhythm(line)
    text = line.text
    lower = text.lower()
    since = rhythm.turns_since_agent
    out = [float(since == k) for k in (0, 1, 2)]
    out += [float(since is not None and since >= 3), float(since is None)]
    out += [float(rhythm.agent_last_asked)]
    partner = rhythm.speaker_was_agent_partner
    out += [float(partner is True), float(partner is False), float(partner is None)]
    out += [
        float(rhythm.speaker_spoke_last),
        min(rhythm.speaker_turns, 8) / 8,
        min(rhythm.agent_turns, 8) / 8,
        min(rhythm.others, 5) / 5,
        min(rhythm.turns, 8) / 8,
    ]
    over = line.over_agent
    out += [float(over is True), float(over is False), float(over is None)]
    out += [float(line.speaker is not None), float(bool(line.people))]
    out += [float(called) for called in _calls(agent, text)]
    out += [
        float(bool(_ROLE_RE.search(text))),
        float(any(word in lower for word in _DEVICE_WORDS)),
        float("?" in text),
    ]
    out += [
        float(bool(_YOU_RE.search(text))),
        float(bool(_OPEN_RE.search(text))),
        min(len(text.split()), 30) / 30,
    ]
    names = [name for name in (line.people or []) if name and name != line.speaker]
    out += [float(any(re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text) for name in names))]
    match = _VOCATIVE_RE.match(text)
    out += [float(bool(match) and match.group(1) != agent.name)]
    match = _TRAILING_VOCATIVE_RE.search(text)
    out += [float(bool(match) and match.group(1) != agent.name)]
    return out


def mentions(agent: SidechatAgent, text: str) -> bool:
    """Whether the text names the agent, by its name or an alias.

    Args:
        agent: The agent.
        text: The text.

    Returns:
        Whether the agent is named in it.
    """
    return _calls(agent, text)[0]
