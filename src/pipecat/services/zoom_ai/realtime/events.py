#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Session configuration models for the Zoom Voice Agent API.

The Voice Agent's events follow the OpenAI Realtime protocol, so server events
are parsed with :mod:`pipecat.services.openai.realtime.events`. Its session
configuration is Zoom's own, grouped by stage of the voice loop:

- :class:`Listen`: the caller's audio, its format and transcription.
- :class:`Think`: the model, its instructions and tools, and the questions it
  may hand to the Decisions API (:class:`Decisions`).
- :class:`Speak`: the agent's audio, its format and voice.

Only the fields a caller sets are sent; anything left unset takes the server's
default. Turn detection, noise reduction, barge-in and history trimming are
built in and not configurable.

See https://developers.zoom.us/docs/ai-services/voice-agent/live-mode/
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _ZoomModel(BaseModel):
    """Base for the session models: a ``type`` discriminator is always sent."""

    def model_post_init(self, context: Any) -> None:
        """Count ``type`` as set, so dumping with ``exclude_unset`` keeps it."""
        if "type" in type(self).model_fields:
            self.__pydantic_fields_set__.add("type")


class AudioFormat(_ZoomModel):
    """Audio encoding for one direction of the session.

    Parameters:
        type: ``"pcm16"``: mono, signed 16-bit little-endian PCM.
        rate: Sample rate in Hz.
    """

    type: Literal["pcm16"] = "pcm16"
    rate: int | None = None


class Vocabulary(_ZoomModel):
    """Words and phrases that customize recognition.

    Parameters:
        phrases: Names and terms to recognize, such as product names.
    """

    phrases: list[str]


class Transcription(_ZoomModel):
    """Transcription of the caller's audio.

    Parameters:
        language: Language of the caller, for example ``"en"``.
        vocabulary: Words and phrases that customize recognition.
    """

    language: str | None = None
    vocabulary: Vocabulary | None = None


class Listen(_ZoomModel):
    """How the agent listens to the caller.

    Parameters:
        format: Input audio encoding. The service sets it from the pipeline's
            input sample rate.
        transcription: Speech recognition settings.
    """

    format: AudioFormat | None = None
    transcription: Transcription | None = None


class Provider(_ZoomModel):
    """The model behind the agent.

    Parameters:
        type: ``"zoom"`` for a Zoom-hosted model, or ``"openai"`` / ``"anthropic"``
            to use your own key for that provider's model.
        model: Model ID, for example ``"gemma-4"``.
        headers: Headers for the provider, such as ``{"X-Api-Key": "..."}`` with
            your own provider key. Required for ``"openai"`` and
            ``"anthropic"``. Left out of the model's ``repr``, so the key isn't
            logged.
        metadata: The provider model's limits, such as ``context_window`` and
            ``max_output_tokens``.
    """

    type: Literal["zoom", "openai", "anthropic"] = "zoom"
    model: str | None = None
    headers: dict[str, str] | None = Field(default=None, repr=False)
    metadata: dict[str, Any] | None = None


class DecisionChoice(BaseModel):
    """One possible answer to a ``choice`` question.

    Parameters:
        value: The answer, a non-empty string. For yes/no, use ``"yes"`` and
            ``"no"``, or a ``predicate`` question.
        description: Helps tell similar choices apart.
    """

    value: str
    description: str | None = None


class DecisionLevel(BaseModel):
    """One level of a ``score`` question, listed lowest first.

    Parameters:
        label: The level's name.
        description: Helps tell similar levels apart.
    """

    label: str
    description: str | None = None


class DecisionQuestion(BaseModel):
    """A question the Decisions API answers about the conversation.

    Parameters:
        type: ``"choice"`` (which of ``choices`` applies), ``"predicate"`` (how
            likely a condition is true) or ``"score"`` (where on ``levels`` it
            falls).
        name: Names the answer; distinct within its question list.
        instructions: What to decide. The conversation is the evidence.
        choices: The possible answers, for a ``choice`` question.
        levels: The ordered levels, lowest first, for a ``score`` question.
    """

    type: Literal["choice", "predicate", "score"]
    name: str
    instructions: str
    choices: list[DecisionChoice] | None = None
    levels: list[DecisionLevel] | None = None


class Delegation(_ZoomModel):
    """A decision the agent may hand to the Decisions API during a turn.

    The agent decides when to delegate from the session instructions and
    ``description``, the way it decides to call a tool. The Voice Agent then asks
    the Decisions API ``questions`` about the conversation, and the turn ends
    once the decision arrives: the agent doesn't continue on its own. Your
    application acts on the answers, adds what happened to the conversation and
    asks for the next reply (see
    :class:`~pipecat.services.zoom_ai.realtime.llm.ZoomAIRealtimeLLMService`).

    Parameters:
        type: Always ``"decisions"``.
        name: The name the agent knows the delegation by, which the instructions
            can refer to. It may not be the name of a tool.
        description: Together with the instructions, tells the agent when to
            delegate.
        questions: What the Decisions API answers.
    """

    type: Literal["decisions"] = "decisions"
    name: str | None = None
    description: str | None = None
    questions: list[DecisionQuestion]


class DecisionTrigger(_ZoomModel):
    """Questions the Decisions API answers after each conversation event of a kind.

    Triggers run apart from the conversation: the agent never waits for them or
    reads their answers, and their results can arrive late and out of order.

    Parameters:
        type: Always ``"decisions"``.
        id: Returned with each result; defaults to ``trigger_<index>``.
        event_type: ``"conversation.item.input_audio_transcription.completed"``
            (after each caller transcript) or
            ``"response.output_audio_transcript.done"`` (after each spoken reply
            that finished uninterrupted).
        questions: What the Decisions API answers, from the latest lines of the
            conversation.
    """

    type: Literal["decisions"] = "decisions"
    id: str | None = None
    event_type: Literal[
        "conversation.item.input_audio_transcription.completed",
        "response.output_audio_transcript.done",
    ]
    questions: list[DecisionQuestion]


class Decisions(BaseModel):
    """The Decisions API questions a session uses.

    Parameters:
        delegation: A decision the agent may hand off during a turn, or ``None``
            to remove one set earlier.
        triggers: Questions asked after conversation events.
    """

    delegation: Delegation | None = None
    triggers: list[DecisionTrigger] | None = None


class Think(_ZoomModel):
    """How the agent decides what to say.

    Parameters:
        provider: The model behind the agent.
        instructions: System instructions.
        tools: Function tools the agent may call: ``{"type": "function", "name",
            "description", "parameters"}``.
        tool_choice: ``"auto"``, ``"none"`` or ``"required"``.
        response_token_limit: Output token cap per response, or ``"inf"``.
        reasoning: Reasoning configuration, for example ``{"effort": "low"}``.
        decisions: Decisions API questions: a delegation, triggers or both.
    """

    provider: Provider | None = None
    instructions: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_token_limit: int | Literal["inf"] | None = None
    reasoning: dict[str, Any] | None = None
    decisions: Decisions | None = None


class Speak(_ZoomModel):
    """How the agent speaks.

    Parameters:
        format: Output audio encoding. The Voice Agent speaks 16 kHz PCM16.
        voice: Voice ID, for example ``"jenna"`` (the default). See the Voice
            Agent docs for the list.
    """

    format: AudioFormat | None = None
    voice: str | None = None


class SessionProperties(_ZoomModel):
    """A Voice Agent session configuration (the ``session`` of ``session.update``).

    Parameters:
        listen: Input audio settings.
        think: Model settings.
        speak: Output audio settings, or ``None`` for text-only replies.
    """

    listen: Listen | None = None
    think: Think | None = None
    speak: Speak | None = None


class SessionUpdateEvent(_ZoomModel):
    """The ``session.update`` client event.

    Parameters:
        type: Always ``"session.update"``.
        session: The session configuration to apply.
    """

    type: Literal["session.update"] = "session.update"
    session: SessionProperties

    def to_wire(self) -> dict[str, Any]:
        """The event as sent, with unset fields omitted."""
        return self.model_dump(exclude_unset=True)


# ---------------------------------------------------------------------------
# Decisions results (server events)
# ---------------------------------------------------------------------------


class DecisionAnswer(BaseModel):
    """The answer to one question. Check ``type`` before reading the other fields.

    Parameters:
        type: ``"choice"``, ``"predicate"``, ``"score"``, or ``"refusal"`` when
            the Decisions API declined the question.
        name: The question's name.
        choice: The chosen value, for a ``choice`` question.
        probability: The chosen value's probability (``choice``), or the
            probability that the condition holds (``predicate``).
        score: The probability-weighted level index, from 0 to the number of
            levels minus one (``score``).
        probabilities: One ``{"value": <level index>, "probability": ...}`` per
            level (``score``).
    """

    model_config = ConfigDict(extra="allow")

    type: str
    name: str = ""
    choice: str | None = None
    probability: float | None = None
    score: float | None = None
    probabilities: list[dict[str, Any]] | None = None


class DelegationCreatedEvent(BaseModel):
    """``session.delegation.created``: the agent handed a decision to the Decisions API.

    Parameters:
        type: Always ``"session.delegation.created"``.
        delegation_id: Matches the decision event that follows.
        response_id: The response that delegated.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["session.delegation.created"] = "session.delegation.created"
    delegation_id: str
    response_id: str | None = None


class DecisionEvent(BaseModel):
    """``decision.completed``, ``decision.failed`` or ``decision.cancelled``.

    A delegation's result carries ``delegation_id``; a trigger's carries
    ``trigger_id`` and ``source_event_type``.

    Parameters:
        type: Which outcome: ``"decision.completed"`` (with ``answers``),
            ``"decision.failed"`` (with ``error``), or ``"decision.cancelled"``
            (the turn was interrupted before the decision arrived; delegations only).
        delegation_id: The delegation this answers.
        trigger_id: The trigger this answers.
        source_event_type: The trigger's ``event_type``.
        item_id: The conversation item the decision was about.
        response_id: The response that delegated, or that a trigger judged.
        answers: One per question, in question order.
        error: Why no decision was made.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["decision.completed", "decision.failed", "decision.cancelled"]
    delegation_id: str | None = None
    trigger_id: str | None = None
    source_event_type: str | None = None
    item_id: str | None = None
    response_id: str | None = None
    answers: list[DecisionAnswer] = Field(default_factory=list)
    error: dict[str, Any] | None = None

    def answer(self, name: str) -> DecisionAnswer | None:
        """The answer to the question called ``name``, or ``None`` if there's none."""
        return next((a for a in self.answers if a.name == name), None)
