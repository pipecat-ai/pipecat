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
- :class:`Think`: the model, its instructions and tools.
- :class:`Speak`: the agent's audio, its format and voice.

Only the fields a caller sets are sent; anything left unset takes the server's
default. Turn detection, noise reduction, barge-in and history trimming are
built in and not configurable.

See https://developers.zoom.us/docs/ai-services/voice-agent/live-mode/
"""

from typing import Any, Literal

from pydantic import BaseModel


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
            your own provider key.
        metadata: The provider model's limits, such as ``context_window`` and
            ``max_output_tokens``.
    """

    type: Literal["zoom", "openai", "anthropic"] = "zoom"
    model: str | None = None
    headers: dict[str, str] | None = None
    metadata: dict[str, Any] | None = None


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
    """

    provider: Provider | None = None
    instructions: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_token_limit: int | Literal["inf"] | None = None
    reasoning: dict[str, Any] | None = None


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
