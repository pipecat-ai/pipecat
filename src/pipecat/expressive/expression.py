#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Expression capabilities: what a TTS service can perform.

Part of expressive conversation, which lets an agent hear how something was
said and answer in kind. An LLM writes expression into its reply as inline
tags: ``[[name]]`` for a sound or silence at one point, ``<<name>>`` for the
delivery of the words after it, ``<<name: words>>`` for the delivery of just
those words, and ``^^word^^`` to stress a word. A delivery lasts until the next
delivery tag or the end of the response, and ``<<neutral>>`` returns to the
voice's baseline.

For example::

    <<cheerful>> It's [[laughs]] ^^back^^ up. [[pause 300]] Your code is <<slow: A B 7>>.

A TTS service declares which of these it can perform with
:class:`ExpressionCapabilities`, and broadcasts them at start in a
:class:`~pipecat.frames.frames.TTSMetadataFrame`, so the LLM can be told which
tags to write.

Pipecat names every delivery and sound the services perform the same way on
each, the universal names: :data:`UNIVERSAL_DELIVERY` and
:data:`UNIVERSAL_SOUNDS`. A universal name in a tag, such as ``<<warm>>`` or
``[[laugh]]``, is sent to each service by its own name for it, such as
ElevenLabs' ``warmly`` or Cartesia's ``laughter``, while everything Pipecat
reports keeps the name as written.

**Alpha.** The tag syntax and these types are likely to change between releases.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, cast, get_args

# The delivery that returns to the voice's baseline. Part of the protocol, so it is
# a delivery whatever a service declares.
NEUTRAL_DELIVERY = "neutral"

# The delivery of stressed words, written ^^words^^.
EMPHASIS_DELIVERY = "emphasis"

# The deliveries that slow the words down and speed them up. Part of the protocol,
# so every service that performs speed names them the same.
SLOW_DELIVERY = "slow"
FAST_DELIVERY = "fast"

# The deliveries and sounds Pipecat names the same on every TTS service, each in one
# spelling: together, everything the services with an expression adapter perform. A
# tag that names one is sent by each service's own name for it
# (``ExpressionCapabilities.service_names``), and reported as written.
UniversalDelivery = Literal[
    "affectionate",
    "agitated",
    "alarmed",
    "amazed",
    "angry",
    "annoyed",
    "anticipating",
    "anxious",
    "apologetic",
    "appalled",
    "ashamed",
    "bored",
    "calm",
    "cautious",
    "compassionate",
    "confident",
    "confused",
    "contemplative",
    "contemptuous",
    "content",
    "crying",
    "curious",
    "cute",
    "deadpan",
    "dejected",
    "delighted",
    "depressed",
    "desperate",
    "determined",
    "disappointed",
    "disdainful",
    "disgusted",
    "dismissive",
    "distant",
    "doubtful",
    "dramatic",
    "elated",
    "embarrassed",
    "empathetic",
    "enthusiastic",
    "envious",
    "euphoric",
    "excited",
    "fast",
    "fearful",
    "flirtatious",
    "frustrated",
    "grateful",
    "guilty",
    "happy",
    "hesitant",
    "hopeful",
    "hurt",
    "hysterical",
    "impressed",
    "indifferent",
    "insecure",
    "ironic",
    "jealous",
    "lonely",
    "mad",
    "melancholic",
    "mischievous",
    "moved",
    "muttering",
    "mysterious",
    "nervous",
    "nostalgic",
    "optimistic",
    "outraged",
    "panicked",
    "peaceful",
    "pessimistic",
    "professional",
    "proud",
    "questioning",
    "reassuring",
    "regretful",
    "rejected",
    "relaxed",
    "resigned",
    "sad",
    "sarcastic",
    "satisfied",
    "scared",
    "screaming",
    "serene",
    "sheepish",
    "shouting",
    "singing",
    "skeptical",
    "slow",
    "surprised",
    "sympathetic",
    "thoughtful",
    "threatened",
    "tired",
    "triumphant",
    "trusting",
    "uncertain",
    "unhappy",
    "upset",
    "warm",
    "whispering",
    "wistful",
    "worried",
]
UniversalSound = Literal[
    "breath",
    "burp",
    "chuckle",
    "clear throat",
    "cough",
    "deep breath",
    "exhale",
    "gasp",
    "giggle",
    "groan",
    "gulp",
    "hiss",
    "hum",
    "inhale",
    "laugh",
    "laugh harder",
    "light breath",
    "lip smack",
    "pant",
    "sharp exhale",
    "sigh",
    "sneeze",
    "sniff",
    "snore",
    "snort",
    "sob",
    "start laughing",
    "swallow",
    "um",
    "wheeze",
    "woo",
    "yawn",
]
UniversalName = UniversalDelivery | UniversalSound
UNIVERSAL_DELIVERY: tuple[UniversalDelivery, ...] = get_args(UniversalDelivery)
UNIVERSAL_SOUNDS: tuple[UniversalSound, ...] = get_args(UniversalSound)

# The universal deliveries and sounds Pipecat Tone's tagger writes.
CORE_DELIVERY: tuple[UniversalDelivery, ...] = (
    "calm",
    "sympathetic",
    "apologetic",
    "warm",
    "excited",
    "surprised",
    "sad",
    "grateful",
    "curious",
    "confident",
    "slow",
    "fast",
)
CORE_SOUNDS: tuple[UniversalSound, ...] = ("laugh", "chuckle", "sigh", "breath")


class DeliveryScope(Enum):
    """Where in a response a TTS service can change delivery.

    In ``"<<calm>> Sure, <<excited>> it worked!"``, a service with
    ``ANYWHERE`` speaks "it worked!" excited. One with ``SENTENCE`` speaks the
    whole sentence calm, the delivery its words start in, and the next sentence
    excited. A span's delivery is the exception: ``SENTENCE`` speaks the whole of
    ``"Your code is <<slow: A B 7>>."`` slowly, so the span's words still are.

    Parameters:
        ANYWHERE: A delivery takes effect where its tag appears.
        SENTENCE: A delivery can change only between requests, for a service
            that takes delivery as a setting of each request, such as MiniMax,
            or that changes it reliably only between generations, such as
            Cartesia. A sentence is one request; when the service streams
            tokens, a request is a token, so a change can take effect within
            a sentence.
    """

    ANYWHERE = "anywhere"
    SENTENCE = "sentence"


@dataclass
class ExpressionTerm:
    """One sound or delivery a TTS service can perform.

    Parameters:
        name: The word written inside the tag, such as ``"laugh"`` or
            ``"cheerful"``.
        description: A short line on when to use it, or empty.
    """

    name: str
    description: str = ""


@dataclass
class ExpressionCapabilities:
    """The expression tags a TTS service can perform.

    The LLM is taught these, and tags outside them are left out of the
    service's speech. A service that can laugh, speak in two moods, pause and
    stress words declares::

        ExpressionCapabilities(
            sounds=[ExpressionTerm("laughs")],
            delivery=[
                ExpressionTerm("cheerful"),
                ExpressionTerm("calm", "to reassure a worried caller"),
            ],
            pause_range_ms=(100, 3000),
            emphasis=True,
        )

    Parameters:
        sounds: Sounds the service can make at one point, written ``[[name]]``.
        delivery: Deliveries the service can apply to the words that follow,
            written ``<<name>>``.
        free_text_sounds: Whether the service also accepts sounds outside
            ``sounds``, described in a few words.
        free_text_delivery: Whether the service also accepts deliveries outside
            ``delivery``, described in a few words.
        pause_range_ms: The shortest and longest pause the service can make,
            in milliseconds, or None when it can't pause.
        emphasis: Whether the service can stress words marked ``^^word^^``.
        delivery_scope: Where in a response the service can change delivery.
        service_names: The names the service performs universal deliveries and
            sounds by (``UNIVERSAL_DELIVERY``, ``UNIVERSAL_SOUNDS``), best first,
            where they differ from the universal names, such as ``{"warm":
            ["content", "happy"], "laugh": ["laughter"]}``. A universal name
            written in a tag is sent by the first of these the service performs.
            A later name stands in when an expression policy leaves out the ones
            before it. A universal name left out is performed by its own, when the
            service declares it or accepts free text.
    """

    sounds: list[ExpressionTerm] = field(default_factory=list)
    delivery: list[ExpressionTerm] = field(default_factory=list)
    free_text_sounds: bool = False
    free_text_delivery: bool = False
    pause_range_ms: tuple[int, int] | None = None
    emphasis: bool = False
    delivery_scope: DeliveryScope = DeliveryScope.ANYWHERE
    service_names: dict[UniversalName, list[str]] = field(default_factory=dict)

    def delivery_name(self, name: str) -> str | None:
        """The name the service performs a delivery by, or None when it can't.

        A universal name is performed by its first name in ``service_names`` the
        service performs, else by its own; any other name by itself. The service
        performs a name it declares, or any name when it accepts free text.

        Args:
            name: The delivery, as written in its tag.

        Returns:
            The name to send the service.
        """
        return self._performed_name(name, self.delivery, self.free_text_delivery)

    def sound_name(self, name: str) -> str | None:
        """The name the service performs a sound by, or None when it can't.

        Read as :meth:`delivery_name` reads a delivery.

        Args:
            name: The sound, as written in its tag.

        Returns:
            The name to send the service.
        """
        return self._performed_name(name, self.sounds, self.free_text_sounds)

    def _performed_name(self, name: str, terms: list[ExpressionTerm], free_text: bool):
        declared = {term.name for term in terms}
        vocabulary = cast(dict[str, list[str]], self.service_names)
        for candidate in [*vocabulary.get(name, []), name]:
            if free_text or candidate in declared:
                return candidate
        return None


# Slow and fast, as a service that performs them declares them.
SPEED_DELIVERY = [
    ExpressionTerm(
        SLOW_DELIVERY, "to say something carefully, such as a number, a code or an address"
    ),
    ExpressionTerm(FAST_DELIVERY, "to say something in passing"),
]
