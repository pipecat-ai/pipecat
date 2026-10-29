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

    <<warm>> It's [[laugh]] ^^back^^ up. [[pause 300]] Your code is <<slow: A B 7>>.

A TTS service declares which of these it can perform with
:class:`ExpressionCapabilities`, and broadcasts them at start in a
:class:`~pipecat.frames.frames.TTSMetadataFrame`, so the LLM can be told which
tags to write.

Pipecat names every delivery and sound the services perform the same way on
each, the universal names: :class:`ExpressionDelivery` and
:class:`ExpressionSound`. A universal name in a tag, such as ``<<warm>>`` or
``[[laugh]]``, is sent to each service by its own name for it, such as
ElevenLabs' ``warmly`` or Cartesia's ``laughter``, while everything Pipecat
reports keeps the name as written.

**Alpha.** The tag syntax and these types are likely to change between releases.
"""

from dataclasses import dataclass, field
from enum import Enum, StrEnum

# The delivery that returns to the voice's baseline. Part of the protocol, so it is
# a delivery whatever a service declares.
NEUTRAL_DELIVERY = "neutral"

# The delivery of stressed words, written ^^words^^.
EMPHASIS_DELIVERY = "emphasis"

# The deliveries that slow the words down and speed them up. Part of the protocol,
# so every service that performs speed names them the same.
SLOW_DELIVERY = "slow"
FAST_DELIVERY = "fast"


class ExpressionDelivery(StrEnum):
    """A delivery by its universal name, the same on every TTS service.

    With :class:`ExpressionSound`, these name everything the services with an
    expression adapter perform, each in one spelling, an adjective for a
    delivery. A tag that names one is sent by each service's own name for it
    (``ExpressionCapabilities.service_vocabulary``), and reported as written. A
    member is a ``str``, so ``ExpressionDelivery.WARM`` is ``"warm"``.
    """

    AFFECTIONATE = "affectionate"
    AGITATED = "agitated"
    ALARMED = "alarmed"
    AMAZED = "amazed"
    AMUSED = "amused"
    ANGRY = "angry"
    ANNOYED = "annoyed"
    ANTICIPATING = "anticipating"
    ANXIOUS = "anxious"
    APOLOGETIC = "apologetic"
    APPALLED = "appalled"
    ASHAMED = "ashamed"
    BORED = "bored"
    CALM = "calm"
    CAUTIOUS = "cautious"
    COMPASSIONATE = "compassionate"
    CONFIDENT = "confident"
    CONFUSED = "confused"
    CONTEMPLATIVE = "contemplative"
    CONTEMPTUOUS = "contemptuous"
    CONTENT = "content"
    CRYING = "crying"
    CURIOUS = "curious"
    CUTE = "cute"
    DEADPAN = "deadpan"
    DEJECTED = "dejected"
    DELIGHTED = "delighted"
    DEPRESSED = "depressed"
    DESPERATE = "desperate"
    DETERMINED = "determined"
    DISAPPOINTED = "disappointed"
    DISDAINFUL = "disdainful"
    DISGUSTED = "disgusted"
    DISMISSIVE = "dismissive"
    DISTANT = "distant"
    DOUBTFUL = "doubtful"
    DRAMATIC = "dramatic"
    ELATED = "elated"
    EMBARRASSED = "embarrassed"
    EMPATHETIC = "empathetic"
    ENCOURAGING = "encouraging"
    ENTHUSIASTIC = "enthusiastic"
    ENVIOUS = "envious"
    EUPHORIC = "euphoric"
    EXCITED = "excited"
    FAST = "fast"
    FEARFUL = "fearful"
    FLIRTATIOUS = "flirtatious"
    FRUSTRATED = "frustrated"
    GRATEFUL = "grateful"
    GUILTY = "guilty"
    HAPPY = "happy"
    HESITANT = "hesitant"
    HOPEFUL = "hopeful"
    HURT = "hurt"
    HYSTERICAL = "hysterical"
    IMPRESSED = "impressed"
    INDIFFERENT = "indifferent"
    INSECURE = "insecure"
    IRONIC = "ironic"
    JEALOUS = "jealous"
    LONELY = "lonely"
    LOUD = "loud"
    MAD = "mad"
    MELANCHOLIC = "melancholic"
    MISCHIEVOUS = "mischievous"
    MOCKING = "mocking"
    MOVED = "moved"
    MUTTERING = "muttering"
    MYSTERIOUS = "mysterious"
    NERVOUS = "nervous"
    NOSTALGIC = "nostalgic"
    OPTIMISTIC = "optimistic"
    OUTRAGED = "outraged"
    PANICKED = "panicked"
    PEACEFUL = "peaceful"
    PESSIMISTIC = "pessimistic"
    PLAYFUL = "playful"
    PROFESSIONAL = "professional"
    PROUD = "proud"
    QUESTIONING = "questioning"
    REASSURING = "reassuring"
    REGRETFUL = "regretful"
    REJECTED = "rejected"
    RELAXED = "relaxed"
    RELIEVED = "relieved"
    RESIGNED = "resigned"
    SAD = "sad"
    SARCASTIC = "sarcastic"
    SATISFIED = "satisfied"
    SCARED = "scared"
    SCREAMING = "screaming"
    SERENE = "serene"
    SERIOUS = "serious"
    SHEEPISH = "sheepish"
    SHOUTING = "shouting"
    SINCERE = "sincere"
    SINGING = "singing"
    SKEPTICAL = "skeptical"
    SLOW = "slow"
    SOFT = "soft"
    SOMBER = "somber"
    STERN = "stern"
    SURPRISED = "surprised"
    SYMPATHETIC = "sympathetic"
    THOUGHTFUL = "thoughtful"
    THREATENED = "threatened"
    TIRED = "tired"
    TRIUMPHANT = "triumphant"
    TRUSTING = "trusting"
    UNCERTAIN = "uncertain"
    UNHAPPY = "unhappy"
    UPSET = "upset"
    URGENT = "urgent"
    WARM = "warm"
    WHISPERING = "whispering"
    WISTFUL = "wistful"
    WORRIED = "worried"


class ExpressionSound(StrEnum):
    """A sound by its universal name, the same on every TTS service.

    Named as :class:`ExpressionDelivery` names a delivery, in a base word, such
    as ``laugh``.
    """

    BREATH = "breath"
    BURP = "burp"
    CACKLE = "cackle"
    CHUCKLE = "chuckle"
    CLEAR_THROAT = "clear throat"
    COUGH = "cough"
    CRY = "cry"
    DEEP_BREATH = "deep breath"
    EXHALE = "exhale"
    GASP = "gasp"
    GIGGLE = "giggle"
    GROAN = "groan"
    GROWL = "growl"
    GRUNT = "grunt"
    GULP = "gulp"
    HICCUP = "hiccup"
    HISS = "hiss"
    HUFF = "huff"
    HUM = "hum"
    INHALE = "inhale"
    LAUGH = "laugh"
    LAUGH_HARDER = "laugh harder"
    LIGHT_BREATH = "light breath"
    LIP_SMACK = "lip smack"
    MOAN = "moan"
    PANT = "pant"
    SCOFF = "scoff"
    SHARP_EXHALE = "sharp exhale"
    SHRIEK = "shriek"
    SHUSH = "shush"
    SIGH = "sigh"
    SNEEZE = "sneeze"
    SNIFF = "sniff"
    SNIFFLE = "sniffle"
    SNORE = "snore"
    SNORT = "snort"
    SOB = "sob"
    START_LAUGHING = "start laughing"
    SWALLOW = "swallow"
    TONGUE_CLICK = "tongue click"
    UM = "um"
    WHEEZE = "wheeze"
    WHIMPER = "whimper"
    WHISTLE = "whistle"
    WOO = "woo"
    YAWN = "yawn"


# The universal deliveries and sounds an LLM is taught when its policy names none, and
# that Pipecat Tone's tagger writes.
CORE_DELIVERY: tuple[ExpressionDelivery, ...] = (
    ExpressionDelivery.CALM,
    ExpressionDelivery.SYMPATHETIC,
    ExpressionDelivery.APOLOGETIC,
    ExpressionDelivery.WARM,
    ExpressionDelivery.EXCITED,
    ExpressionDelivery.SURPRISED,
    ExpressionDelivery.SAD,
    ExpressionDelivery.GRATEFUL,
    ExpressionDelivery.CURIOUS,
    ExpressionDelivery.CONFIDENT,
    ExpressionDelivery.SLOW,
    ExpressionDelivery.FAST,
)
CORE_SOUNDS: tuple[ExpressionSound, ...] = (
    ExpressionSound.LAUGH,
    ExpressionSound.CHUCKLE,
    ExpressionSound.SIGH,
    ExpressionSound.BREATH,
)


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
        name: The name the service performs it by, such as ``"laughter"`` or
            ``"cheerful"``.
        description: A short line on when to use it, or empty.
    """

    name: str
    description: str = ""


@dataclass
class ExpressionCapabilities:
    """The expression tags a TTS service can perform.

    The LLM is taught these by their universal names, and tags outside them are
    left out of the service's speech. A service that can laugh, speak in two
    moods, pause and stress words declares::

        ExpressionCapabilities(
            sounds=[ExpressionTerm("laughs")],
            delivery=[
                ExpressionTerm("warmly"),
                ExpressionTerm("calm", "to reassure a worried caller"),
            ],
            pause_range_ms=(100, 3000),
            emphasis=True,
            service_vocabulary={
                ExpressionSound.LAUGH: ["laughs"],
                ExpressionDelivery.WARM: ["warmly"],
            },
        )

    Parameters:
        sounds: Sounds the service can make at one point, by its own names.
        delivery: Deliveries the service can apply to the words that follow, by
            its own names.
        allows_free_text_sounds: Whether the service also accepts sounds outside
            ``sounds``, described in a few words.
        allows_free_text_delivery: Whether the service also accepts deliveries
            outside ``delivery``, described in a few words.
        pause_range_ms: The shortest and longest pause the service can make,
            in milliseconds, or None when it can't pause.
        emphasis: Whether the service can stress words marked ``^^word^^``.
        delivery_scope: Where in a response the service can change delivery.
        service_vocabulary: The names the service performs universal deliveries and
            sounds by (:class:`ExpressionDelivery`, :class:`ExpressionSound`),
            best first, where they differ from the universal names, such as
            ``{ExpressionDelivery.WARM: ["content", "happy"]}``. A universal name
            written in a tag is sent by the first of these the service performs.
            A universal name left out is performed by its own, when the service
            declares it or accepts free text.
    """

    sounds: list[ExpressionTerm] = field(default_factory=list)
    delivery: list[ExpressionTerm] = field(default_factory=list)
    allows_free_text_sounds: bool = False
    allows_free_text_delivery: bool = False
    pause_range_ms: tuple[int, int] | None = None
    emphasis: bool = False
    delivery_scope: DeliveryScope = DeliveryScope.ANYWHERE
    service_vocabulary: dict[ExpressionDelivery | ExpressionSound, list[str]] = field(
        default_factory=dict
    )

    def delivery_name(self, name: str) -> str | None:
        """The name the service performs a delivery by, or None when it can't.

        A universal delivery is performed by its first name in ``service_vocabulary``
        the service performs, else by its own; any other name by itself. The service
        performs a name it declares, or any name when it accepts free text.

        Args:
            name: The delivery, as written in its tag.

        Returns:
            The name to send the service.
        """
        return self._performed_name(
            name, self.delivery, ExpressionDelivery, self.allows_free_text_delivery
        )

    def sound_name(self, name: str) -> str | None:
        """The name the service performs a sound by, or None when it can't.

        Read as :meth:`delivery_name` reads a delivery.

        Args:
            name: The sound, as written in its tag.

        Returns:
            The name to send the service.
        """
        return self._performed_name(
            name, self.sounds, ExpressionSound, self.allows_free_text_sounds
        )

    def _performed_name(
        self,
        name: str,
        terms: list[ExpressionTerm],
        universal: type[ExpressionDelivery] | type[ExpressionSound],
        free_text: bool,
    ):
        declared = {term.name for term in terms}
        try:
            names = self.service_vocabulary.get(universal(name), [])
        except ValueError:  # Not a universal name, such as an accent.
            names = []
        for candidate in [*names, name]:
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
