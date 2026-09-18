#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Voicemail classification with a TypeSafe ``Choice`` instead of a text LLM.

:class:`~pipecat.extensions.voicemail.voicemail_detector.VoicemailDetector`
takes an LLM that answers "CONVERSATION" or "VOICEMAIL".
:class:`TypeSafeVoicemailClassifier` is that LLM, backed by one TypeSafe
judgment per caller turn on the transcript so far. The detector itself does
not know the difference.

Requires the ``typesafe`` extra: ``uv add "pipecat-ai[typesafe]"``.
"""

from typing import Any

from pipecat.services.typesafe.choice_llm import TypeSafeChoiceLLMService
from pipecat.services.typesafe.judge import TypeSafeJudge

CONVERSATION = "CONVERSATION"
"""The label meaning a live person answered."""

VOICEMAIL = "VOICEMAIL"
"""The label meaning a recording or automated system answered."""

DEFAULT_INSTRUCTIONS = (
    "A bot placed an outbound phone call and `transcript` is a speech-to-text transcript "
    "of everything heard so far from the side that answered. Did a live person answer, or "
    "did the call reach a voicemail or other automated system?"
)

DEFAULT_CRITERIA: dict[str, Any] = {
    CONVERSATION: {
        "what": (
            "A live person answered and is talking to the caller: a greeting, a question "
            "to the caller, or other spontaneous speech that expects a reply"
        ),
        "not_for": (
            "A recorded greeting, even a casual one, that tells the caller what to do "
            "instead of waiting for them to speak"
        ),
        "examples": [
            "Hello?",
            "Hi",
            "Yeah?",
            "John speaking",
            "Who is this?",
            "Can I help you?",
            "Sorry, I'm in the middle of something, what's up?",
        ],
    },
    VOICEMAIL: {
        "what": (
            "A recording or automated system: a voicemail greeting (typically 'you've "
            "reached', 'not available right now', 'leave a message', 'I'll get back to "
            "you'), a carrier message ('not in service', 'mailbox is full'), or a business "
            "message played to every caller ('our office is currently closed')"
        ),
        "not_for": (
            "A person who is busy or distracted but is still speaking to the caller and "
            "waiting for an answer"
        ),
        "examples": [
            "Hi, you've reached Jamie. Please leave a message.",
            "This is Sarah, I'm not available right now, leave your name and number.",
            "I can't come to the phone right now. Leave a message after the tone.",
            "The number you have dialed is not in service.",
            "The mailbox is full.",
            "Thank you for calling. Our office is currently closed.",
        ],
    },
}


class TypeSafeVoicemailClassifier(TypeSafeChoiceLLMService):
    """A voicemail classifier LLM whose verdicts are TypeSafe judgments.

    Pass it as the ``llm`` of a
    :class:`~pipecat.extensions.voicemail.voicemail_detector.VoicemailDetector`.
    Each caller turn becomes one ``Choice`` between ``CONVERSATION`` and
    ``VOICEMAIL`` over the transcript so far, answered in about a fifth of a
    second with no text generated. A verdict below ``confidence_threshold``
    is answered with an empty response, which the detector treats as no
    verdict yet, so a lone "sorry, I can't come to the phone right now" is
    judged again together with whatever follows it.

    Example::

        detector = VoicemailDetector(llm=TypeSafeVoicemailClassifier())
    """

    def __init__(
        self,
        *,
        judge: TypeSafeJudge | None = None,
        instructions: str = DEFAULT_INSTRUCTIONS,
        criteria: dict[str, Any] | None = None,
        confidence_threshold: float = 0.5,
        **kwargs,
    ):
        """Initialize the classifier.

        Args:
            judge: The TypeSafe client wrapper. Defaults to a new judge that
                reads ``TYPESAFE_API_KEY``.
            instructions: The question asked about the transcript, which the
                state exposes as `transcript`.
            criteria: Descriptions of the two options, keyed ``CONVERSATION``
                and ``VOICEMAIL``, as strings or as objects with ``what``,
                ``not_for`` and ``examples``. Defaults to :data:`DEFAULT_CRITERIA`.
            confidence_threshold: Minimum confidence to answer with a verdict.
                Below it the classifier answers nothing and the detector waits
                for the caller side to say more.
            **kwargs: Passed to :class:`TypeSafeChoiceLLMService`.
        """
        options = dict(criteria) if criteria is not None else dict(DEFAULT_CRITERIA)
        missing = {CONVERSATION, VOICEMAIL} - set(options)
        if missing:
            raise ValueError(f"criteria must describe both labels; missing {sorted(missing)}")
        super().__init__(
            judge=judge if judge is not None else TypeSafeJudge(),
            instructions=instructions,
            criteria=options,
            confidence_threshold=confidence_threshold,
            **kwargs,
        )
