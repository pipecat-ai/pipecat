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

DEFAULT_CRITERIA = {
    CONVERSATION: (
        "A live person answered: a personal greeting such as 'Hello?', 'Hi', 'Yeah?' or "
        "'John speaking'; a question to the caller such as 'Who is this?' or 'Can I help "
        "you?'; or other spontaneous speech that expects a reply"
    ),
    VOICEMAIL: (
        "A recording or automated system: a voicemail greeting such as 'you've reached', "
        "'not available right now', 'leave a message', 'leave your name and number' or "
        "'I'll get back to you'; a carrier message such as 'not in service', 'mailbox is "
        "full' or 'has not been set up'; or a business message such as 'our office is "
        "currently closed'"
    ),
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
        criteria: dict[str, str] | None = None,
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
                and ``VOICEMAIL``. Defaults to :data:`DEFAULT_CRITERIA`.
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
