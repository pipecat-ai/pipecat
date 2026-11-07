#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""How a TTS service performs expression tags.

Part of expressive conversation (alpha). Each TTS service has an expression
adapter, a :class:`BaseExpressionAdapter`, the way each LLM service has an LLM
adapter. The adapter declares what the service can perform and formats it into the text
sent to the service: words in their delivery, and sounds and pauses. The base
adapter performs nothing, so a service without one speaks only the words.

**Alpha.** The tag syntax and these types are likely to change between releases.
"""

from typing import Any

from pipecat.expressive.expression import EMPHASIS_DELIVERY, ExpressionCapabilities


class BaseExpressionAdapter:
    """Formats expression for one TTS service. The base performs nothing.

    Its capabilities are what the service can perform: the service broadcasts
    them at start, and reads and formats tags by them. The service calls the
    adapter only with what they say it supports: other delivery is taken off
    the words first, and other sounds are never formatted. Each name reaches the
    adapter as the service's own: a universal name is translated by the
    capabilities' ``service_vocabulary`` first, so ``warm`` reaches ElevenLabs' adapter
    as ``warmly``.

    Markup is set apart from the words by spaces, so the words a service with
    word timestamps reports line up with the text it was sent.
    """

    def capabilities(self) -> ExpressionCapabilities | None:
        """The expression the service can perform, or None for none.

        Returns:
            The capabilities, which the service broadcasts at start.
        """
        return None

    def format_text(
        self, text: str, delivery: list[str], previous_delivery: list[str] | None
    ) -> str:
        """Format words to speak in their delivery.

        Args:
            text: The words, as prepared for the service.
            delivery: The supported delivery to speak them in, by the service's
                names, empty for the voice's baseline.
            previous_delivery: The delivery of the words before them in the response,
                or None for its first words, so markup can be emitted only
                where the delivery changes.

        Returns:
            The text to send to the service.
        """
        return text

    def request_settings(self, delivery: list[str]) -> dict[str, Any]:
        """Settings to send beside a request's text, for its delivery.

        For a service that takes some of its expression as request settings
        rather than as markup in the text, such as a speed. The service merges
        them into its request; each adapter documents where.

        Args:
            delivery: The supported delivery the request's first words are
                spoken in, by the service's names, empty for the voice's baseline.

        Returns:
            The settings, empty for none.
        """
        return {}

    def format_end(self, delivery: list[str]) -> str:
        """Close a sentence's request, given the delivery its last words are spoken in.

        For a service whose markup lasts past the request it is sent in, so
        the request can set the delivery back for the text after it.

        Args:
            delivery: The delivery of the request's last words, by the service's
                names.

        Returns:
            The text to end the request with, or an empty string.
        """
        return ""

    def format_sound(self, name: str) -> str:
        """Format a supported sound, such as a laugh, at its place in the text.

        Args:
            name: The sound, by the service's name for it.

        Returns:
            The text to send to the service.
        """
        return ""

    def format_pause(self, duration_ms: int) -> str:
        """Format a pause at its place in the text.

        Args:
            duration_ms: How long, in milliseconds, within the service's
                pause range.

        Returns:
            The text to send to the service.
        """
        return ""


def delivery_tags(delivery: list[str] | None, renames: dict[str, str]) -> list[str]:
    """A delivery's names as a provider's tags, for an adapter that writes each as one.

    Emphasis is left out, since such a provider stresses words its own way.

    Args:
        delivery: The delivery, or None.
        renames: The provider's tag for a name it calls something else, such as
            ``slowly`` for ``slow``.

    Returns:
        The tags, in order.
    """
    return [renames.get(name, name) for name in delivery or [] if name != EMPHASIS_DELIVERY]
