#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""How a TTS service performs expression tags.

Part of expressive conversation (alpha). Each TTS service has an expression
adapter, a :class:`BaseExpressionAdapter`, the way each LLM service has an LLM
adapter. The adapter declares what the service can perform and renders it into the text
sent to the service: words in their modes, and events and pauses. The base
adapter performs nothing, so a service without one speaks only the words.

**Alpha.** The tag syntax and these types are likely to change between releases.
"""

from pipecat.expressive.expression import ExpressionCapabilities


class BaseExpressionAdapter:
    """Renders expression for one TTS service. The base performs nothing.

    Its capabilities are what the service can perform: the service broadcasts
    them at start, and reads and renders tags by them. The service calls the
    adapter only with what they say it supports: other modes are taken off
    the words first, and other events are never rendered.

    Markup is set apart from the words by spaces, so the words a service with
    word timestamps reports line up with the text it was sent.
    """

    def capabilities(self) -> ExpressionCapabilities | None:
        """The expression the service can perform, or None for none.

        Returns:
            The capabilities, which the service broadcasts at start.
        """
        return None

    def render_text(self, text: str, modes: list[str], previous_modes: list[str] | None) -> str:
        """Render words to speak in their modes.

        Args:
            text: The words, as prepared for the service.
            modes: The supported modes to speak them in, empty for the voice's
                baseline.
            previous_modes: The modes of the words before them in the response,
                or None for its first words, so markup can be emitted only
                where the modes change.

        Returns:
            The text to send to the service.
        """
        return text

    def render_event(self, name: str) -> str:
        """Render a supported event, such as a laugh, at its place in the text.

        Args:
            name: The event, as written between the brackets.

        Returns:
            The text to send to the service.
        """
        return ""

    def render_pause(self, duration_ms: int) -> str:
        """Render a pause at its place in the text.

        Args:
            duration_ms: How long, in milliseconds, within the service's
                pause range.

        Returns:
            The text to send to the service.
        """
        return ""
