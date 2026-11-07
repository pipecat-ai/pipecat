#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""What a bot does in a session, reported so clients can configure themselves.

:class:`~pipecat.pipeline.worker.PipelineWorker` derives a
:class:`BotCapabilities` from its pipeline's transports and parameters, and RTVI
sends it to the client in the ``bot-ready`` message.
"""

from pydantic import BaseModel


class BotCapabilities(BaseModel):
    """What a bot does in a session.

    Every field is optional: ``None`` means unknown, and a client should keep
    its default behavior for it. ``False`` means the bot does not do it, so a
    client can hide the matching UI.

    Parameters:
        audio_in: Whether the bot receives the user's audio.
        audio_out: Whether the bot sends audio to the user.
        video_in: Whether the bot receives the user's video.
        screen_in: Whether the bot receives the user's screen share. Known when
            the transport captures its video sources from ``video_in_sources``;
            unknown when the application captures them itself.
        video_out: Whether the bot sends video to the user.
        metrics: Whether the bot reports metrics.
    """

    audio_in: bool | None = None
    audio_out: bool | None = None
    video_in: bool | None = None
    screen_in: bool | None = None
    video_out: bool | None = None
    metrics: bool | None = None

    def combine(self, other: "BotCapabilities") -> "BotCapabilities":
        """Combine the capabilities of two parts of the same bot.

        A field is ``True`` if either side is ``True``, ``False`` if either side
        is ``False`` and neither is ``True``, and ``None`` if both are unknown.

        Args:
            other: The capabilities to combine with these.

        Returns:
            The combined capabilities.
        """
        combined = {}
        for field in type(self).model_fields:
            values = [v for v in (getattr(self, field), getattr(other, field)) if v is not None]
            combined[field] = any(values) if values else None
        return BotCapabilities(**combined)

    def override(self, other: "BotCapabilities") -> "BotCapabilities":
        """Apply the fields ``other`` knows on top of these capabilities.

        Args:
            other: The capabilities whose known (non-``None``) fields win.

        Returns:
            These capabilities with ``other``'s known fields applied.
        """
        return self.model_copy(update=other.model_dump(exclude_none=True))
