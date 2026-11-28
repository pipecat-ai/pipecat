#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Media encryption settings for the SIP transport.

Kept apart from :mod:`pipecat.transports.sip.connection` so the runner can
name the setting without importing the SIP stack, which is an optional
extra.
"""

from enum import StrEnum


class SIPMediaEncryption(StrEnum):
    """Whether a SIP account's call audio and video are encrypted.

    Passed as ``SIPConnection(media_encryption=...)``; the runner reads it
    from ``SIP_MEDIA_ENCRYPTION``.

    Parameters:
        NONE: Calls placed are unencrypted. Calls received follow the
            caller: unencrypted, or SRTP when the caller offers it.
        SRTP: Every call is encrypted with SRTP, its keys exchanged in the
            call setup (SDES). A received call that offers no SRTP is
            refused with 488 before the application sees it. The keys are
            only as private as the signalling that carries them, so pair
            it with ``transport="tls"``.
    """

    NONE = "none"
    SRTP = "srtp"

    @classmethod
    def parse(cls, value: "SIPMediaEncryption | str") -> "SIPMediaEncryption":
        """Return the member for ``value``, a member or its name in any case.

        Raises:
            ValueError: ``value`` is not a string, or names no member.
        """
        if isinstance(value, str):
            try:
                return cls(value.strip().lower())
            except ValueError:
                pass
        allowed = ", ".join(repr(member.value) for member in cls)
        raise ValueError(f"media encryption must be one of {allowed}, got {value!r}")
