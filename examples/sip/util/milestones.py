#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Numbered call-progress milestones for the SIP examples.

:func:`attach_milestones` registers event handlers on a
:class:`~pipecat.transports.sip.transport.SIPTransport` that log one INFO
line per stage of a call, so a developer watching the terminal can tell how
far a call got and where it stalled:

- ``[1/6]`` REGISTER accepted (or skipped, in trunk mode)
- ``[2/6]`` INVITE seen, call ringing
- ``[3/6]`` call answered, media flowing
- ``[4/6]`` pipeline started — logged by ``bot.py``, not here
- ``[5/6]`` call ended
- ``[6/6]`` final media quality snapshot

Handlers are appended to whatever the bot already registered, so attaching
milestones never displaces bot behavior.

Run this module directly to print every milestone against a fake transport::

    uv run python examples/sip/util/milestones.py
"""

import asyncio
import os

from loguru import logger

# Error payload substrings that identify a likely cause, most specific first.
_ERROR_HINTS = (
    (
        ("401", "403", "unauthorized", "forbidden", "authentication"),
        "likely cause: bad credentials — check SIP_USER/SIP_PASS "
        "(and SIP_AUTH_USER when the auth user differs from the AOR)",
    ),
    (
        ("408", "timeout", "timed out", "no route", "unreachable", "connection refused", "dns"),
        "likely cause: registrar unreachable — check SIP_DOMAIN, SIP_TRANSPORT, "
        "and that outbound SIP is not firewalled",
    ),
    (
        ("rtp", "no audio", "media", "one-way", "ice", "stun"),
        "likely cause: NAT — media is not crossing; see the Symmetric RTP and "
        "medianat notes in README.md",
    ),
)


def _error_hint(payload) -> str:
    """Pick a likely-cause hint for an error payload.

    Args:
        payload: Error string or dict the transport reported.

    Returns:
        One-line hint, or an empty string when nothing matches.
    """
    text = str(payload).lower()
    for needles, hint in _ERROR_HINTS:
        if any(n in text for n in needles):
            return hint
    return ""


def attach_milestones(transport, *, direction: str) -> None:
    """Log numbered call-progress milestones for a SIP call.

    Args:
        transport: SIP transport to observe.
        direction: ``"dialin"`` when the bot answers calls, ``"dialout"``
            when it places them. Selects which answer event reports
            ``[3/6]``.

    Raises:
        ValueError: ``direction`` is neither ``"dialin"`` nor ``"dialout"``.
    """
    if direction not in ("dialin", "dialout"):
        raise ValueError(f"direction must be 'dialin' or 'dialout', got {direction!r}")

    # Each milestone reports once per bot run: the stack re-fires ringing per
    # provisional response, and call end arrives on two or three events.
    seen = set()

    def once(key: str) -> bool:
        if key in seen:
            return False
        seen.add(key)
        return True

    @transport.event_handler("on_dialin_ready")
    async def _on_dialin_ready(transport, aor):
        logger.info(f"[1/6] REGISTER ok — reachable at {aor}")

    @transport.event_handler("on_connected")
    async def _on_connected(transport):
        # Trunk mode authenticates per INVITE, so no REGISTER ever lands.
        if os.getenv("SIP_REG_INTERVAL") == "0":
            logger.info("[1/6] trunk mode, skipping REGISTER")

    @transport.event_handler("on_call_state_updated")
    async def _on_call_state_updated(transport, state):
        if state == "ringing" and once("ringing"):
            logger.info("[2/6] INVITE — call ringing")

    answered_event = "on_dialin_connected" if direction == "dialin" else "on_dialout_answered"

    @transport.event_handler(answered_event)
    async def _on_answered(transport, data):
        if once("answered"):
            logger.info("[3/6] answered, media flowing (first RTP received)")

    async def _on_ended(transport, *args):
        if once("ended"):
            logger.info("[5/6] call ended")

    for event in ("on_dialin_stopped", "on_dialout_stopped", "on_client_disconnected"):
        transport.add_event_handler(event, _on_ended)

    @transport.event_handler("on_call_quality_stats")
    async def _on_call_quality_stats(transport, stats):
        logger.info(f"[6/6] quality: {stats}")

    @transport.event_handler("on_dtmf_event")
    async def _on_dtmf_event(transport, data):
        tone = data.get("tone") if isinstance(data, dict) else data
        logger.info(f"DTMF {tone}")

    async def _on_warning(transport, data):
        logger.warning(f"SIP audio warning: {data}")

    transport.add_event_handler("on_dialin_warning", _on_warning)
    transport.add_event_handler("on_dialout_warning", _on_warning)

    @transport.event_handler("on_error")
    async def _on_error(transport, error):
        hint = _error_hint(error)
        logger.error(f"SIP error: {error}{f' — {hint}' if hint else ''}")


class _FakeTransport:
    """Records handlers by event name and fires them on demand."""

    def __init__(self):
        self.handlers = {}

    def event_handler(self, event_name: str):
        def decorator(handler):
            self.add_event_handler(event_name, handler)
            return handler

        return decorator

    def add_event_handler(self, event_name: str, handler):
        self.handlers.setdefault(event_name, []).append(handler)

    async def fire(self, event_name: str, *args):
        for handler in self.handlers.get(event_name, []):
            await handler(self, *args)


async def _smoke():
    """Fire every milestone handler with a representative payload."""
    for direction, answered in (
        ("dialin", "on_dialin_connected"),
        ("dialout", "on_dialout_answered"),
    ):
        logger.info(f"--- direction={direction} ---")
        transport = _FakeTransport()
        os.environ["SIP_REG_INTERVAL"] = "0"
        attach_milestones(transport, direction=direction)

        await transport.fire("on_connected")
        await transport.fire("on_dialin_ready", "sip:pipecat-harness-bot@example.sip.daily.co")
        await transport.fire("on_call_state_updated", "ringing")
        await transport.fire(answered, {"sessionId": "s1", "direction": direction[4:]})
        await transport.fire("on_dtmf_event", {"sessionId": "s1", "tone": "5"})
        await transport.fire(
            "on_dialin_warning", {"sessionId": "s1", "errorMsg": "rtp stream stalled"}
        )
        await transport.fire("on_call_state_updated", "closed")
        await transport.fire("on_dialin_stopped", {"sessionId": "s1"})
        await transport.fire("on_client_disconnected", {"id": "s1"})
        await transport.fire("on_call_quality_stats", "duration=12s loss=0.0% jitter=3ms rtt=41ms")
        for error in (
            {"code": 401, "errorMsg": "Unauthorized"},
            "408 Request Timeout contacting registrar",
            "no RTP received for 30s",
        ):
            await transport.fire("on_error", error)


if __name__ == "__main__":
    asyncio.run(_smoke())
