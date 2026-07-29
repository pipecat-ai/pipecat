"""Scenario registry for the transport latency benchmark.

Each scenario names one end-to-end topology: which transport stack the
measuring client uses, where the echo bot runs, and what carries the media.
The registry is the single source of truth for scenario order (tables and
charts render in this order) and for the chart groups that pair a WebRTC
scenario with its MoQ counterpart at the same tier.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    slug: str
    transport: str  # client stack: "moq" | "webrtc" | "daily"
    group: str  # chart group, see GROUPS
    bot: str  # "local" (uv run echo_bot.py) | "pcc" (Pipecat Cloud agent)
    description: str

    @property
    def floor_key(self) -> str | None:
        """Which client-stack floor applies (None: no in-process floor exists)."""
        return self.transport if self.transport in ("moq", "webrtc") else None


SCENARIOS: dict[str, Scenario] = {
    s.slug: s
    for s in [
        Scenario(
            "webrtc-local",
            "webrtc",
            "local-direct",
            "local",
            "SmallWebRTC, host candidates only, one SRTP hop on loopback",
        ),
        Scenario(
            "moq-serve",
            "moq",
            "local-direct",
            "local",
            "MoQ server mode, one QUIC hop on loopback",
        ),
        Scenario(
            "webrtc-turn-local",
            "webrtc",
            "local-relay",
            "local",
            "SmallWebRTC, TURN-forced via dockerized coturn (:3478)",
        ),
        Scenario(
            "moq-relay-local",
            "moq",
            "local-relay",
            "local",
            "MoQ client mode via dockerized moq-relay (:4443)",
        ),
        Scenario(
            "webrtc-turn-deployed",
            "webrtc",
            "deployed-relay",
            "local",
            "SmallWebRTC, TURN-forced via a deployed TURN server",
        ),
        Scenario(
            "moq-relay-deployed",
            "moq",
            "deployed-relay",
            "local",
            "MoQ client mode via a deployed standalone relay (--relay-url)",
        ),
        Scenario(
            "daily-pcc",
            "daily",
            "cloud",
            "pcc",
            "Daily transport, echo bot on Pipecat Cloud",
        ),
        Scenario(
            "moq-pcc",
            "moq",
            "cloud",
            "pcc",
            "MoQ client mode, echo bot on Pipecat Cloud via a deployed relay",
        ),
    ]
}

# Chart groups: each sub-chart pairs the WebRTC and MoQ scenario at one tier.
GROUPS: dict[str, str] = {
    "local-direct": "Local direct (loopback)",
    "local-relay": "Local relay (docker)",
    "deployed-relay": "Deployed relay (internet)",
    "cloud": "Pipecat Cloud",
}

# Scenarios that need nothing beyond the repo + local docker containers.
ALL_LOCAL = ["webrtc-local", "moq-serve", "webrtc-turn-local", "moq-relay-local"]


def scenario_order(slug: str) -> int:
    """Registry position of a slug (unknown slugs sort last)."""
    slugs = list(SCENARIOS)
    return slugs.index(slug) if slug in slugs else len(slugs)
