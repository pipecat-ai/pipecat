#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Prometheus metrics for a Deepgram Voice Agent API server.

The Voice Agent API exposes per-turn latencies as Prometheus histograms on
``/metrics``. :class:`VoiceAgentMetrics` keeps them for every session on a
server and renders them in the Prometheus text format.
"""

import threading
from collections.abc import Sequence

DEFAULT_BUCKETS = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0)


class _Histogram:
    def __init__(self, name: str, help_text: str, buckets: Sequence[float]):
        self.name = name
        self.help_text = help_text
        self.buckets = tuple(buckets)
        self.counts = [0] * len(self.buckets)
        self.total = 0.0
        self.count = 0

    def observe(self, value: float):
        for i, bound in enumerate(self.buckets):
            if value <= bound:
                self.counts[i] += 1
        self.total += value
        self.count += 1

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help_text}", f"# TYPE {self.name} histogram"]
        for bound, count in zip(self.buckets, self.counts):
            lines.append(f'{self.name}_bucket{{le="{bound}"}} {count}')
        lines.append(f'{self.name}_bucket{{le="+Inf"}} {self.count}')
        lines.append(f"{self.name}_sum {self.total}")
        lines.append(f"{self.name}_count {self.count}")
        return lines


class VoiceAgentMetrics:
    """Latency histograms across a server's Voice Agent sessions.

    Each ``LatencyReport`` a session sends is recorded under
    ``<prefix>_<latency name>``, e.g. ``stem_agent_total_latency``. The
    default prefix matches Deepgram's hosted Voice Agent, so dashboards built
    for it keep working.
    """

    LATENCIES = {
        "stt_latency": "Time the speech-to-text trails the audio received, in seconds.",
        "ttt_token_latency": "Time from the end of a user turn to the LLM's first token.",
        "ttt_text_latency": "Time from the end of a user turn to the LLM's first text token.",
        "tts_latency": "Time from the LLM's first text to the agent's first audio.",
        "total_latency": "Time from the end of a user turn to the agent's first audio.",
    }

    def __init__(self, prefix: str = "stem_agent", buckets: Sequence[float] = DEFAULT_BUCKETS):
        """Initialize the metrics.

        Args:
            prefix: Prefix of the metric names.
            buckets: Histogram bucket bounds, in seconds.
        """
        self._lock = threading.Lock()
        self._histograms = {
            name: _Histogram(f"{prefix}_{name}", help_text, buckets)
            for name, help_text in self.LATENCIES.items()
        }

    def observe(self, latency: str, value: float):
        """Record a latency.

        Args:
            latency: The latency's name, as in ``LatencyReport`` (e.g. ``tts_latency``).
            value: The latency, in seconds.
        """
        histogram = self._histograms.get(latency)
        if histogram is None:
            return
        with self._lock:
            histogram.observe(value)

    def render(self) -> str:
        """Render the metrics in the Prometheus text format.

        Returns:
            The metrics text.
        """
        with self._lock:
            lines = [line for h in self._histograms.values() for line in h.render()]
        return "\n".join(lines) + "\n"
