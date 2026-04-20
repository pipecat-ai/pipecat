"""Text formatting utilities for the turn-taking demo tool.

Provides ASCII timeline, per-turn comparison table, summary, and
per-analyzer timeline report formatting.
"""

import math
import os
import statistics
from typing import Dict, List, Optional, Tuple

from demo_types import (
    METHOD_ON_DEMAND,
    METHOD_STREAMING,
    METHOD_TIMEOUT,
    AnalyzerResult,
    TurnEvent,
)


def method_symbol(method: str) -> str:
    """Single-char symbol for a detection method."""
    return {"streaming": "S", "on-demand": "D", "timeout": "T"}.get(method, "?")


def _delay_label(delay: Optional[float]) -> str:
    """Qualitative label for a detection delay."""
    if delay is None:
        return ""
    if delay <= 0.001:
        return "instant*"
    if delay < 0.3:
        return "fast"
    if delay < 1.0:
        return "moderate"
    return "SLOW"


def format_ascii_timeline(
    duration_secs: float,
    speech_segments: List[Tuple[float, float]],
    results: Dict[str, AnalyzerResult],
    width: int = 80,
) -> str:
    """Render an ASCII timeline showing speech and turn markers.

    Args:
        duration_secs: Total audio duration in seconds.
        speech_segments: List of (start, end) tuples.
        results: Mapping of analyzer name to AnalyzerResult.
        width: Terminal width for the timeline bar.

    Returns:
        Multi-line ASCII timeline string.
    """
    label_width = max(len(n) for n in results) + 2 if results else 10
    label_width = max(label_width, 10)
    bar_width = width - label_width - 1

    def time_to_col(t: float) -> int:
        return min(int(t / duration_secs * bar_width), bar_width - 1)

    lines: List[str] = []
    lines.append("")
    lines.append("Visual Timeline:")

    # Time ruler
    tick_interval = 5
    ruler_label = " " * label_width
    ruler_ticks = " " * label_width
    for sec in range(0, int(duration_secs) + 1, tick_interval):
        col = time_to_col(sec)
        tag = f"{sec}"
        if col + len(tag) <= bar_width:
            ruler_label = (
                ruler_label[: label_width + col] + tag + ruler_label[label_width + col + len(tag) :]
            )
        ruler_ticks = ruler_ticks[: label_width + col] + "|" + ruler_ticks[label_width + col + 1 :]
    lines.append(f"  {'Time(s)':>{label_width - 2}}  {ruler_label[label_width:]}")
    lines.append(f"  {'':>{label_width - 2}}  {ruler_ticks[label_width:]}")

    # Speech row
    speech_bar = list("·" * bar_width)
    for start, end in speech_segments:
        c0 = time_to_col(start)
        c1 = time_to_col(end)
        for c in range(c0, min(c1 + 1, bar_width)):
            speech_bar[c] = "█"
    lines.append(f"  {'Speech':>{label_width - 2}}  {''.join(speech_bar)}")

    # Per-analyzer rows — position markers at effective time
    # (timestamp + vad_stop_secs) so the visual gap from speech end
    # reflects total latency, not just model processing time.
    for name, result in results.items():
        row = list("·" * bar_width)
        for event in result.turn_events:
            effective_time = event.timestamp + (event.vad_stop_secs or 0)
            col = time_to_col(effective_time)
            sym = method_symbol(event.method)
            if 0 <= col < bar_width:
                row[col] = sym
        lines.append(f"  {name:>{label_width - 2}}  {''.join(row)}")

    # Legend
    lines.append("")
    lines.append(
        f"  {'Legend:':>{label_width - 2}}  "
        "█=speech  S=streaming  D=on-demand  T=timeout  ·=silence"
    )
    lines.append("")

    return "\n".join(lines)


def format_comparison_table(
    speech_segments: List[Tuple[float, float]],
    results: Dict[str, AnalyzerResult],
) -> str:
    """Format a per-turn side-by-side comparison table.

    Matches each turn event to the nearest preceding speech segment
    and shows detection delay and verdict.
    """
    names = list(results.keys())
    if len(names) < 2:
        return ""

    seg_ends = [end for _, end in speech_segments]

    def find_speech_end(ts: float) -> Optional[float]:
        best = None
        for se in seg_ends:
            if se <= ts + 0.01:
                best = se
        return best

    all_anchors: set[float] = set()
    for result in results.values():
        for event in result.turn_events:
            anchor = find_speech_end(event.timestamp)
            if anchor is not None:
                all_anchors.add(anchor)
    anchors = sorted(all_anchors)

    anchor_events: Dict[float, Dict[str, Optional[TurnEvent]]] = {}
    for anchor in anchors:
        anchor_events[anchor] = {}
        for name, result in results.items():
            match = None
            for event in result.turn_events:
                ea = find_speech_end(event.timestamp)
                if ea is not None and abs(ea - anchor) < 0.01:
                    match = event
                    break
            anchor_events[anchor][name] = match

    lines: List[str] = []
    lines.append("")
    sep = "=" * 90
    lines.append(sep)
    lines.append("Per-Turn Comparison")
    lines.append(sep)
    lines.append("")

    col_w = 24
    hdr = f"  {'Turn':>4}  {'Speech End':>10}"
    for name in names:
        hdr += f"  {name:>{col_w}}"
    hdr += f"  {'Verdict':>{col_w}}"
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))

    for idx, anchor in enumerate(anchors, 1):
        row = f"  {idx:>4}  {anchor:>9.2f}s"

        delays: Dict[str, float] = {}
        for name in names:
            event = anchor_events[anchor].get(name)
            if event is None:
                row += f"  {'(missed)':>{col_w}}"
            else:
                td = event.total_delay
                m = method_symbol(event.method)
                if td is not None:
                    cell = f"{event.timestamp:.2f}s +{td:.2f}s [{m}]"
                else:
                    cell = f"{event.timestamp:.2f}s [?]"
                row += f"  {cell:>{col_w}}"
                delays[name] = td if td is not None else float("inf")

        if len(delays) >= 2:
            sorted_d = sorted(delays.items(), key=lambda x: x[1])
            fastest_name, fastest_val = sorted_d[0]
            second_val = sorted_d[1][1]
            delta = second_val - fastest_val
            if delta < 0.01:
                verdict = "tie"
            else:
                verdict = f"{fastest_name} +{delta:.2f}s faster"
        elif len(delays) == 1:
            only_name = list(delays.keys())[0]
            verdict = f"{only_name} (only)"
        else:
            verdict = ""
        row += f"  {verdict:>{col_w}}"
        lines.append(row)

    lines.append("")
    lines.append(
        "  [S]=streaming (real-time)  [D]=on-demand (at VAD stop)  [T]=timeout (silence fallback)"
    )
    lines.append(
        "  * Delays are total latency from estimated speech end."
        " [D]/[T] include VAD stop_secs wait before the analyzer is invoked."
    )
    lines.append(sep)

    return "\n".join(lines)


def format_summary(
    results: Dict[str, AnalyzerResult],
) -> str:
    """Format an enhanced comparison summary with consistency metrics."""
    if len(results) < 2:
        return ""

    lines: List[str] = []
    sep = "=" * 90
    lines.append("")
    lines.append(sep)
    lines.append("Comparison Summary")
    lines.append(sep)
    lines.append("")

    for name, result in results.items():
        n_turns = len(result.turn_events)
        total_delays = [e.total_delay for e in result.turn_events if e.total_delay is not None]

        methods: Dict[str, int] = {}
        for e in result.turn_events:
            methods[e.method] = methods.get(e.method, 0) + 1

        has_vad_wait = any(e.vad_stop_secs for e in result.turn_events)

        lines.append(f"  {name}:")
        lines.append(f"    Turns detected:  {n_turns}")
        lines.append(f"    Init time:       {result.init_time_ms:.0f}ms")

        if total_delays:
            avg = sum(total_delays) / len(total_delays)
            med = statistics.median(total_delays)
            std = math.sqrt(sum((d - avg) ** 2 for d in total_delays) / len(total_delays))
            lines.append(
                f"    Total latency: median={med:.3f}s  avg={avg:.3f}s"
                f"  min={min(total_delays):.3f}s  max={max(total_delays):.3f}s"
                f"  stddev={std:.3f}s"
            )
            if has_vad_wait:
                vad_secs = result.turn_events[0].vad_stop_secs
                lines.append(f"    (includes {vad_secs:.1f}s VAD wait before analyzer is invoked)")
            if len(total_delays) >= 3 and max(total_delays) > med * 2.5 and med > 0.01:
                lines.append(
                    f"    Note: max ({max(total_delays):.3f}s) >> median ({med:.3f}s)"
                    " -- likely VAD truncated speech, not analyzer latency"
                )
        else:
            lines.append("    Total latency: N/A")

        method_parts = []
        for m in [METHOD_STREAMING, METHOD_ON_DEMAND, METHOD_TIMEOUT]:
            if m in methods:
                method_parts.append(f"{methods[m]} {m}")
        lines.append(f"    Detection methods: {', '.join(method_parts)}")

        lines.append("")

    lines.append(sep)
    return "\n".join(lines)


def format_timeline(
    analyzer_name: str,
    input_path: str,
    sample_rate: int,
    duration_secs: float,
    threshold: float,
    frame_duration_ms: int,
    speech_segments: List[Tuple[float, float]],
    result: AnalyzerResult,
    viva_filter_used: bool = False,
) -> str:
    """Format a text timeline report for one analyzer."""
    lines: List[str] = []
    sep = "=" * 60

    lines.append(sep)
    lines.append(f"Turn-Taking Analysis: {analyzer_name}")
    lines.append(sep)
    lines.append(f"Audio: {os.path.basename(input_path)} ({sample_rate} Hz, {duration_secs:.2f}s)")
    lines.append(f"Threshold: {threshold}, Frame duration: {frame_duration_ms}ms")
    lines.append("VAD: Silero (stop_secs=0.2)")
    if viva_filter_used:
        lines.append("Noise filter: Krisp VIVA")
    lines.append(f"Analyzer init time: {result.init_time_ms:.1f}ms")
    lines.append("")

    lines.append("Speech Segments:")
    if speech_segments:
        for start, end in speech_segments:
            dur = end - start
            lines.append(f"  [{start:7.2f}s - {end:7.2f}s ] speech ({dur:.2f}s)")
    else:
        lines.append("  (none detected)")
    lines.append("")

    lines.append("Turn Boundaries Detected:")
    if result.turn_events:
        for i, event in enumerate(result.turn_events, 1):
            td = event.total_delay
            m = method_symbol(event.method)
            delay_str = ""
            if td is not None:
                label = _delay_label(td)
                if event.vad_stop_secs is not None:
                    d = event.detection_delay or 0.0
                    delay_str = (
                        f"  total: {td:.3f}s [{m}]"
                        f" ({event.vad_stop_secs:.1f}s VAD + {d:.3f}s model) {label}"
                    )
                else:
                    delay_str = f"  delay: {td:.3f}s [{m}] {label}"
            lines.append(f"  Turn {i}: {event.timestamp:7.2f}s{delay_str}")
    else:
        lines.append("  (none detected)")
    lines.append("")

    lines.append("Summary:")
    lines.append(f"  Turns detected: {len(result.turn_events)}")
    total_delays = [e.total_delay for e in result.turn_events if e.total_delay is not None]
    if total_delays:
        avg = sum(total_delays) / len(total_delays)
        med = statistics.median(total_delays)
        std = math.sqrt(sum((d - avg) ** 2 for d in total_delays) / len(total_delays))
        lines.append(f"  Total latency: median={med:.3f}s  avg={avg:.3f}s (stddev: {std:.3f}s)")
        lines.append(f"  Min: {min(total_delays):.3f}s  Max: {max(total_delays):.3f}s")
        if len(total_delays) >= 3 and max(total_delays) > med * 2.5 and med > 0.01:
            lines.append(
                f"  Note: max ({max(total_delays):.3f}s) >> median ({med:.3f}s)"
                " -- outlier likely due to VAD truncating speech"
            )

    methods: Dict[str, int] = {}
    for e in result.turn_events:
        methods[e.method] = methods.get(e.method, 0) + 1
    if methods:
        parts = [f"{v} {k}" for k, v in methods.items()]
        lines.append(f"  Detection methods: {', '.join(parts)}")

    lines.append("")
    lines.append("  [S]=streaming  [D]=on-demand  [T]=timeout")
    lines.append(
        "  * Total latency = time from estimated speech end to turn detection."
        " [D]/[T] include VAD stop_secs wait."
    )
    lines.append(sep)

    return "\n".join(lines)
