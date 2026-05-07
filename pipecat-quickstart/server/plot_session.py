#!/usr/bin/env python3
"""Plot a session JSONL log produced by ConversationLogObserver in bot_dev.py.

Usage::

    python plot_session.py logs/session_<ts>_<id>.jsonl
    python plot_session.py logs/session_<ts>_<id>.jsonl --save plot.png

Three vertically stacked subplots share the same time axis (seconds from
pipeline start):
  1. Bot state  — step plot over 5 categorical states
  2. Bot speech — boolean filled step (speaking / silent)
  3. Turn probability — scatter + line from the smart-turn model
"""

import argparse
import json
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOT_STATES = ["idle", "listening", "user_speaking", "processing", "speaking"]
STATE_COLORS = {
    "idle": "#aaaaaa",
    "listening": "#4fc3f7",
    "user_speaking": "#81c784",
    "processing": "#ffb74d",
    "speaking": "#e57373",
}


# ---------------------------------------------------------------------------
# Loading & deduplication
# ---------------------------------------------------------------------------

def load_events(path: Path) -> tuple[list[dict], dict]:
    """Load JSONL, deduplicate, and report anomalies.

    Returns:
        events: deduplicated list of event dicts (with 'ts_s' added)
        report: dict summarising duplication diagnostics
    """
    raw: list[dict] = []

    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  WARNING line {lineno}: JSON parse error — {exc}", file=sys.stderr)

    # --- exact duplicates (every field identical) --------------------------
    seen_exact: dict[str, int] = {}
    exact_dup_count = 0
    deduped: list[dict] = []
    for ev in raw:
        key = json.dumps(ev, sort_keys=True)
        if key in seen_exact:
            exact_dup_count += 1
        else:
            seen_exact[key] = 1
            deduped.append(ev)

    # --- same (ts_ns, event) with *different* payload ----------------------
    # Group by (ts_ns, event) and look for payload variance
    bucket: dict[tuple, list[dict]] = defaultdict(list)
    for ev in deduped:
        bucket[(ev.get("ts_ns"), ev.get("event"))].append(ev)

    conflict_groups: list[list[dict]] = []
    for (ts, evt), group in bucket.items():
        if len(group) > 1:
            # Check whether all payloads are identical after removing ts_ns
            payloads = [
                {k: v for k, v in e.items() if k not in ("ts_ns", "event")}
                for e in group
            ]
            if len({json.dumps(p, sort_keys=True) for p in payloads}) > 1:
                conflict_groups.append(group)

    # Add seconds column
    for ev in deduped:
        ev["ts_s"] = ev["ts_ns"] / 1e9

    # Guard: remove events with timestamps beyond 10 minutes — these are
    # almost certainly wall-clock Unix ns accidentally used as pipeline offsets.
    MAX_SESSION_S = 600.0
    outliers = [ev for ev in deduped if ev["ts_s"] > MAX_SESSION_S]
    deduped = [ev for ev in deduped if ev["ts_s"] <= MAX_SESSION_S]

    report = {
        "raw_count": len(raw),
        "exact_dup_count": exact_dup_count,
        "after_dedup": len(deduped),
        "conflict_groups": conflict_groups,
        "outlier_count": len(outliers),
        "outlier_events": outliers,
    }
    return deduped, report


def print_quality_report(report: dict, path: Path) -> None:
    raw = report["raw_count"]
    exact = report["exact_dup_count"]
    conflicts = report["conflict_groups"]
    after = report["after_dedup"]
    outliers = report.get("outlier_count", 0)
    outlier_events = report.get("outlier_events", [])

    issues = exact > 0 or len(conflicts) > 0 or outliers > 0
    tag = "WARNING" if issues else "OK"

    print(f"\n=== Data-quality report for {path.name} [{tag}] ===")
    print(f"  Raw lines:            {raw}")
    print(f"  Exact duplicates:     {exact}  (removed)")
    print(f"  Events after dedup:   {after}")

    if outliers > 0:
        pct = 100 * outliers / raw
        print(
            f"\n  ⚠  {outliers} event(s) ({pct:.1f}% of raw) had ts_ns > 10 min "
            "and were removed (likely wall-clock Unix ns used as pipeline offset)."
        )

    if exact > 0:
        pct = 100 * exact / raw
        print(
            f"\n  ⚠  {exact} exact-duplicate lines ({pct:.1f}% of raw). "
            "This likely means on_push_frame fires multiple times per frame "
            "(once per pipeline segment boundary). Consider deduplicating at "
            "the observer level."
        )

    if conflicts:
        print(
            f"\n  ⚠  {len(conflicts)} group(s) share the same (ts_ns, event) "
            "but carry DIFFERENT payloads — possible logging race or "
            "frame-reuse issue:"
        )
        for i, group in enumerate(conflicts[:5], 1):  # show at most 5
            print(f"    Group {i} ({group[0]['event']} @ ts_ns={group[0]['ts_ns']}):")
            for ev in group:
                payload = {k: v for k, v in ev.items() if k not in ("ts_ns", "event")}
                print(f"      {payload}")
        if len(conflicts) > 5:
            print(f"    ... and {len(conflicts) - 5} more group(s).")

    print()


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def build_state_series(events: list[dict]) -> tuple[list[float], list[int]]:
    """Return (times, state_indices) as a step series."""
    state_events = [e for e in events if e["event"] == "bot_state_changed"]
    # Prepend t=0 with "idle"
    times = [0.0] + [e["ts_s"] for e in state_events]
    states = ["idle"] + [e["state"] for e in state_events]
    indices = [BOT_STATES.index(s) for s in states]
    return times, indices


def build_speech_series(events: list[dict]) -> tuple[list[float], list[int]]:
    """Return (times, active_int) as a boolean step series."""
    speech_events = [e for e in events if e["event"] == "bot_speech_changed"]
    times = [0.0] + [e["ts_s"] for e in speech_events]
    active = [0] + [int(e["active"]) for e in speech_events]
    return times, active


def build_turn_prob_series(events: list[dict]) -> tuple[list[float], list[float], list[bool]]:
    """Return (times, probabilities, is_complete_flags)."""
    tp_events = [e for e in events if e["event"] == "turn_probability"]
    times = [e["ts_s"] for e in tp_events]
    probs = [e["probability"] for e in tp_events]
    complete = [e.get("is_complete", False) for e in tp_events]
    return times, probs, complete


def build_transcript_series(events: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (user_turns, bot_turns) as lists of {start_s, end_s, text}."""
    user_turns = [
        {
            "start_s": e["start_ts_ns"] / 1e9,
            "end_s": e["end_ts_ns"] / 1e9,
            "text": e.get("text", ""),
        }
        for e in events if e["event"] == "user_turn"
    ]
    bot_turns = [
        {
            "start_s": e["start_ts_ns"] / 1e9,
            "end_s": e["end_ts_ns"] / 1e9,
            "text": e.get("text", ""),
        }
        for e in events if e["event"] == "bot_turn"
    ]
    return user_turns, bot_turns


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def plot(events: list[dict], title: str, save_path: str | None = None) -> None:
    t_max = max((e["ts_s"] for e in events), default=1.0) * 1.05

    fig, axes = plt.subplots(
        4, 1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1, 2, 3], "hspace": 0.08},
    )

    # ---- Subplot 1: bot state --------------------------------------------
    ax_state = axes[0]
    state_times, state_idx = build_state_series(events)

    # Draw colored background bands for each segment
    for i in range(len(state_times)):
        t0 = state_times[i]
        t1 = state_times[i + 1] if i + 1 < len(state_times) else t_max
        s = BOT_STATES[state_idx[i]]
        ax_state.axvspan(t0, t1, color=STATE_COLORS[s], alpha=0.25)

    ax_state.step(state_times, state_idx, where="post", color="steelblue", linewidth=1.5)
    ax_state.set_yticks(range(len(BOT_STATES)))
    ax_state.set_yticklabels(BOT_STATES, fontsize=8)
    ax_state.set_ylabel("Bot state", fontsize=9)
    ax_state.set_ylim(-0.5, len(BOT_STATES) - 0.5)
    ax_state.grid(axis="x", linestyle=":", alpha=0.4)

    # Legend patches
    patches = [mpatches.Patch(color=c, alpha=0.5, label=s) for s, c in STATE_COLORS.items()]
    ax_state.legend(handles=patches, fontsize=7, loc="upper right", ncol=5)

    # ---- Subplot 2: bot speech -------------------------------------------
    ax_speech = axes[1]
    sp_times, sp_active = build_speech_series(events)
    ax_speech.step(sp_times, sp_active, where="post", color="#e57373", linewidth=1.5)
    ax_speech.fill_between(sp_times, sp_active, step="post", color="#e57373", alpha=0.25)
    ax_speech.set_yticks([0, 1])
    ax_speech.set_yticklabels(["silent", "speaking"], fontsize=8)
    ax_speech.set_ylabel("Bot speech", fontsize=9)
    ax_speech.set_ylim(-0.1, 1.4)
    ax_speech.grid(axis="x", linestyle=":", alpha=0.4)

    # ---- Subplot 3: turn probability ------------------------------------
    ax_tp = axes[2]
    tp_times, tp_probs, tp_complete = build_turn_prob_series(events)

    if tp_times:
        # Line connecting all points
        ax_tp.plot(tp_times, tp_probs, color="gray", linewidth=0.8, zorder=1)
        # Scatter: colour by is_complete
        colors = ["#e57373" if c else "#81c784" for c in tp_complete]
        ax_tp.scatter(tp_times, tp_probs, c=colors, s=40, zorder=2)
        # 0.5 threshold line
        ax_tp.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

        # Legend
        inc_patch = mpatches.Patch(color="#81c784", label="incomplete")
        cmp_patch = mpatches.Patch(color="#e57373", label="is_complete=True")
        ax_tp.legend(handles=[inc_patch, cmp_patch], fontsize=7, loc="upper right")
    else:
        ax_tp.text(
            0.5, 0.5, "No turn_probability events",
            ha="center", va="center", transform=ax_tp.transAxes, fontsize=9,
        )

    ax_tp.set_ylabel("Turn probability", fontsize=9)
    ax_tp.set_ylim(-0.05, 1.05)
    ax_tp.grid(axis="x", linestyle=":", alpha=0.4)

    # ---- Subplot 4: transcripts -----------------------------------------
    ax_text = axes[3]
    user_turns, bot_turns = build_transcript_series(events)

    # Bars are static — draw once
    rng = np.random.default_rng(42)
    user_jitters = rng.uniform(-0.12, 0.12, size=max(len(user_turns), 1)).tolist()
    bot_jitters = rng.uniform(-0.12, 0.12, size=max(len(bot_turns), 1)).tolist()

    for i, turn in enumerate(user_turns):
        width = max(turn["end_s"] - turn["start_s"], 0.1)
        ax_text.barh(1 + user_jitters[i], width, left=turn["start_s"], height=0.5, color="#81c784", alpha=0.4)

    for i, turn in enumerate(bot_turns):
        width = max(turn["end_s"] - turn["start_s"], 0.1)
        ax_text.barh(0 + bot_jitters[i], width, left=turn["start_s"], height=0.5, color="#e57373", alpha=0.4)

    ax_text.set_yticks([0, 1])
    ax_text.set_yticklabels(["bot", "user"], fontsize=8)
    ax_text.set_ylabel("Transcript", fontsize=9)
    ax_text.set_ylim(-0.6, 1.8)
    ax_text.set_xlabel("Time (s)", fontsize=9)
    ax_text.grid(axis="x", linestyle=":", alpha=0.4)

    # Text labels are redrawn on resize so wrapping adapts to the figure width
    transcript_texts: list = []

    def _redraw_transcript_labels() -> None:
        for txt in transcript_texts:
            txt.remove()
        transcript_texts.clear()

        try:
            ax_px = ax_text.get_window_extent().width
        except Exception:
            ax_px = 0

        xlim = ax_text.get_xlim()
        data_range = (xlim[1] - xlim[0]) or 1.0
        # Approximate: fontsize 6.5 pt ≈ 0.55 px per character (screen DPI)
        chars_per_unit = (ax_px / data_range) / (6.5 * 0.55) if ax_px > 0 else 120.0 / t_max

        for i, turn in enumerate(user_turns):
            w = max(turn["end_s"] - turn["start_s"], 0.1)
            transcript_texts.append(ax_text.text(
                turn["start_s"] + w / 2, 1 + user_jitters[i],
                textwrap.fill(turn["text"], width=max(5, int(w * chars_per_unit * 0.75))),
                ha="center", va="center", fontsize=6.5, clip_on=True,
            ))

        for i, turn in enumerate(bot_turns):
            w = max(turn["end_s"] - turn["start_s"], 0.1)
            transcript_texts.append(ax_text.text(
                turn["start_s"] + w / 2, 0 + bot_jitters[i],
                textwrap.fill(turn["text"], width=max(5, int(w * chars_per_unit*0.75))),
                ha="center", va="center", fontsize=6.5, clip_on=True,
            ))

        if not user_turns and not bot_turns:
            transcript_texts.append(ax_text.text(
                0.5, 0.5, "No transcript events (run bot again to capture)",
                ha="center", va="center", transform=ax_text.transAxes, fontsize=9,
            ))

    _redraw_transcript_labels()

    if not save_path:
        fig.canvas.mpl_connect(
            "resize_event",
            lambda _e: (_redraw_transcript_labels(), fig.canvas.draw_idle()),
        )

        # ---- Title & layout --------------------------------------------------
        fig.suptitle(title, fontsize=11, y=1.01)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved plot to {save_path}")
        else:
            plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a pipecat session JSONL log.")
    parser.add_argument("logfile", nargs="?", default=None, help="Path to the session .jsonl file")
    parser.add_argument("--save", metavar="PATH", help="Save plot to file instead of displaying")
    args = parser.parse_args()

# if logfile is None, interpret it as 0, meaning "the most recent log file in logs/". Likewise, interpret -1 as the next most recent, and so on. This allows quick plotting of the latest session without needing to copy-paste the filename.
    if args.logfile is None:
        log_idx = 0
    elif args.logfile.lstrip("-").isdigit():
        log_idx = -int(args.logfile)
    else: 
        path = Path(args.logfile)
        log_idx = None
      
    if log_idx is not None:
        log_dir = Path("logs")
        if not log_dir.exists() or not log_dir.is_dir():
            print(f"ERROR: logs/ directory not found in current path", file=sys.stderr)
            sys.exit(1)
        log_files = sorted(log_dir.glob("session_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not log_files:
            print(f"ERROR: no session_*.jsonl files found in logs/", file=sys.stderr)
            sys.exit(1)
        path = log_files[log_idx]

        
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    events, report = load_events(path)
    print_quality_report(report, path)

    title = f"Session: {path.stem}"
    plot(events, title=title, save_path=args.save)


if __name__ == "__main__":
    main()
