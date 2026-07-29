"""Chart rendering for the transport latency benchmark.

Reads per-trial JSON from ``results/`` and writes PNGs:

- ``chart-all.png`` — every scenario with results, in registry order.
- ``chart-<group>.png`` — one per chart group (local-direct, local-relay,
  deployed-relay, cloud), pairing the WebRTC and MoQ scenario at that tier.

Each bar is a scenario's raw p50 RTT, split into the client stack's
in-process floor (gray, from ``floor-<transport>.json``) and the excess over
it (colored by transport) — so bars are honest end-to-end numbers while the
solid segments stay comparable across stacks. Whiskers mark p95; dots mark
individual trial p50s. Daily has no in-process floor (media always traverses
Daily infra), so its bar is a single solid segment.

Standalone: uv run python benchmarks/transport_latency/charts.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from scenarios import GROUPS, SCENARIOS, scenario_order

# Reference dataviz palette (light mode), validated: CVD/normal-vision floors
# pass all-pairs for these three slots; aqua's sub-3:1 surface contrast is
# relieved by the direct value labels on every bar.
TRANSPORT_COLOR = {"webrtc": "#2a78d6", "moq": "#eb6834", "daily": "#1baf7a"}
FLOOR_COLOR = "#c3c2b7"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def _load(results_dir: Path) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    """Trial rows grouped by scenario slug, and floor stats by transport."""
    by_scenario: dict[str, list[dict]] = {}
    floors: dict[str, dict] = {}
    for f in sorted(results_dir.glob("*.json")):
        data = json.loads(f.read_text())
        if data.get("kind") == "floor":
            floors[data["transport"]] = data["stats"]
        elif "stats" in data and data.get("scenario") in SCENARIOS:
            # Glass-to-glass runs (browser method "a") include OS audio I/O and
            # are never charted against the network-path numbers.
            if data.get("browser_method") == "a":
                continue
            if data["stats"].get("n"):
                by_scenario.setdefault(data["scenario"], []).append(data)
    for trials in by_scenario.values():
        trials.sort(key=lambda r: r.get("trial", 0))
    return by_scenario, floors


def _bar_data(slug: str, trials: list[dict], floors: dict[str, dict]) -> dict:
    scenario = SCENARIOS[slug]
    p50s = [t["stats"]["p50_ms"] for t in trials]
    p95s = [t["stats"]["p95_ms"] for t in trials]
    total = float(np.median(p50s))
    floor_stats = floors.get(scenario.floor_key or "", {})
    floor = min(float(floor_stats["p50_ms"]), total) if floor_stats.get("p50_ms") else 0.0
    return {
        "slug": slug,
        "transport": scenario.transport,
        "client": scenario.client,
        "floor": floor,
        "excess": total - floor,
        "total": total,
        "p95": float(np.median(p95s)),
        "trial_p50s": p50s,
        "n": sum(t["stats"].get("n", 0) for t in trials),
        "drops": sum(t.get("drops", 0) for t in trials),
    }


def _render(bars: list[dict], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(
        figsize=(max(4.5, 1.7 * len(bars) + 1.6), 4.6), dpi=150, layout="constrained"
    )
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    x = np.arange(len(bars))
    width = 0.56
    y_span = max(max(b["p95"], *b["trial_p50s"]) for b in bars)
    for i, b in enumerate(bars):
        color = TRANSPORT_COLOR[b["transport"]]
        if b["floor"] > 0:
            ax.bar(i, b["floor"], width, color=FLOOR_COLOR, edgecolor=SURFACE, linewidth=1.5)
            # Label the floor value inside its segment when it fits.
            if b["floor"] > 0.12 * y_span:
                ax.annotate(
                    f"floor {b['floor']:.0f} ms",
                    (i, b["floor"] / 2),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=INK_SECONDARY,
                )
        ax.bar(
            i,
            b["excess"],
            width,
            bottom=b["floor"],
            color=color,
            edgecolor=SURFACE,
            linewidth=1.5,
            # Browser-measured bars are hatched: same transport color, but the
            # measuring stack is Chrome, so they only compare with each other.
            hatch="//" if b.get("client") == "browser" else None,
        )
        # p95 whisker from the bar top.
        ax.plot([i, i], [b["total"], b["p95"]], color=INK_SECONDARY, linewidth=1.2)
        ax.plot([i - 0.07, i + 0.07], [b["p95"], b["p95"]], color=INK_SECONDARY, linewidth=1.2)
        # Individual trial p50s, ringed with the surface color.
        ax.scatter(
            [i] * len(b["trial_p50s"]),
            b["trial_p50s"],
            s=22,
            color=color,
            edgecolor=SURFACE,
            linewidth=1.2,
            zorder=3,
        )
        label_y = max(b["p95"], max(b["trial_p50s"], default=0))
        ax.annotate(
            f"{b['total']:.0f} ms",
            (i, label_y),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=INK,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{b['slug']}\nn={b['n']}  drops={b['drops']}" for b in bars],
        fontsize=8.5,
        color=INK_SECONDARY,
    )
    ax.set_ylabel("round-trip latency, p50 (ms)", fontsize=9, color=INK_SECONDARY)
    ax.set_ylim(bottom=0)
    ax.margins(y=0.14)

    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, length=0)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=TRANSPORT_COLOR[t])
        for t in TRANSPORT_COLOR
        if any(b["transport"] == t for b in bars)
    ]
    labels = [t for t in TRANSPORT_COLOR if any(b["transport"] == t for b in bars)]
    floor_by_transport = {
        b["transport"]: b["floor"] for b in bars if b["floor"] > 0
    }  # identical for every bar of a stack
    if floor_by_transport:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=FLOOR_COLOR))
        values = " · ".join(f"{t} {f:.3g} ms" for t, f in floor_by_transport.items())
        labels.append(f"client-stack floor ({values})")
    fig.suptitle(title, fontsize=11, color=INK)
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncols=len(labels),
        frameon=False,
        fontsize=8.5,
        labelcolor=INK_SECONDARY,
    )

    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def _render_series(trials_by_slug: dict[str, list[dict]], title: str, out_path: Path) -> None:
    """Per-marker RTT over trial time, one line per scenario (first trial).

    Shows what the aggregate bars hide: jitter-buffer creep, periodic
    spikes, and settling after the warmup cut.
    """
    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=150, layout="constrained")
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    plotted = False
    for slug, trials in trials_by_slug.items():
        series = trials[0].get("series") or []
        if not series:
            continue
        t = [p[0] for p in series]
        rtt = [p[1] for p in series]
        scenario = SCENARIOS[slug]
        ax.plot(
            t,
            rtt,
            color=TRANSPORT_COLOR[scenario.transport],
            linewidth=1.1,
            linestyle="--" if scenario.client == "browser" else "-",
            marker=".",
            markersize=2.5,
            label=slug,
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("time since trial start (s)", fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel("round-trip latency (ms)", fontsize=9, color=INK_SECONDARY)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, length=0)
    fig.suptitle(title, fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def render_charts(results_dir: Path) -> list[Path]:
    """Render chart-all.png plus per-group bar and time-series PNGs."""
    by_scenario, floors = _load(results_dir)
    if not by_scenario:
        return []
    slugs = sorted(by_scenario, key=scenario_order)
    bars = {slug: _bar_data(slug, by_scenario[slug], floors) for slug in slugs}

    written: list[Path] = []
    out = results_dir / "chart-all.png"
    _render([bars[s] for s in slugs], "Audio round-trip latency — all scenarios", out)
    written.append(out)

    for group, group_title in GROUPS.items():
        group_slugs = [s for s in slugs if SCENARIOS[s].group == group]
        if not group_slugs:
            continue
        out = results_dir / f"chart-{group}.png"
        _render(
            [bars[s] for s in group_slugs],
            f"Audio round-trip latency — {group_title}",
            out,
        )
        written.append(out)
        if any(by_scenario[s][0].get("series") for s in group_slugs):
            out = results_dir / f"chart-series-{group}.png"
            _render_series(
                {s: by_scenario[s] for s in group_slugs},
                f"RTT over time — {group_title}",
                out,
            )
            written.append(out)
    return written


if __name__ == "__main__":
    results = Path(__file__).parent / "results"
    paths = render_charts(results)
    if not paths:
        print(f"no trial results in {results}")
    for p in paths:
        print(f"chart -> {p}")
