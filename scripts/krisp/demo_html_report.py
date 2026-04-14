"""HTML report generation for the turn-taking demo tool.

Produces a self-contained HTML file with an interactive timeline,
summary cards, and a per-turn comparison table.
"""

import math
import os
from typing import Dict, List, Tuple

from demo_types import AnalyzerResult


def generate_html_report(
    input_path: str,
    duration_secs: float,
    sample_rate: int,
    speech_segments: List[Tuple[float, float]],
    results: Dict[str, AnalyzerResult],
    viva_filter_used: bool,
    output_path: str,
) -> None:
    """Generate a self-contained HTML report with interactive timeline.

    Args:
        input_path: Path to the source audio file (used for display name).
        duration_secs: Total audio duration in seconds.
        sample_rate: Audio sample rate in Hz.
        speech_segments: List of (start, end) speech segment tuples.
        results: Mapping of analyzer name to AnalyzerResult.
        viva_filter_used: Whether the VIVA noise filter was applied.
        output_path: Destination path for the HTML file.
    """
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0"]

    seg_json = ", ".join(f"[{s:.3f},{e:.3f}]" for s, e in speech_segments)

    analyzer_blocks = []
    for idx, (name, result) in enumerate(results.items()):
        color = colors[idx % len(colors)]
        events_json = ", ".join(
            f'{{"t":{e.timestamp:.3f},'
            f'"d":{e.detection_delay if e.detection_delay is not None else 0:.3f},'
            f'"m":"{e.method}"}}'
            for e in result.turn_events
        )
        delays = [
            e.detection_delay for e in result.turn_events if e.detection_delay is not None
        ]
        avg_d = sum(delays) / len(delays) if delays else 0
        std_d = (
            math.sqrt(sum((d - avg_d) ** 2 for d in delays) / len(delays)) if delays else 0
        )
        analyzer_blocks.append(
            f'{{"name":"{name}","color":"{color}","init_ms":{result.init_time_ms:.1f},'
            f'"turns":{len(result.turn_events)},"avg_delay":{avg_d:.3f},'
            f'"std_delay":{std_d:.3f},"events":[{events_json}]}}'
        )
    analyzers_json = ",\n      ".join(analyzer_blocks)

    basename = os.path.basename(input_path)
    filter_note = '  &mdash; Noise filter: Krisp VIVA' if viva_filter_used else ''

    html = _HTML_TEMPLATE.format(
        basename=basename,
        duration_secs=duration_secs,
        sample_rate=sample_rate,
        filter_note=filter_note,
        seg_json=seg_json,
        analyzers_json=analyzers_json,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# HTML template (kept as a module-level constant to avoid cluttering the
# function body with hundreds of lines of markup).
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Turn-Taking Analysis: {basename}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #1a1a2e; color: #eee; padding: 24px; }}
  h1 {{ font-size: 1.4em; margin-bottom: 4px; color: #fff; }}
  .meta {{ color: #888; font-size: 0.85em; margin-bottom: 20px; }}
  .timeline-container {{ background: #16213e; border-radius: 8px; padding: 16px 20px;
                         margin-bottom: 20px; overflow-x: auto; }}
  .timeline-row {{ display: flex; align-items: center; margin: 4px 0; height: 28px; }}
  .timeline-label {{ width: 140px; min-width: 140px; text-align: right; padding-right: 12px;
                     font-size: 0.85em; color: #aaa; white-space: nowrap; }}
  .timeline-track {{ position: relative; flex: 1; height: 100%; background: #0f3460;
                     border-radius: 4px; }}
  .speech-bar {{ position: absolute; height: 100%; background: #4ecca3; border-radius: 3px;
                 opacity: 0.8; }}
  .turn-marker {{ position: absolute; width: 3px; height: 100%; border-radius: 2px;
                  cursor: pointer; }}
  .turn-marker:hover {{ filter: brightness(1.4); }}
  .turn-marker .tooltip {{ display: none; position: absolute; bottom: 110%; left: 50%;
    transform: translateX(-50%); background: #222; color: #fff; padding: 6px 10px;
    border-radius: 4px; font-size: 0.75em; white-space: nowrap; z-index: 10;
    pointer-events: none; box-shadow: 0 2px 8px rgba(0,0,0,.5); }}
  .turn-marker:hover .tooltip {{ display: block; }}
  .ruler {{ position: relative; height: 20px; }}
  .ruler-tick {{ position: absolute; top: 0; font-size: 0.7em; color: #666;
                 transform: translateX(-50%); }}
  .ruler-line {{ position: absolute; top: 14px; width: 1px; height: 6px; background: #444; }}
  .summary {{ display: flex; gap: 16px; flex-wrap: wrap; margin-top: 16px; }}
  .card {{ background: #16213e; border-radius: 8px; padding: 16px 20px; flex: 1;
           min-width: 260px; }}
  .card h3 {{ font-size: 1em; margin-bottom: 10px; }}
  .card .stat {{ display: flex; justify-content: space-between; padding: 3px 0;
                 font-size: 0.85em; }}
  .card .stat .label {{ color: #888; }}
  .card .warn {{ color: #ffb74d; font-size: 0.8em; margin-top: 8px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 0.85em; }}
  th {{ text-align: left; padding: 8px 10px; border-bottom: 2px solid #333; color: #aaa;
       font-weight: 600; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #222; }}
  .faster {{ color: #4ecca3; }}
  .slower {{ color: #ff6b6b; }}
  .method {{ display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 0.75em;
             font-weight: 600; }}
  .method-streaming {{ background: #1b5e20; color: #a5d6a7; }}
  .method-on-demand {{ background: #e65100; color: #ffcc80; }}
  .method-timeout {{ background: #b71c1c; color: #ef9a9a; }}
  footer {{ margin-top: 24px; color: #555; font-size: 0.75em; text-align: center; }}
</style>
</head>
<body>
<h1>Turn-Taking Analysis</h1>
<p class="meta">{basename} &mdash; {duration_secs:.2f}s, {sample_rate} Hz &mdash; \
VAD: Silero (stop_secs=0.2){filter_note}</p>

<div class="timeline-container" id="tl"></div>
<div class="summary" id="cards"></div>
<div class="timeline-container" style="margin-top:20px">
  <h3 style="margin-bottom:12px;font-size:1em;">Per-Turn Comparison</h3>
  <table id="cmp"></table>
</div>
<footer>Generated by demo_turn_taking.py</footer>

<script>
const DUR = {duration_secs:.3f};
const SPEECH = [{seg_json}];
const ANALYZERS = [
      {analyzers_json}
];

const tl = document.getElementById('tl');
const cards = document.getElementById('cards');
const cmp = document.getElementById('cmp');

function pct(t) {{ return (t / DUR * 100).toFixed(4) + '%'; }}
function mclass(m) {{ return 'method-' + m.replace(' ','-').replace('on-demand','on-demand'); }}
function msym(m) {{ return m === 'streaming' ? 'S' : m === 'on-demand' ? 'D' : 'T'; }}

// Ruler
let ruler = '<div class="timeline-row"><div class="timeline-label"></div>' +
  '<div class="timeline-track ruler">';
for (let s = 0; s <= DUR; s += 5) {{
  ruler += '<span class="ruler-tick" style="left:' + pct(s) + '">' + s + 's</span>';
  ruler += '<span class="ruler-line" style="left:' + pct(s) + '"></span>';
}}
ruler += '</div></div>';
tl.innerHTML = ruler;

// Speech row
let sr = '<div class="timeline-row"><div class="timeline-label">Speech</div>' +
  '<div class="timeline-track">';
SPEECH.forEach(s => {{
  sr += '<div class="speech-bar" style="left:' + pct(s[0]) + ';width:' +
        pct(s[1]-s[0]) + '"></div>';
}});
sr += '</div></div>';
tl.innerHTML += sr;

// Analyzer rows
ANALYZERS.forEach(a => {{
  let row = '<div class="timeline-row"><div class="timeline-label" style="color:' +
    a.color + '">' + a.name + '</div><div class="timeline-track">';
  a.events.forEach(e => {{
    row += '<div class="turn-marker" style="left:' + pct(e.t) + ';background:' +
      a.color + '"><div class="tooltip">' + e.t.toFixed(2) + 's &mdash; delay: ' +
      e.d.toFixed(3) + 's [' + msym(e.m) + ']</div></div>';
  }});
  row += '</div></div>';
  tl.innerHTML += row;
}});

// Summary cards
ANALYZERS.forEach(a => {{
  let warn = '';
  const instant = a.events.filter(e => e.d < 0.001).length;
  if (instant > a.turns * 0.5 && a.turns > 1)
    warn = '<div class="warn">&#9888; ' + instant + '/' + a.turns +
      ' turns detected at 0ms \\u2014 analyzer confirms VAD without independent analysis</div>';
  const methods = {{}};
  a.events.forEach(e => {{ methods[e.m] = (methods[e.m]||0) + 1; }});
  let mstr = Object.entries(methods).map(([k,v]) => v + ' ' + k).join(', ');
  cards.innerHTML += '<div class="card"><h3 style="color:' + a.color + '">' + a.name + '</h3>'
    + '<div class="stat"><span class="label">Turns</span><span>' + a.turns + '</span></div>'
    + '<div class="stat"><span class="label">Avg delay</span><span>' +
        a.avg_delay.toFixed(3) + 's</span></div>'
    + '<div class="stat"><span class="label">Stddev</span><span>' +
        a.std_delay.toFixed(3) + 's</span></div>'
    + '<div class="stat"><span class="label">Init time</span><span>' +
        a.init_ms.toFixed(0) + 'ms</span></div>'
    + '<div class="stat"><span class="label">Methods</span><span>' + mstr + '</span></div>'
    + warn + '</div>';
}});

// Comparison table
if (ANALYZERS.length >= 2) {{
  const segEnds = SPEECH.map(s => s[1]);
  function findAnchor(t) {{
    let best = null;
    segEnds.forEach(se => {{ if (se <= t + 0.01) best = se; }});
    return best;
  }}
  const anchors = new Set();
  ANALYZERS.forEach(a => a.events.forEach(e => {{
    const an = findAnchor(e.t); if (an !== null) anchors.add(an);
  }}));
  const sorted = [...anchors].sort((a,b) => a-b);

  let th = '<tr><th>#</th><th>Speech End</th>';
  ANALYZERS.forEach(a => {{
    th += '<th style="color:' + a.color + '">' + a.name + '</th>';
  }});
  th += '<th>Verdict</th></tr>';
  cmp.innerHTML = th;

  sorted.forEach((anchor, idx) => {{
    let row = '<tr><td>' + (idx+1) + '</td><td>' + anchor.toFixed(2) + 's</td>';
    const delays = {{}};
    ANALYZERS.forEach(a => {{
      const ev = a.events.find(e => {{
        const ea = findAnchor(e.t);
        return ea !== null && Math.abs(ea - anchor) < 0.01;
      }});
      if (!ev) {{ row += '<td style="color:#555">(missed)</td>'; return; }}
      const mc = mclass(ev.m);
      row += '<td>' + ev.t.toFixed(2) + 's <span style="color:#888">+' +
        ev.d.toFixed(2) + 's</span> <span class="method ' + mc + '">' +
        msym(ev.m) + '</span></td>';
      delays[a.name] = ev.d;
    }});

    const entries = Object.entries(delays).sort((a,b) => a[1]-b[1]);
    let verdict = '';
    if (entries.length >= 2) {{
      const delta = entries[1][1] - entries[0][1];
      if (delta < 0.01) verdict = '<span style="color:#888">tie</span>';
      else verdict = '<span class="faster">' + entries[0][0] + '</span>' +
        ' <span style="color:#888">+' + delta.toFixed(2) + 's faster</span>';
    }}
    row += '<td>' + verdict + '</td></tr>';
    cmp.innerHTML += row;
  }});
}}
</script>
</body>
</html>"""
