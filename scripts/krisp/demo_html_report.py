"""HTML report generation for the turn-taking demo tool.

Produces a self-contained HTML file with an interactive timeline,
summary cards, per-turn comparison table, and embedded audio players.
"""

import base64
import io
import math
import os
import statistics
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from demo_types import METHOD_TIMEOUT, AnalyzerResult


def _encode_wav_base64(audio: np.ndarray, sample_rate: int) -> str:
    """Encode int16 audio as a base64 WAV data URI."""
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def generate_html_report(
    input_path: str,
    duration_secs: float,
    sample_rate: int,
    speech_segments: List[Tuple[float, float]],
    results: Dict[str, AnalyzerResult],
    viva_filter_used: bool,
    output_path: str,
    annotated_audio: Optional[Dict[str, np.ndarray]] = None,
    vad_info: str = "Silero",
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
        annotated_audio: Optional mapping of analyzer name to annotated
            int16 audio arrays. When provided, audio players are embedded.
        vad_info: Human-readable VAD configuration summary for the report.
    """
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0"]

    seg_json = ", ".join(f"[{s:.3f},{e:.3f}]" for s, e in speech_segments)

    # Build per-analyzer audio data URIs
    audio_uris: Dict[str, str] = {}
    if annotated_audio:
        for name, samples in annotated_audio.items():
            audio_uris[name] = _encode_wav_base64(samples, sample_rate)

    analyzer_blocks = []
    for idx, (name, result) in enumerate(results.items()):
        color = colors[idx % len(colors)]
        events_json = ", ".join(
            f'{{"t":{e.timestamp:.3f},'
            f'"d":{e.total_delay if e.total_delay is not None else 0:.3f},'
            f'"vad":{e.vad_stop_secs if e.vad_stop_secs is not None else 0:.3f},'
            f'"ss":{e.silence_start if e.silence_start is not None else e.timestamp:.3f},'
            f'"m":"{e.method}"}}'
            for e in result.turn_events
        )
        timeout_count = sum(1 for e in result.turn_events if e.method == METHOD_TIMEOUT)
        total_delays = [
            e.total_delay for e in result.turn_events if e.total_delay is not None
        ]
        avg_d = sum(total_delays) / len(total_delays) if total_delays else 0
        med_d = statistics.median(total_delays) if total_delays else 0
        std_d = (
            math.sqrt(sum((d - avg_d) ** 2 for d in total_delays) / len(total_delays))
            if total_delays
            else 0
        )
        max_d = max(total_delays) if total_delays else 0
        timeout_secs_val = result.timeout_secs if result.timeout_secs is not None else 0
        audio_uri = audio_uris.get(name, "")
        analyzer_blocks.append(
            f'{{"name":"{name}","color":"{color}","init_ms":{result.init_time_ms:.1f},'
            f'"turns":{len(result.turn_events)},"avg_delay":{avg_d:.3f},'
            f'"med_delay":{med_d:.3f},"max_delay":{max_d:.3f},'
            f'"std_delay":{std_d:.3f},"timeout_count":{timeout_count},'
            f'"timeout_secs":{timeout_secs_val:.1f},'
            f'"audio":"{audio_uri}",'
            f'"events":[{events_json}]}}'
        )
    analyzers_json = ",\n      ".join(analyzer_blocks)

    basename = os.path.basename(input_path)
    filter_note = "  &mdash; Noise filter: Krisp VIVA" if viva_filter_used else ""

    html = _HTML_TEMPLATE.format(
        basename=basename,
        duration_secs=duration_secs,
        sample_rate=sample_rate,
        filter_note=filter_note,
        vad_info=vad_info,
        seg_json=seg_json,
        analyzers_json=analyzers_json,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


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
  .transport {{ display: flex; align-items: center; gap: 12px; margin-bottom: 12px;
                padding: 8px 12px; background: #0f3460; border-radius: 6px; }}
  .transport-btn {{ width: 32px; height: 32px; border-radius: 50%; border: none;
                    background: #4ecca3; color: #1a1a2e; font-size: 13px; cursor: pointer;
                    display: flex; align-items: center; justify-content: center;
                    flex-shrink: 0; }}
  .transport-btn:hover {{ filter: brightness(1.2); }}
  .transport-time {{ font-size: 0.85em; color: #ccc; min-width: 100px;
                     font-variant-numeric: tabular-nums; }}
  .transport-select {{ background: #16213e; color: #eee; border: 1px solid #333;
                       border-radius: 4px; padding: 4px 8px; font-size: 0.82em;
                       font-family: inherit; cursor: pointer; }}
  .transport-label {{ font-size: 0.78em; color: #888; }}
  .transport-spacer {{ flex: 1; }}
  .tl-wrap {{ position: relative; }}
  .timeline-row {{ display: flex; align-items: center; margin: 4px 0; height: 28px; }}
  .timeline-label {{ width: 140px; min-width: 140px; text-align: right; padding-right: 12px;
                     font-size: 0.85em; color: #aaa; white-space: nowrap; }}
  .timeline-track {{ position: relative; flex: 1; height: 100%; background: #0f3460;
                     border-radius: 4px; cursor: pointer; }}
  .speech-bar {{ position: absolute; height: 100%; background: #4ecca3; border-radius: 3px;
                 opacity: 0.8; pointer-events: none; min-width: 2px; }}
  .wait-gap {{ position: absolute; height: 100%; opacity: 0.2; border-radius: 3px;
               pointer-events: none; min-width: 1px; }}
  .turn-marker {{ position: absolute; width: 3px; height: 100%; border-radius: 2px;
                  cursor: pointer; z-index: 2; }}
  .turn-marker:hover {{ filter: brightness(1.4); }}
  .turn-marker .tooltip {{ display: none; position: absolute; bottom: 110%; left: 50%;
    transform: translateX(-50%); background: #222; color: #fff; padding: 6px 10px;
    border-radius: 4px; font-size: 0.75em; white-space: nowrap; z-index: 10;
    pointer-events: none; box-shadow: 0 2px 8px rgba(0,0,0,.5); }}
  .turn-marker:hover .tooltip {{ display: block; }}
  .playhead {{ position: absolute; top: 0; bottom: 0; width: 2px; background: #fff;
               pointer-events: none; z-index: 5; opacity: 0; transition: opacity 0.15s;
               box-shadow: 0 0 6px rgba(255,255,255,.4); }}
  .playhead.visible {{ opacity: 1; }}
  .playhead-dot {{ position: absolute; top: -4px; left: -3px; width: 8px; height: 8px;
                   border-radius: 50%; background: #fff; }}
  .ruler {{ position: relative; height: 20px; cursor: pointer; }}
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
VAD: {vad_info}{filter_note}</p>

<div class="timeline-container">
  <div class="transport" id="transport" style="display:none">
    <button class="transport-btn" id="play-btn" title="Play / Pause">&#9654;</button>
    <span class="transport-time" id="play-time">0:00.0 / 0:00.0</span>
    <div class="transport-spacer"></div>
    <span class="transport-label">Audio track</span>
    <select class="transport-select" id="track-select"></select>
  </div>
  <div class="tl-wrap" id="tl-wrap">
    <div id="tl"></div>
    <div class="playhead" id="playhead"><div class="playhead-dot"></div></div>
  </div>
</div>
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
const tlWrap = document.getElementById('tl-wrap');
const playhead = document.getElementById('playhead');
const playBtn = document.getElementById('play-btn');
const playTime = document.getElementById('play-time');
const trackSelect = document.getElementById('track-select');
const transport = document.getElementById('transport');

function pct(t) {{ return (t / DUR * 100).toFixed(4) + '%'; }}
function pctR(t) {{ return ((1 - t / DUR) * 100).toFixed(4) + '%'; }}
function mclass(m) {{ return 'method-' + m.replace(' ','-').replace('on-demand','on-demand'); }}
function msym(m) {{ return m === 'streaming' ? 'S' : m === 'on-demand' ? 'D' : 'T'; }}
function fmt(t) {{
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return m + ':' + (s < 10 ? '0' : '') + s.toFixed(1);
}}

// ---- Audio player setup ----
const audioEls = {{}};
let activeAudio = null;
const hasAudio = ANALYZERS.some(a => a.audio && a.audio.length > 0);

if (hasAudio) {{
  transport.style.display = 'flex';
  ANALYZERS.forEach((a, i) => {{
    if (!a.audio || a.audio.length === 0) return;
    const el = new Audio(a.audio);
    el.preload = 'auto';
    audioEls[a.name] = el;
    const opt = document.createElement('option');
    opt.value = a.name;
    opt.textContent = a.name;
    opt.style.color = a.color;
    trackSelect.appendChild(opt);
  }});
  const firstName = Object.keys(audioEls)[0];
  if (firstName) {{
    activeAudio = audioEls[firstName];
    trackSelect.value = firstName;
  }}
  trackSelect.addEventListener('change', () => {{
    const wasPlaying = activeAudio && !activeAudio.paused;
    const curTime = activeAudio ? activeAudio.currentTime : 0;
    if (activeAudio) activeAudio.pause();
    activeAudio = audioEls[trackSelect.value] || null;
    if (activeAudio) {{
      activeAudio.currentTime = Math.min(curTime, activeAudio.duration || DUR);
      if (wasPlaying) activeAudio.play();
    }}
  }});
}}

function togglePlay() {{
  if (!activeAudio) return;
  if (activeAudio.paused) {{
    activeAudio.play();
    playBtn.innerHTML = '&#9646;&#9646;';
    playhead.classList.add('visible');
  }} else {{
    activeAudio.pause();
    playBtn.innerHTML = '&#9654;';
  }}
}}

playBtn.addEventListener('click', togglePlay);

// Keyboard: space = play/pause
document.addEventListener('keydown', e => {{
  if (e.code === 'Space' && hasAudio && e.target === document.body) {{
    e.preventDefault();
    togglePlay();
  }}
}});

// ---- Playhead animation ----
function tickPlayhead() {{
  if (activeAudio && !activeAudio.paused) {{
    const t = activeAudio.currentTime;
    const pctVal = Math.min(t / DUR * 100, 100);
    const firstTrack = tl.querySelector('.timeline-track');
    if (firstTrack) {{
      const tlRect = tlWrap.getBoundingClientRect();
      const trackRect = firstTrack.getBoundingClientRect();
      const left = (trackRect.left - tlRect.left) + (trackRect.width * pctVal / 100);
      playhead.style.left = left + 'px';
    }}
    playTime.textContent = fmt(t) + ' / ' + fmt(DUR);
  }}
  requestAnimationFrame(tickPlayhead);
}}
requestAnimationFrame(tickPlayhead);

if (hasAudio) {{
  Object.values(audioEls).forEach(el => {{
    el.addEventListener('ended', () => {{
      playBtn.innerHTML = '&#9654;';
      playhead.classList.remove('visible');
    }});
  }});
}}

// ---- Click-to-seek on timeline tracks ----
function seekFromClick(e) {{
  if (!activeAudio) return;
  const track = e.target.closest('.timeline-track');
  if (!track) return;
  const rect = track.getBoundingClientRect();
  const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  activeAudio.currentTime = frac * DUR;
  playhead.classList.add('visible');
  const pctVal = frac * 100;
  const tlRect = tlWrap.getBoundingClientRect();
  const left = (rect.left - tlRect.left) + (rect.width * pctVal / 100);
  playhead.style.left = left + 'px';
  playTime.textContent = fmt(activeAudio.currentTime) + ' / ' + fmt(DUR);
}}

// ---- Build timeline rows ----

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
  sr += '<div class="speech-bar" style="left:' + pct(s[0]) + ';right:' +
        pctR(s[1]) + '"></div>';
}});
sr += '</div></div>';
tl.innerHTML += sr;

// Analyzer rows with wait-gap visualization
ANALYZERS.forEach(a => {{
  let row = '<div class="timeline-row"><div class="timeline-label" style="color:' +
    a.color + '">' + a.name + '</div><div class="timeline-track">';
  a.events.forEach(e => {{
    // Effective position: offset by VAD wait so markers reflect total latency
    var ePos = e.t + e.vad;
    if (e.d > 0.05) {{
      row += '<div class="wait-gap" style="left:' + pct(e.ss) + ';right:' +
        pctR(ePos) + ';background:' + a.color + '" title="total ' +
        e.d.toFixed(3) + 's"></div>';
    }}
    var tip = ePos.toFixed(2) + 's &mdash; total: ' + e.d.toFixed(3) + 's [' + msym(e.m) + ']';
    if (e.vad > 0) tip += ' (' + e.vad.toFixed(1) + 's VAD + ' + (e.d - e.vad).toFixed(3) + 's model)';
    row += '<div class="turn-marker" style="left:' + pct(ePos) + ';background:' +
      a.color + '"><div class="tooltip">' + tip + '</div></div>';
  }});
  row += '</div></div>';
  tl.innerHTML += row;
}});

// Attach click-to-seek on all tracks
tl.addEventListener('click', seekFromClick);

// ---- Summary cards ----
ANALYZERS.forEach(a => {{
  let warn = '';
  const instant = a.events.filter(e => e.d < 0.001).length;
  if (instant > a.turns * 0.5 && a.turns > 1)
    warn = '<div class="warn">&#9888; ' + instant + '/' + a.turns +
      ' turns at 0ms \\u2014 analyzer confirms VAD, no independent analysis</div>';
  if (a.timeout_count > 0)
    warn += '<div class="warn">&#9888; ' + a.timeout_count + '/' + a.turns +
      ' turns used timeout fallback \\u2014 user waited full silence timer</div>';
  if (a.timeout_secs > 0 && instant > a.turns * 0.5 && a.turns > 1)
    warn += '<div class="warn">&#9888; When model is uncertain, worst-case response = ' +
      a.timeout_secs.toFixed(1) + 's (silence timeout fallback).</div>';
  if (a.turns >= 3 && a.max_delay > a.med_delay * 2.5 && a.med_delay > 0.01)
    warn += '<div class="warn">&#8505; max (' + a.max_delay.toFixed(3) +
      's) >> median (' + a.med_delay.toFixed(3) +
      's) \\u2014 outlier likely due to VAD truncating speech, not analyzer latency</div>';
  const methods = {{}};
  a.events.forEach(e => {{ methods[e.m] = (methods[e.m]||0) + 1; }});
  let mstr = Object.entries(methods).map(([k,v]) => v + ' ' + k).join(', ');
  let worstCase = '';
  if (a.timeout_secs > 0)
    worstCase = '<div class="stat"><span class="label">Worst-case response</span>' +
      '<span style="color:#ff6b6b">' + a.timeout_secs.toFixed(1) + 's</span></div>';
  cards.innerHTML += '<div class="card"><h3 style="color:' + a.color + '">' + a.name + '</h3>'
    + '<div class="stat"><span class="label">Turns</span><span>' + a.turns + '</span></div>'
    + '<div class="stat"><span class="label">Median response time</span><span>' +
        a.med_delay.toFixed(3) + 's</span></div>'
    + '<div class="stat"><span class="label">Avg response time</span><span>' +
        a.avg_delay.toFixed(3) + 's</span></div>'
    + worstCase
    + '<div class="stat"><span class="label">Stddev</span><span>' +
        a.std_delay.toFixed(3) + 's</span></div>'
    + '<div class="stat"><span class="label">Init time</span><span>' +
        a.init_ms.toFixed(0) + 'ms</span></div>'
    + '<div class="stat"><span class="label">Methods</span><span>' + mstr + '</span></div>'
    + warn + '</div>';
}});

// ---- Comparison table ----
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
      if (!ev) {{ row += '<td style="color:#3a4a5a">(missed)</td>'; return; }}
      const mc = mclass(ev.m);
      row += '<td>' + ev.t.toFixed(2) + 's <span style="color:#6b7a8d">+' +
        ev.d.toFixed(2) + 's</span> <span class="method ' + mc + '">' +
        msym(ev.m) + '</span></td>';
      delays[a.name] = ev.d;
    }});

    const entries = Object.entries(delays).sort((a,b) => a[1]-b[1]);
    let verdict = '';
    if (entries.length >= 2) {{
      const delta = entries[1][1] - entries[0][1];
      if (delta < 0.01) verdict = '<span style="color:#6b7a8d">tie</span>';
      else verdict = '<span class="faster">' + entries[0][0] + '</span>' +
        ' <span style="color:#6b7a8d">+' + delta.toFixed(2) + 's faster</span>';
    }}
    row += '<td>' + verdict + '</td></tr>';
    cmp.innerHTML += row;
  }});
}}
</script>
</body>
</html>"""
