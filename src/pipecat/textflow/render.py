#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Render a :class:`~pipecat.textflow.trace.TextFlowTrace` in the terminal."""

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .trace import TextFlowSegment, TextFlowTrace

_OUTCOME_STYLES = {
    "spoken": "green",
    "skipped": "yellow",
    "dropped": "red",
    "unspoken": "magenta",
}


def _quoted(text: str | None, *, missing: str = "—") -> Text:
    if text is None:
        return Text(missing, style="dim")
    return Text(repr(text))


def _segments_table(segments: list[TextFlowSegment]) -> Table:
    table = Table(title="Segments", expand=True, show_lines=True)
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("type", no_wrap=True)
    table.add_column("outcome", no_wrap=True)
    table.add_column("LLM text", overflow="fold")
    table.add_column("context", overflow="fold")
    table.add_column("TTS text", overflow="fold")
    table.add_column("client text", overflow="fold")
    for i, segment in enumerate(segments, start=1):
        outcome = Text(segment.outcome, style=_OUTCOME_STYLES.get(segment.outcome, ""))
        tts = _quoted(segment.tts_text, missing=f"not sent ({segment.outcome})")
        sent = segment.rtvi_messages[-1] if segment.rtvi_messages else None
        client = _quoted((sent.get("data") or {}).get("text") if sent else None, missing="not sent")
        table.add_row(
            str(i),
            segment.text_type,
            outcome,
            _quoted(segment.llm_text),
            _quoted(segment.context_text, missing="nothing recorded"),
            tts,
            client,
        )
    return table


def _events(segments: list[TextFlowSegment]) -> Table | None:
    rows = [(i, event) for i, segment in enumerate(segments, start=1) for event in segment.events]
    if not rows:
        return None
    table = Table(title="What each stage did", expand=True)
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("stage", no_wrap=True)
    table.add_column("action", no_wrap=True)
    table.add_column("before", overflow="fold")
    table.add_column("after", overflow="fold")
    for i, event in rows:
        action = f"{event.kind} {event.name}".strip()
        after = _quoted(event.after, missing="(removed)")
        if event.kind == "error":
            after = Text(event.detail, style="red")
        table.add_row(str(i), event.stage, action, _quoted(event.before), after)
    return table


def _words(segments: list[TextFlowSegment]) -> Table | None:
    rows = [(i, word) for i, segment in enumerate(segments, start=1) for word in segment.words]
    if not rows:
        return None
    table = Table(title="Words as spoken", expand=True)
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("provider word", no_wrap=True)
    table.add_column("LLM span", no_wrap=True)
    table.add_column("spoken so far", overflow="fold")
    table.add_column("remaining", overflow="fold")
    for i, word in rows:
        table.add_row(
            str(i),
            repr(word.text),
            _quoted(word.llm_text),
            _quoted(word.accumulated),
            _quoted(word.remaining),
        )
    return table


def _consumers(trace: TextFlowTrace) -> Group:
    context = (
        "\n".join(f"[{m.get('role')}] {m.get('content')!r}" for m in trace.context_messages)
        or "(nothing reached the context)"
    )

    numbers = {s.segment_id: i for i, s in enumerate(trace.segments, start=1)}
    client_lines = []
    for message in trace.rtvi_messages:
        data = message.get("data") or {}
        kind = message.get("type", "")
        if kind == "bot-output":
            progress = data.get("spoken_progress") or {}
            status = data.get("spoken_status", "")
            number = numbers.get(data.get("segment_id"), "?")
            line = f"{kind} segment {number} {status}: {data.get('text')!r}"
            if progress:
                line += f"  spoken={progress.get('accumulated_text')!r}"
            client_lines.append(line)
        elif "text" in data:
            client_lines.append(f"{kind}: {data['text']!r}")
        else:
            client_lines.append(kind)
    client = "\n".join(client_lines) or "(RTVI disabled)"

    provider = (
        "\n".join(
            f"{i}. {request['text']!r}" for i, request in enumerate(trace.provider_requests, 1)
        )
        or "(nothing was synthesized)"
    )

    return Group(
        Panel(context, title="Conversation context"),
        Panel(client, title="RTVI client"),
        Panel(provider, title="TTS provider"),
    )


def _steps(trace: TextFlowTrace) -> Table:
    table = Table(title="Steps", expand=True)
    table.add_column("t", justify="right", no_wrap=True)
    table.add_column("where", no_wrap=True)
    table.add_column("what", no_wrap=True)
    table.add_column("detail", overflow="fold")
    styles = {"rtvi": "magenta", "provider": "yellow", "aggregate": "green", "stage": "cyan"}
    for entry in trace.timeline:
        where = (
            entry.source
            if entry.kind in ("aggregate", "stage")
            else f"{entry.source} → {entry.destination}"
        )
        table.add_row(
            f"{entry.t:.2f}s",
            Text(where, style=styles.get(entry.kind, "")),
            entry.label,
            Text(entry.summary, style="dim"),
        )
    return table


def render_trace(
    trace: TextFlowTrace,
    console: Console | None = None,
    *,
    words: bool = True,
    steps: bool = False,
):
    """Print a trace to the console.

    Args:
        trace: The trace to print.
        console: The console to print to; a new one when None.
        words: Whether to include the per-word table.
        steps: Whether to list every step of the run, in order.
    """
    console = console or Console()
    if trace.bot is not None:
        pipeline = Text()
        for i, name in enumerate(trace.bot.pipeline):
            if i:
                pipeline.append(" → ", style="dim")
            pipeline.append(name, style="bold green" if i in trace.bot.traced else "")
        lines: list[Text] = []
        if trace.bot.tts_service is not None:
            how = (
                "live, calling its provider"
                if trace.setup.tts.live
                else "its parts on the stand-in"
            )
            lines.append(Text(f"TTS: {trace.bot.tts_service}, {how}", style="dim"))
        lines.extend(Text(f"note: {n}", style="yellow") for n in trace.bot.notes)
        body = Group(pipeline, *lines)
        console.print(Panel(body, title=f"Bot: {trace.bot.path}"))
    console.print(Panel(Text(trace.text), title=f"LLM response · setup: {trace.setup.name}"))
    chunks = Text(" | ".join(repr(c) for c in trace.chunks), style="dim")
    console.print(Panel(chunks, title=f"Streamed as {len(trace.chunks)} chunks"))
    console.print(_segments_table(trace.segments))
    if (events := _events(trace.segments)) is not None:
        console.print(events)
    if words and (table := _words(trace.segments)) is not None:
        console.print(table)
    console.print(_consumers(trace))
    if steps:
        console.print(_steps(trace))
    for error in trace.errors:
        console.print(Text(f"error: {error}", style="red"))
