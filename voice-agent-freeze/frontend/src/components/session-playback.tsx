"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { SESSION_RECORDING_ID } from "@/lib/session";
import { cn } from "@/lib/utils";

type Role = "user" | "assistant";

type Turn = {
  id: string;
  role: Role;
  content: string;
  start_ts: number | null;
  end_ts: number;
  latency?: number;
  interrupted?: boolean;
};

type PlaybackTurn = Turn & { start: number; end: number };

type TranscriptPayload =
  | Turn[]
  | {
      recording_started_at?: number | null;
      recording_ended_at?: number | null;
      turns?: Turn[];
    };

type ParsedSession = {
  turns: PlaybackTurn[];
  timelineBaseUnix: number;
  recordingEndedAtUnix: number | null;
};

const FREEZE_MIN_DURATION_SECONDS = 5;

const ms = (s: number) => `${Math.round(s * 1000)}ms`;

const formatTime = (seconds: number) => {
  const m = Math.floor(seconds / 60);
  const sec = Math.floor(seconds % 60);
  const fraction = Math.floor((seconds % 1) * 1000);
  return `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}.${String(fraction).padStart(3, "0")}`;
};

function parseTranscriptPayload(payload: TranscriptPayload): ParsedSession | null {
  const raw = Array.isArray(payload) ? payload : (payload.turns ?? []);
  if (raw.length === 0) return null;

  const withStart = raw.filter((t) => t.start_ts !== null);
  if (withStart.length === 0) return null;

  const recordingStartedAtUnix = !Array.isArray(payload) ? payload.recording_started_at ?? null : null;
  const recordingEndedAtUnix = !Array.isArray(payload) ? payload.recording_ended_at ?? null : null;
  const base = recordingStartedAtUnix ?? (withStart[0].start_ts as number);

  const turns = withStart
    .map((turn, i) => {
      const rawStart = turn.start_ts as number;
      const start = Math.max(0, rawStart - base);
      const end = Math.max(start + 0.001, turn.end_ts - base);
      return {
        ...turn,
        start,
        end,
        latency: turn.latency ?? undefined,
        interrupted: Boolean(turn.interrupted),
        content: turn.content ?? "",
        id: turn.id || `${turn.role}-${i}`,
      };
    })
    .filter((t) => Number.isFinite(t.start) && Number.isFinite(t.end))
    .sort((a, b) => a.start - b.start);

  return turns.length ? { turns, timelineBaseUnix: base, recordingEndedAtUnix } : null;
}

function freezeOverlay(
  turns: PlaybackTurn[],
  recordingEndedAtUnix: number | null,
  timelineBaseUnix: number,
  audioDuration: number
): { freezeSeconds: number; overlayLeftPct: number; overlayWidthPct: number } | null {
  if (recordingEndedAtUnix == null || !turns.length || audioDuration <= 0) return null;
  const last = turns[turns.length - 1];
  if (last.role !== "user") return null;

  const freezeSeconds = recordingEndedAtUnix - last.end_ts;
  if (freezeSeconds <= FREEZE_MIN_DURATION_SECONDS) return null;

  const freezeEndRel = recordingEndedAtUnix - timelineBaseUnix;
  const span = Math.max(0, Math.min(freezeEndRel, audioDuration) - last.end);
  if (span <= 0) return null;

  const left = Math.min(100, Math.max(0, (last.end / audioDuration) * 100));
  const width = Math.min(100 - left, (span / audioDuration) * 100);
  return { freezeSeconds, overlayLeftPct: left, overlayWidthPct: width };
}

export default function SessionPlayback() {
  const id = SESSION_RECORDING_ID;
  const waveformRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WaveSurfer | null>(null);
  const turnsRef = useRef<PlaybackTurn[]>([]);
  const activeIdxRef = useRef(-1);
  const audioReadyRef = useRef(false);

  const [session, setSession] = useState<ParsedSession | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [duration, setDuration] = useState(0);
  const [activeIdx, setActiveIdx] = useState(-1);
  const [playing, setPlaying] = useState(false);

  turnsRef.current = session?.turns ?? [];

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`/sessions/${id}.json`);
        if (!res.ok) throw new Error(`Transcript not found (${res.status})`);
        const parsed = parseTranscriptPayload((await res.json()) as TranscriptPayload);
        if (!parsed) throw new Error("Transcript is empty or invalid");
        if (!cancelled) {
          setSession(parsed);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Failed to load transcript");
          setSession(null);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const seek = useCallback((start: number) => {
    const ws = wsRef.current;
    if (!ws || duration <= 0) return;
    ws.seekTo(Math.min(1, Math.max(0, start / duration)));
  }, [duration]);

  useEffect(() => {
    const el = waveformRef.current;
    if (!el) return;

    const ws = WaveSurfer.create({
      container: el,
      url: `/sessions/${id}.wav`,
      height: 140,
      waveColor: "#334155",
      progressColor: "#38bdf8",
      cursorColor: "#e2e8f0",
      barWidth: 2,
      barGap: 1,
      normalize: true,
    });
    wsRef.current = ws;
    audioReadyRef.current = false;
    setReady(false);
    setDuration(0);
    setActiveIdx(-1);
    activeIdxRef.current = -1;
    setPlaying(false);

    ws.on("ready", () => {
      audioReadyRef.current = true;
      setReady(true);
      setError(null);
      setDuration(ws.getDuration());
    });
    ws.on("timeupdate", (time) => {
      const turns = turnsRef.current;
      if (!turns.length) return;
      const idx = turns.findIndex((t) => time >= t.start && time < t.end);
      if (idx !== activeIdxRef.current) {
        activeIdxRef.current = idx;
        setActiveIdx(idx);
      }
    });
    ws.on("play", () => setPlaying(true));
    ws.on("pause", () => setPlaying(false));
    ws.on("finish", () => setPlaying(false));
    ws.on("error", () => {
      if (!audioReadyRef.current) setError("Unable to load audio file");
    });

    return () => {
      ws.destroy();
      wsRef.current = null;
    };
  }, []);

  const freeze = useMemo(
    () =>
      session
        ? freezeOverlay(session.turns, session.recordingEndedAtUnix, session.timelineBaseUnix, duration)
        : null,
    [session, duration]
  );

  const turns = session?.turns ?? [];
  const assistantMarkers = useMemo(() => turns.filter((t) => t.role === "assistant"), [turns]);

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-5xl flex-col gap-6 px-4 py-8 sm:px-6">
      <header className="space-y-1">
        <h1 className="text-2xl font-semibold text-slate-100">Session playback</h1>
        <p className="text-sm text-slate-400">{id}</p>
      </header>

      {error && (
        <div className="rounded-md border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">{error}</div>
      )}

      {freeze && (
        <div className="rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm text-amber-100">
          <strong className="font-medium">Freeze detected</strong>
          <span className="text-amber-200/90">
            {" "}
            — No assistant reply after last user turn; gap to call end (&gt;{FREEZE_MIN_DURATION_SECONDS}s):{" "}
            {formatTime(freeze.freezeSeconds)} ({ms(freeze.freezeSeconds)})
          </span>
        </div>
      )}

      <Card className="p-4 shadow-lg shadow-slate-950/30">
        <div className="mb-4 flex items-center justify-between gap-3">
          <Button type="button" onClick={() => wsRef.current?.playPause()} disabled={!ready}>
            {playing ? "Pause" : "Play"}
          </Button>
          <span className="text-xs text-slate-400">{formatTime(duration)}</span>
        </div>

        <div className="relative">
          <div ref={waveformRef} className="rounded-md border border-slate-800 bg-slate-950" />

          {ready && duration > 0 && freeze && freeze.overlayWidthPct > 0 && (
            <div
              className="pointer-events-none absolute inset-y-0 bg-amber-500/15"
              style={{ left: `${freeze.overlayLeftPct}%`, width: `${freeze.overlayWidthPct}%` }}
              title="No assistant audio (freeze)"
            />
          )}

          {ready && duration > 0 && assistantMarkers.length > 0 && (
            <div className="pointer-events-none absolute inset-0">
              {assistantMarkers.map((turn) => {
                const leftPct = Math.min(100, Math.max(0, (turn.start / duration) * 100));
                return (
                  <div
                    key={turn.id}
                    className="absolute inset-y-0"
                    style={{ left: `${leftPct}%`, transform: "translateX(-0.5px)" }}
                  >
                    {turn.latency !== undefined && (
                      <span className="absolute -top-7 -translate-x-1/2 text-xs font-semibold text-yellow-400 sm:text-sm">
                        {ms(turn.latency)}
                      </span>
                    )}
                    <span className="block h-full w-px bg-rose-500" />
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </Card>

      <Card className="flex min-h-0 flex-1 p-2 pb-0">
        <ScrollArea className="h-[55vh] w-full">
          <div className="space-y-3 p-2 pb-8">
            {turns.map((turn, index) => (
              <Card
                key={turn.id}
                role="button"
                tabIndex={0}
                onClick={() => seek(turn.start)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    seek(turn.start);
                  }
                }}
                className={cn(
                  "cursor-pointer transition-colors",
                  index === activeIdx ? "border-sky-400/70 bg-sky-500/10" : "hover:border-slate-500/80"
                )}
              >
                <CardHeader className="pb-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge variant={turn.role === "user" ? "user" : "assistant"}>
                      {turn.role === "user" ? "User" : "Assistant"}
                    </Badge>
                    {turn.role === "assistant" && turn.latency !== undefined && (
                      <Badge variant="latency">{ms(turn.latency)}</Badge>
                    )}
                    {turn.interrupted && <Badge variant="interrupted">Interrupted</Badge>}
                    <span className="text-xs text-slate-400">{formatTime(turn.start)}</span>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-200">{turn.content}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </ScrollArea>
      </Card>
    </main>
  );
}
