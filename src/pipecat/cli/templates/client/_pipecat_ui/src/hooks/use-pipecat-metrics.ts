"use client";

import type { PipecatClient, PipecatMetricsData } from "@pipecat-ai/client-js";
import { RTVIEvent } from "@pipecat-ai/client-js";
import { usePipecatClient } from "@pipecat-ai/client-react";
import { useEffect, useMemo } from "react";
import { create } from "zustand";

export type MetricCategory =
  "ttfb" | "ttfa" | "processing" | "characters" | "stt_usage";

const METRIC_CATEGORIES: MetricCategory[] = [
  "ttfb",
  "ttfa",
  "processing",
  "characters",
  "stt_usage",
];

function numberOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/** The SDK omits some server metrics. ttfa reads its headline value; stt_usage carries incremental audio_seconds. */
const CATEGORY_SAMPLERS: Record<
  MetricCategory,
  (entry: Record<string, unknown>) => number | null
> = {
  ttfb: (entry) => numberOrNull(entry.value),
  ttfa: (entry) => numberOrNull(entry.ttfa),
  processing: (entry) => numberOrNull(entry.value),
  characters: (entry) => numberOrNull(entry.value),
  stt_usage: (entry) =>
    numberOrNull(
      (entry.value as Record<string, unknown> | null)?.audio_seconds,
    ),
};

/** Categories whose samples are deltas accumulated into a running total. */
const CUMULATIVE_CATEGORIES = new Set<MetricCategory>(["stt_usage"]);

export interface MetricPoint {
  /** Epoch milliseconds when the metrics batch arrived. */
  time: number;
  value: number;
}

export interface MetricSeries {
  category: MetricCategory;
  processor: string;
  /** Most recent value in the series. */
  latest: number;
  /** Rolling window of samples, capped at the store's maxPoints. */
  points: MetricPoint[];
}

export interface TokenTotals {
  /** Prompt tokens — net of the cache on services that report cache reads. */
  prompt: number;
  completion: number;
  /** Gross total; not always prompt + completion (see cacheRead). */
  total: number;
  /** Cache-read input tokens, when the provider reports them. */
  cacheRead: number;
  /** Reasoning tokens, when the provider reports them. */
  reasoning: number;
}

const ZERO_TOKENS: TokenTotals = {
  prompt: 0,
  completion: 0,
  total: 0,
  cacheRead: 0,
  reasoning: 0,
};

interface PipecatMetricsState {
  /** Series keyed by `${category}:${processor}`. */
  series: Record<string, MetricSeries>;
  /** Running token totals for the session. */
  tokens: TokenTotals;
  /** Whether any token metrics arrived this session (zero totals can be real). */
  hasTokens: boolean;
  /** Per-series point cap. Grows only (max requested wins). */
  maxPoints: number;
  /** Folds one RTVIEvent.Metrics payload into the store. */
  ingest: (data: PipecatMetricsData, time: number) => void;
  /** Clears session data (series + tokens); keeps maxPoints. */
  reset: () => void;
  setMaxPoints: (maxPoints: number) => void;
}

function seriesKey(category: MetricCategory, processor: string): string {
  return `${category}:${processor}`;
}

/** Shared session metrics. Keep a hook mounted to capture events while metric views are closed. */
export const usePipecatMetricsStore = create<PipecatMetricsState>()((set) => ({
  series: {},
  tokens: ZERO_TOKENS,
  hasTokens: false,
  maxPoints: 100,
  ingest: (data, time) =>
    set((state) => {
      const series = { ...state.series };
      let seriesChanged = false;

      for (const category of METRIC_CATEGORIES) {
        const entries = (data as Record<string, unknown>)[category];
        if (!Array.isArray(entries)) continue;
        for (const raw of entries) {
          if (typeof raw !== "object" || raw === null) continue;
          const entry = raw as Record<string, unknown>;
          if (typeof entry.processor !== "string") continue;
          const sample = CATEGORY_SAMPLERS[category](entry);
          if (sample === null) continue;
          const key = seriesKey(category, entry.processor);
          const existing = series[key];
          const value = CUMULATIVE_CATEGORIES.has(category)
            ? (existing?.latest ?? 0) + sample
            : sample;
          const points = [...(existing?.points ?? []), { time, value }].slice(
            -state.maxPoints,
          );
          series[key] = {
            category,
            processor: entry.processor,
            latest: value,
            points,
          };
          seriesChanged = true;
        }
      }

      // `tokens` is not part of PipecatMetricsData's type, so parse it
      // defensively off the wire shape: [{ prompt_tokens, … }]. Entries are
      // per-completion deltas; cache/reasoning fields are optional.
      const rawTokens = (data as { tokens?: unknown }).tokens;
      let tokens = state.tokens;
      let hasTokens = state.hasTokens;
      if (Array.isArray(rawTokens) && rawTokens.length > 0) {
        const num = (value: unknown) =>
          typeof value === "number" && Number.isFinite(value) ? value : 0;
        let next = state.tokens;
        for (const raw of rawTokens) {
          if (typeof raw !== "object" || raw === null) continue;
          const batch = raw as Record<string, unknown>;
          next = {
            prompt: next.prompt + num(batch.prompt_tokens),
            completion: next.completion + num(batch.completion_tokens),
            total: next.total + num(batch.total_tokens),
            cacheRead: next.cacheRead + num(batch.cache_read_input_tokens),
            reasoning: next.reasoning + num(batch.reasoning_tokens),
          };
          hasTokens = true;
        }
        tokens = next;
      }

      if (!seriesChanged && tokens === state.tokens) return state;
      return { series, tokens, hasTokens };
    }),
  reset: () => set({ series: {}, tokens: ZERO_TOKENS, hasTokens: false }),
  setMaxPoints: (maxPoints) =>
    set((state) => ({
      maxPoints: Math.max(state.maxPoints, Math.max(1, Math.floor(maxPoints))),
    })),
}));

// Listener attachment: ref-counted per client so any number of subscribers
// share one RTVI listener, attached by the first and detached by the last.

const listenerRefCounts = new Map<PipecatClient, number>();

function handleMetrics(data: PipecatMetricsData) {
  usePipecatMetricsStore.getState().ingest(data, Date.now());
}

function handleConnected() {
  usePipecatMetricsStore.getState().reset();
}

function attachListeners(client: PipecatClient) {
  const count = listenerRefCounts.get(client) ?? 0;
  listenerRefCounts.set(client, count + 1);
  if (count > 0) return;
  client.on(RTVIEvent.Metrics, handleMetrics);
  client.on(RTVIEvent.Connected, handleConnected);
}

function detachListeners(client: PipecatClient) {
  const count = listenerRefCounts.get(client) ?? 0;
  if (count <= 1) {
    listenerRefCounts.delete(client);
    client.off(RTVIEvent.Metrics, handleMetrics);
    client.off(RTVIEvent.Connected, handleConnected);
    return;
  }
  listenerRefCounts.set(client, count - 1);
}

/** Shared by every public hook: feed the store from the context client. */
function useAttachMetricsListeners() {
  const client = usePipecatClient();
  useEffect(() => {
    if (!client) return;
    attachListeners(client);
    return () => detachListeners(client);
  }, [client]);
}

export interface UsePipecatMetricsOptions {
  /**
   * Raises the per-series point cap (default 100). The cap is shared by all
   * subscribers — the largest requested value wins.
   */
  maxPoints?: number;
}

export interface UsePipecatMetricsReturn {
  /** All series, stable-sorted by category then processor. */
  series: MetricSeries[];
  /** Running token totals for the session. */
  tokens: TokenTotals;
  /** Whether any token metrics arrived this session. */
  hasTokens: boolean;
  /** Clears the session's collected data. */
  reset: () => void;
}

/**
 * Subscribes to the shared RTVI metrics store, attaching the client listener
 * on first use. Must be rendered inside a PipecatClientProvider.
 */
export function usePipecatMetrics(
  options?: UsePipecatMetricsOptions,
): UsePipecatMetricsReturn {
  useAttachMetricsListeners();
  const maxPoints = options?.maxPoints;
  useEffect(() => {
    if (typeof maxPoints === "number") {
      usePipecatMetricsStore.getState().setMaxPoints(maxPoints);
    }
  }, [maxPoints]);

  const seriesMap = usePipecatMetricsStore((state) => state.series);
  const tokens = usePipecatMetricsStore((state) => state.tokens);
  const hasTokens = usePipecatMetricsStore((state) => state.hasTokens);
  const reset = usePipecatMetricsStore((state) => state.reset);

  const series = useMemo(
    () =>
      Object.values(seriesMap).sort(
        (a, b) =>
          a.category.localeCompare(b.category) ||
          a.processor.localeCompare(b.processor),
      ),
    [seriesMap],
  );

  return { series, tokens, hasTokens, reset };
}

/** Latest value, or null before a sample. Omit processor to follow the most recently updated one. */
export function usePipecatMetricValue(
  category: MetricCategory,
  processor?: string,
): number | null {
  useAttachMetricsListeners();
  return usePipecatMetricsStore((state) => {
    if (processor) {
      return state.series[seriesKey(category, processor)]?.latest ?? null;
    }
    let best: MetricSeries | null = null;
    let bestTime = -1;
    for (const key in state.series) {
      const entry = state.series[key]!;
      if (entry.category !== category) continue;
      const lastTime = entry.points[entry.points.length - 1]?.time ?? 0;
      if (lastTime > bestTime) {
        best = entry;
        bestTime = lastTime;
      }
    }
    return best?.latest ?? null;
  });
}

export interface UsePipecatTokenTotalsReturn {
  tokens: TokenTotals;
  /** False until any token metrics arrive (zero totals can be real). */
  hasTokens: boolean;
}

/**
 * Reads the session's running token totals. Must be rendered inside a
 * PipecatClientProvider.
 */
export function usePipecatTokenTotals(): UsePipecatTokenTotalsReturn {
  useAttachMetricsListeners();
  const tokens = usePipecatMetricsStore((state) => state.tokens);
  const hasTokens = usePipecatMetricsStore((state) => state.hasTokens);
  return { tokens, hasTokens };
}
