"use client";

import type { PipecatClient, TransportState } from "@pipecat-ai/client-js";
import { RTVIEvent } from "@pipecat-ai/client-js";
import { usePipecatClient } from "@pipecat-ai/client-react";
import { useEffect, useMemo, useRef } from "react";
import { create } from "zustand";

export interface PipecatEventLog {
  /** Monotonic id (capture order). */
  id: string;
  /** RTVIEvent name, e.g. "botStartedSpeaking". */
  type: string;
  /** The event's payload: the single callback argument, or an array of them. */
  data: unknown;
  timestamp: Date;
}

export interface PipecatEventGroup {
  /** Id of the group's first event. */
  id: string;
  /** Type of the group's first event. */
  type: string;
  events: readonly PipecatEventLog[];
}

/** High-frequency events never captured into the shared log. */
const CAPTURE_IGNORED: string[] = [RTVIEvent.LocalAudioLevel];

interface PipecatEventStreamState {
  /** The shared capture log, oldest first, capped at maxEvents. */
  events: readonly PipecatEventLog[];
  /** While true, new events are dropped; listeners stay attached. */
  paused: boolean;
  /** Log cap. Grows only (max requested wins). */
  maxEvents: number;
  append: (batch: PipecatEventLog[]) => void;
  clear: () => void;
  setPaused: (paused: boolean) => void;
  setMaxEvents: (maxEvents: number) => void;
}

/** Shared event backlog. Keep a hook mounted to capture events while log views are closed. */
export const usePipecatEventStreamStore = create<PipecatEventStreamState>()(
  (set) => ({
    events: [],
    paused: false,
    maxEvents: 500,
    append: (batch) =>
      set((state) => ({
        events: [...state.events, ...batch].slice(-state.maxEvents),
      })),
    clear: () => {
      // Drop the not-yet-flushed batch too — events captured before a clear
      // must not resurface after it.
      pendingBatch = [];
      if (flushHandle !== null) {
        cancelAnimationFrame(flushHandle);
        flushHandle = null;
      }
      set({ events: [] });
    },
    setPaused: (paused) => set({ paused }),
    setMaxEvents: (maxEvents) =>
      set((state) => ({
        maxEvents: Math.max(
          state.maxEvents,
          Math.max(1, Math.floor(maxEvents)),
        ),
      })),
  }),
);

// Capture: every RTVIEvent minus CAPTURE_IGNORED, batched into the store on
// requestAnimationFrame so event bursts cost one render per frame.

let eventSeq = 0;
let pendingBatch: PipecatEventLog[] = [];
let flushHandle: number | null = null;

function flushPending() {
  flushHandle = null;
  if (pendingBatch.length === 0) return;
  const batch = pendingBatch;
  pendingBatch = [];
  usePipecatEventStreamStore.getState().append(batch);
}

function captureEvent(type: string, args: unknown[]) {
  if (usePipecatEventStreamStore.getState().paused) return;
  pendingBatch.push({
    id: `e${++eventSeq}`,
    type,
    data: args.length > 1 ? args : args[0],
    timestamp: new Date(),
  });
  flushHandle ??= requestAnimationFrame(flushPending);
}

/** Shared per-event capture handlers, so on/off stays symmetric. */
const captureHandlers = new Map<string, (...args: unknown[]) => void>();

function captureHandler(type: string): (...args: unknown[]) => void {
  let handler = captureHandlers.get(type);
  if (!handler) {
    handler = (...args: unknown[]) => captureEvent(type, args);
    captureHandlers.set(type, handler);
  }
  return handler;
}

const CAPTURED_EVENTS = Object.values(RTVIEvent).filter(
  (event): event is RTVIEvent =>
    typeof event === "string" && !CAPTURE_IGNORED.includes(event),
);

const listenerRefCounts = new Map<PipecatClient, number>();
/** Per-client auto-clear closures (they track that client's last state). */
const autoClearHandlers = new Map<
  PipecatClient,
  (state: TransportState) => void
>();

function attachListeners(client: PipecatClient) {
  const count = listenerRefCounts.get(client) ?? 0;
  listenerRefCounts.set(client, count + 1);
  if (count > 0) return;

  for (const event of CAPTURED_EVENTS) {
    client.on(event, captureHandler(event));
  }

  let lastState: TransportState | null = null;
  const autoClear = (state: TransportState) => {
    if (lastState === "disconnected" && state === "initializing") {
      usePipecatEventStreamStore.getState().clear();
    }
    lastState = state;
  };
  autoClearHandlers.set(client, autoClear);
  client.on(RTVIEvent.TransportStateChanged, autoClear);
}

function detachListeners(client: PipecatClient) {
  const count = listenerRefCounts.get(client) ?? 0;
  if (count > 1) {
    listenerRefCounts.set(client, count - 1);
    return;
  }
  listenerRefCounts.delete(client);
  for (const event of CAPTURED_EVENTS) {
    client.off(event, captureHandler(event));
  }
  const autoClear = autoClearHandlers.get(client);
  if (autoClear) {
    client.off(RTVIEvent.TransportStateChanged, autoClear);
    autoClearHandlers.delete(client);
  }
}

export interface UsePipecatEventStreamOptions {
  /**
   * Raises the shared log cap (default 500). The cap is shared by all
   * subscribers — the largest requested value wins.
   */
  maxEvents?: number;
  /** Only these event types pass this subscriber's filter (wins over ignoreEvents). */
  includeEvents?: string[];
  /** Event types hidden from this subscriber (LocalAudioLevel is never captured). */
  ignoreEvents?: string[];
  /** Also derive `groups`, merging consecutive events with the same key. */
  groupConsecutive?: boolean;
  /** Grouping key (default: the event type). */
  groupKey?: (event: PipecatEventLog) => string;
  /**
   * Fired once per new filter-passing event. Not fired for backlog present
   * when the subscriber mounted.
   */
  onEvent?: (event: PipecatEventLog) => void;
}

export interface UsePipecatEventStreamReturn {
  /** This subscriber's filtered view of the shared log, oldest first. */
  events: readonly PipecatEventLog[];
  /** Consecutive-run groups (empty unless groupConsecutive). */
  groups: readonly PipecatEventGroup[];
  /** Shared pause state: while true, no subscriber receives new events. */
  paused: boolean;
  setPaused: (paused: boolean) => void;
  /** Clears the shared log for every subscriber. */
  clear: () => void;
}

function eventSeqOf(event: PipecatEventLog): number {
  return Number(event.id.slice(1));
}

/**
 * Subscribes to the shared RTVI event log, attaching the client listeners on
 * first use. Capture is shared and rAF-batched; filtering and grouping are
 * computed per subscriber. Must be rendered inside a PipecatClientProvider.
 */
export function usePipecatEventStream(
  options?: UsePipecatEventStreamOptions,
): UsePipecatEventStreamReturn {
  const client = usePipecatClient();
  useEffect(() => {
    if (!client) return;
    attachListeners(client);
    return () => detachListeners(client);
  }, [client]);

  const maxEvents = options?.maxEvents;
  useEffect(() => {
    if (typeof maxEvents === "number") {
      usePipecatEventStreamStore.getState().setMaxEvents(maxEvents);
    }
  }, [maxEvents]);

  const all = usePipecatEventStreamStore((state) => state.events);
  const paused = usePipecatEventStreamStore((state) => state.paused);
  const setPaused = usePipecatEventStreamStore((state) => state.setPaused);
  const clear = usePipecatEventStreamStore((state) => state.clear);

  // Filter arrays are typically inline literals; key the memo on contents.
  const includeKey = options?.includeEvents?.join("\0") ?? null;
  const ignoreKey = options?.ignoreEvents?.join("\0") ?? null;

  const events = useMemo(() => {
    if (includeKey !== null) {
      const include = new Set(includeKey === "" ? [] : includeKey.split("\0"));
      return all.filter((event) => include.has(event.type));
    }
    if (ignoreKey !== null) {
      const ignore = new Set(ignoreKey === "" ? [] : ignoreKey.split("\0"));
      return all.filter((event) => !ignore.has(event.type));
    }
    return all;
  }, [all, includeKey, ignoreKey]);

  const groupConsecutive = options?.groupConsecutive ?? false;
  const groupKey = options?.groupKey;
  const groups = useMemo(() => {
    if (!groupConsecutive) return [];
    const out: Array<PipecatEventGroup & { key: string }> = [];
    for (const event of events) {
      const key = groupKey ? groupKey(event) : event.type;
      const last = out[out.length - 1];
      if (last && last.key === key) {
        out[out.length - 1] = { ...last, events: [...last.events, event] };
      } else {
        out.push({ id: event.id, type: event.type, key, events: [event] });
      }
    }
    return out.map((group) => ({
      id: group.id,
      type: group.type,
      events: group.events,
    }));
  }, [events, groupConsecutive, groupKey]);

  // onEvent: fire only for events newer than anything seen at mount time.
  const onEventRef = useRef(options?.onEvent);
  onEventRef.current = options?.onEvent;
  const lastSeenSeqRef = useRef<number | null>(null);
  useEffect(() => {
    const tailSeq =
      events.length > 0 ? eventSeqOf(events[events.length - 1]!) : 0;
    if (lastSeenSeqRef.current === null) {
      lastSeenSeqRef.current = tailSeq;
      return;
    }
    const lastSeen = lastSeenSeqRef.current;
    if (onEventRef.current) {
      for (const event of events) {
        if (eventSeqOf(event) > lastSeen) onEventRef.current(event);
      }
    }
    lastSeenSeqRef.current = Math.max(lastSeen, tailSeq);
  }, [events]);

  return { events, groups, paused, setPaused, clear };
}
