"use client";

import { RTVIEvent, type BotOutputData } from "@pipecat-ai/client-js";
import { PauseIcon, PlayIcon, SearchIcon, Trash2Icon } from "lucide-react";
import * as React from "react";

import {
  ConsolePanel,
  ConsolePanelActions,
  ConsolePanelContent,
  ConsolePanelHeader,
  ConsolePanelTitle,
} from "@/components/pipecat/console/panel";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  usePipecatEventStream,
  type PipecatEventLog,
} from "@/hooks/use-pipecat-event-stream";
import { cn } from "@/lib/utils";

const TIME_FORMAT = new Intl.DateTimeFormat(undefined, {
  hour12: false,
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  fractionalSecondDigits: 3,
});

function summarizeBotOutput(data: Partial<BotOutputData>): string {
  const spoken =
    data.will_be_spoken === false
      ? "not spoken"
      : data.spoken_status !== undefined
        ? `spoken: ${data.spoken_status}`
        : data.spoken !== undefined
          ? `spoken: ${data.spoken}`
          : null;
  const details = [
    data.aggregated_by,
    data.segment_id !== undefined ? `#${data.segment_id}` : null,
    spoken,
  ].filter(Boolean);
  return `(${details.join(", ")}): ${data.text ?? ""}`;
}

function summarize(event: PipecatEventLog): string {
  const { data } = event;
  if (data === undefined) return "";
  if (event.type === RTVIEvent.BotOutput && typeof data === "object") {
    return summarizeBotOutput(data as Partial<BotOutputData>);
  }
  try {
    const json = JSON.stringify(data);
    return json.length > 120 ? `${json.slice(0, 120)}…` : json;
  } catch {
    return String(data);
  }
}

/** Chatter covered by botOutput rows and the metrics panel. */
const HIDDEN_EVENTS: string[] = [
  RTVIEvent.Metrics,
  RTVIEvent.BotTranscript,
  RTVIEvent.BotLlmStarted,
  RTVIEvent.BotLlmText,
  RTVIEvent.BotLlmStopped,
  RTVIEvent.BotTtsStarted,
  RTVIEvent.BotTtsText,
  RTVIEvent.BotTtsStopped,
];

/**
 * RTVI 2.0.0+ re-sends a segment's botOutput on every spoken progress update
 * (about once per TTS word). Keep one row per spoken status change, so an
 * interrupted segment shows as new/in-progress without a completed row. Keys
 * on segment_id, never text: separate responses can share wording.
 */
function dropRepeatedBotOutput(
  events: readonly PipecatEventLog[],
): PipecatEventLog[] {
  const statuses = new Map<number, string>();
  return events.filter((event) => {
    if (event.type !== RTVIEvent.BotOutput) return true;
    const data = event.data as Partial<BotOutputData> | undefined;
    if (data?.segment_id === undefined || data.spoken_status === undefined) {
      return true;
    }
    if (statuses.get(data.segment_id) === data.spoken_status) return false;
    statuses.set(data.segment_id, data.spoken_status);
    return true;
  });
}

function EventRow({ event }: { event: PipecatEventLog }) {
  const [expanded, setExpanded] = React.useState(false);
  const summary = summarize(event);
  return (
    <div data-slot="console-event" className="font-mono text-xs">
      <button
        type="button"
        onClick={() => setExpanded((value) => !value)}
        className="hover:bg-muted/50 grid w-full grid-cols-[min-content_min-content_1fr] items-baseline gap-x-3 rounded-sm px-1 py-0.5 text-left"
        aria-expanded={expanded}
      >
        <span className="text-muted-foreground whitespace-nowrap">
          {TIME_FORMAT.format(event.timestamp)}
        </span>
        <span className="font-semibold whitespace-nowrap">{event.type}</span>
        <span className="text-muted-foreground truncate">{summary}</span>
      </button>
      {expanded && event.data !== undefined && (
        <pre className="bg-muted/50 my-1 ml-4 overflow-x-auto rounded-sm p-2">
          {JSON.stringify(event.data, null, 2)}
        </pre>
      )}
    </div>
  );
}

export interface ConsoleEventsPanelProps {
  /** Compact rendering for a collapsed pane: header strip only. */
  collapsed?: boolean;
  className?: string;
}

/**
 * Live RTVI event log over the shared use-pipecat-event-stream store:
 * filter-as-you-type, pause/resume, clear, click-to-expand payloads, and
 * scroll pinning that follows the tail until you scroll away. Metrics and bot
 * LLM/TTS events are hidden, and botOutput shows once per spoken status.
 * Capture is shared, so a collapsed panel misses nothing. Must be rendered
 * inside a PipecatClientProvider.
 */
export function ConsoleEventsPanel({
  collapsed = false,
  className,
}: ConsoleEventsPanelProps) {
  const { events, paused, setPaused, clear } = usePipecatEventStream({
    ignoreEvents: HIDDEN_EVENTS,
  });
  const [filter, setFilter] = React.useState("");

  const filtered = React.useMemo(() => {
    const needle = filter.trim().toLowerCase();
    const shown = dropRepeatedBotOutput(events);
    if (!needle) return shown;
    return shown.filter((event) => event.type.toLowerCase().includes(needle));
  }, [events, filter]);

  // Scroll pinning: stick to the tail unless the user scrolled away.
  const scrollRef = React.useRef<HTMLDivElement | null>(null);
  const pinnedRef = React.useRef(true);
  const handleScroll = React.useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    pinnedRef.current =
      Math.ceil(el.scrollHeight - el.scrollTop) <=
      Math.ceil(el.clientHeight) + 4;
  }, []);
  React.useEffect(() => {
    const el = scrollRef.current;
    if (el && pinnedRef.current) {
      el.scrollTo({ top: el.scrollHeight, behavior: "instant" });
    }
  }, [filtered]);

  return (
    <ConsolePanel
      className={className}
      data-slot="console-events-panel"
      data-state={collapsed ? "collapsed" : "expanded"}
    >
      <ConsolePanelHeader>
        <ConsolePanelTitle>Events</ConsolePanelTitle>
        {!collapsed && (
          <ConsolePanelActions>
            <div className="relative hidden @xs/panel:block">
              <SearchIcon className="text-muted-foreground pointer-events-none absolute top-1/2 left-2 size-3.5 -translate-y-1/2" />
              <Input
                value={filter}
                onChange={(event) => setFilter(event.target.value)}
                placeholder="Filter events"
                aria-label="Filter events"
                className="h-7 w-40 pl-7 text-xs"
              />
            </div>
            <Button
              variant="ghost"
              size="icon-sm"
              aria-label={
                paused ? "Resume event capture" : "Pause event capture"
              }
              aria-pressed={paused}
              onClick={() => setPaused(!paused)}
            >
              {paused ? <PlayIcon /> : <PauseIcon />}
            </Button>
            <Button
              variant="ghost"
              size="icon-sm"
              aria-label="Clear events"
              onClick={clear}
            >
              <Trash2Icon />
            </Button>
          </ConsolePanelActions>
        )}
      </ConsolePanelHeader>
      {!collapsed && (
        <ConsolePanelContent
          ref={scrollRef}
          onScroll={handleScroll}
          className={cn("flex flex-col gap-0.5")}
        >
          {filtered.length === 0 ? (
            <div className="text-muted-foreground flex h-full min-h-16 items-center justify-center text-xs">
              {!filter.trim()
                ? "Events appear once a session is live."
                : "No events match the filter."}
            </div>
          ) : (
            filtered.map((event) => <EventRow key={event.id} event={event} />)
          )}
        </ConsolePanelContent>
      )}
    </ConsolePanel>
  );
}
