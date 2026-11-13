"use client";

import { RTVIEvent } from "@pipecat-ai/client-js";
import {
  usePipecatClientTransportState,
  useRTVIClientEvent,
} from "@pipecat-ai/client-react";
import { Loader2Icon } from "lucide-react";
import { useEffect, useState } from "react";

import { cn } from "@/lib/utils";

const CONNECTING_STATES = [
  "initializing",
  "authenticating",
  "authenticated",
  "connecting",
];

export interface ClientStatusValueProps {
  /** Transport-like state string to display. */
  state?: string | null;
  className?: string;
}

/** One status value, colored and animated by connection state. */
export function ClientStatusValue({
  state,
  className,
}: ClientStatusValueProps) {
  return (
    <span
      className={cn(
        "text-muted-foreground flex items-center justify-end gap-1.5 font-mono text-xs leading-none font-medium tracking-wide uppercase",
        {
          "text-active": state === "connected" || state === "ready",
          "text-destructive": state === "error",
          "text-muted-foreground/60": state === "disconnected" || !state,
          "animate-pulse": CONNECTING_STATES.includes(state || ""),
        },
        className,
      )}
    >
      {state || "—"}
      {state &&
        ["authenticating", "authenticated", "connecting"].includes(state) && (
          <Loader2Icon className="size-3 animate-spin" />
        )}
    </span>
  );
}

export interface ClientStatusProps {
  /** Hide the agent (bot) status row. */
  noAgentState?: boolean;
  /** Hide the client (transport) status row. */
  noClientState?: boolean;
  className?: string;
}

/**
 * Client and agent connection status rows, live-updated from transport
 * state and bot lifecycle events.
 * Must be rendered inside a PipecatClientProvider.
 */
export function ClientStatus({
  noAgentState = false,
  noClientState = false,
  className,
}: ClientStatusProps) {
  const transportState = usePipecatClientTransportState();

  const [botStatus, setBotStatus] = useState<
    "disconnected" | "connecting" | "connected" | "ready" | null
  >(null);

  useEffect(() => {
    if (transportState === "connecting") {
      setBotStatus("connecting");
    }
  }, [transportState]);

  useRTVIClientEvent(RTVIEvent.BotReady, () => setBotStatus("ready"));
  useRTVIClientEvent(RTVIEvent.BotConnected, () => setBotStatus("connected"));
  useRTVIClientEvent(RTVIEvent.Disconnected, () =>
    setBotStatus("disconnected"),
  );
  useRTVIClientEvent(RTVIEvent.BotDisconnected, () =>
    setBotStatus("disconnected"),
  );

  if (noAgentState && noClientState) return null;

  return (
    <dl
      data-slot="client-status"
      className={cn(
        "grid grid-cols-[auto_minmax(0,1fr)] items-center gap-2 text-sm",
        className,
      )}
    >
      {!noClientState && (
        <>
          <dt className="text-muted-foreground">Client</dt>
          <dd className="min-w-0">
            <ClientStatusValue state={transportState} />
          </dd>
        </>
      )}
      {!noAgentState && (
        <>
          <dt className="text-muted-foreground">Agent</dt>
          <dd className="min-w-0">
            <ClientStatusValue state={botStatus} />
          </dd>
        </>
      )}
    </dl>
  );
}
