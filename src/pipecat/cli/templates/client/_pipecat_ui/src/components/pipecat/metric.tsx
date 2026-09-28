"use client";

import * as React from "react";

import { cn } from "@/lib/utils";

export interface MetricProps {
  /** Tile label, e.g. "TTFB · tts". */
  label: React.ReactNode;
  /** Datapoint to display; null/undefined renders `empty` instead. */
  value?: number | null;
  /** Unit suffix rendered after the value, e.g. "ms". */
  unit?: string;
  /** Formats the numeric value (default: locale string, at most 1 decimal). */
  format?: (value: number) => string;
  /** Rendered in place of the value when there is none (default "–"). */
  empty?: React.ReactNode;
  className?: string;
}

/** Default number formatting: locale-aware, at most one decimal place. */
function formatMetricValue(value: number): string {
  return value.toLocaleString(undefined, { maximumFractionDigits: 1 });
}

/** Props-driven metric tile; no client or chart dependency. */
export function Metric({
  label,
  value,
  unit,
  format = formatMetricValue,
  empty = "–",
  className,
}: MetricProps) {
  const hasValue = typeof value === "number" && Number.isFinite(value);
  const formatted = React.useMemo(
    () => (hasValue ? format(value) : null),
    [hasValue, format, value],
  );
  return (
    <div
      data-slot="metric"
      data-state={hasValue ? "live" : "empty"}
      className={cn("flex min-w-0 flex-col gap-1", className)}
    >
      <span
        data-slot="metric-label"
        className="text-muted-foreground truncate text-xs"
      >
        {label}
      </span>
      <span
        data-slot="metric-value"
        className="text-2xl leading-none font-semibold tabular-nums"
      >
        {formatted !== null ? (
          <>
            {formatted}
            {unit ? (
              <span className="text-muted-foreground ml-1 text-sm font-normal">
                {unit}
              </span>
            ) : null}
          </>
        ) : (
          empty
        )}
      </span>
    </div>
  );
}
