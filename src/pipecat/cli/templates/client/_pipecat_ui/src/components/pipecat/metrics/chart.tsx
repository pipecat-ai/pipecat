"use client";

import * as React from "react";
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";

import type { MetricCategory, MetricSeries } from "@/hooks/use-pipecat-metrics";
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";
import { cn } from "@/lib/utils";

/** Display titles shared by the metrics blocks. */
export const METRIC_CATEGORY_TITLES: Record<MetricCategory, string> = {
  ttfb: "TTFB",
  ttfa: "TTFA",
  processing: "Processing time",
  characters: "TTS characters",
  stt_usage: "STT audio",
};

/** Chart heading unit suffix per category. */
export const METRIC_CATEGORY_UNITS: Record<MetricCategory, string> = {
  ttfb: " (ms)",
  ttfa: " (ms)",
  processing: " (ms)",
  characters: "",
  stt_usage: " (s)",
};

/** Processor names become ChartConfig keys, which become CSS variable names. */
function sanitizeKey(processor: string): string {
  return processor.replace(/[^a-zA-Z0-9_-]/g, "-");
}

function formatTime(time: number): string {
  return new Date(time).toLocaleTimeString(undefined, { hour12: false });
}

/**
 * Merges per-processor point series into recharts rows keyed by the metrics
 * batch timestamp (one RTVI metrics event carries all processors at once, so
 * timestamps align across series).
 */
function chartRows(
  seriesList: MetricSeries[],
  scale: number,
): Array<Record<string, number>> {
  const rows = new Map<number, Record<string, number>>();
  for (const series of seriesList) {
    const key = sanitizeKey(series.processor);
    for (const point of series.points) {
      const row = rows.get(point.time) ?? { time: point.time };
      row[key] = point.value * scale;
      rows.set(point.time, row);
    }
  }
  return [...rows.values()].sort((a, b) => a.time! - b.time!);
}

export interface MetricSeriesChartProps {
  /** Series drawn as one line per processor. */
  series: MetricSeries[];
  /** Multiplier applied to raw values before plotting (e.g. 1000 for s → ms). */
  scale?: number;
  className?: string;
}

/** Linear interpolation preserves sparse per-turn samples; colors use the chart palette. */
export function MetricSeriesChart({
  series,
  scale = 1,
  className,
}: MetricSeriesChartProps) {
  const rows = chartRows(series, scale);
  const config: ChartConfig = {};
  series.forEach((s, index) => {
    config[sanitizeKey(s.processor)] = {
      label: s.processor,
      color: `var(--chart-${(index % 5) + 1})`,
    };
  });

  return (
    <ChartContainer config={config} className={cn("h-48 w-full", className)}>
      <LineChart data={rows} accessibilityLayer>
        <CartesianGrid vertical={false} />
        <XAxis
          dataKey="time"
          tickFormatter={formatTime}
          tickLine={false}
          axisLine={false}
          tickMargin={8}
          minTickGap={32}
        />
        <YAxis width={40} tickLine={false} axisLine={false} tickMargin={4} />
        <ChartTooltip
          content={
            <ChartTooltipContent
              labelFormatter={(_, payload) => {
                const time = payload?.[0]?.payload?.time;
                return typeof time === "number" ? formatTime(time) : "";
              }}
            />
          }
        />
        {series.length > 1 && <ChartLegend content={<ChartLegendContent />} />}
        {series.map((s) => {
          const key = sanitizeKey(s.processor);
          return (
            <Line
              key={key}
              dataKey={key}
              type="linear"
              stroke={`var(--color-${key})`}
              strokeWidth={2}
              dot={{ r: 2 }}
              isAnimationActive={false}
              connectNulls
            />
          );
        })}
      </LineChart>
    </ChartContainer>
  );
}

export interface MetricsEmptyProps {
  children?: React.ReactNode;
  /** data-slot for the empty container, e.g. "metrics-performance". */
  slot: string;
  className?: string;
}

/** Shared empty state for the metrics blocks. */
export function MetricsEmpty({ children, slot, className }: MetricsEmptyProps) {
  return (
    <div
      data-slot={slot}
      data-state="empty"
      className={cn(
        "text-muted-foreground flex h-full min-h-24 flex-col items-center justify-center gap-1 text-sm",
        className,
      )}
    >
      {children ?? (
        <>
          <span>No metrics yet</span>
          <span className="text-xs">
            Metrics stream in once an agent session is live.
          </span>
        </>
      )}
    </div>
  );
}
