"use client";

import * as React from "react";

import { Metric } from "@/components/pipecat/metric";
import {
  METRIC_CATEGORY_TITLES,
  METRIC_CATEGORY_UNITS,
  MetricSeriesChart,
  MetricsEmpty,
} from "@/components/pipecat/metrics/chart";
import {
  usePipecatMetrics,
  type MetricSeries,
} from "@/hooks/use-pipecat-metrics";
import { cn } from "@/lib/utils";

export type MetricsPerformanceCategory = "ttfb" | "ttfa" | "processing";

export interface MetricsPerformanceViewProps {
  /** Metric series; non-performance categories are ignored. */
  series: MetricSeries[];
  /** Which categories get tiles and charts (default ["ttfb"]). */
  categories?: MetricsPerformanceCategory[];
  /** Hides the tile grid. Default false. */
  noTiles?: boolean;
  /** Hides the charts. Default false. */
  noCharts?: boolean;
  /** Processor names excluded from tiles and charts. */
  ignoreProcessors?: string[];
  /** Rendered when there is nothing to show yet. */
  empty?: React.ReactNode;
  className?: string;
}

/** Latency tiles and charts in milliseconds. ttfa requires Pipecat server ≥1.7. */
export function MetricsPerformanceView({
  series,
  categories = ["ttfb"],
  noTiles = false,
  noCharts = false,
  ignoreProcessors,
  empty,
  className,
}: MetricsPerformanceViewProps) {
  const visible = React.useMemo(() => {
    const bySelection = series.filter((s) =>
      (categories as string[]).includes(s.category),
    );
    return ignoreProcessors?.length
      ? bySelection.filter((s) => !ignoreProcessors.includes(s.processor))
      : bySelection;
  }, [series, categories, ignoreProcessors]);

  if (visible.length === 0) {
    return (
      <MetricsEmpty slot="metrics-performance" className={className}>
        {empty}
      </MetricsEmpty>
    );
  }

  return (
    <div
      data-slot="metrics-performance"
      data-state="live"
      className={cn("@container/metrics flex flex-col gap-6", className)}
    >
      {!noTiles && (
        <div
          data-slot="metrics-tiles"
          className="grid grid-cols-2 gap-4 @md/metrics:grid-cols-3 @xl/metrics:grid-cols-4"
        >
          {visible.map((s) => (
            <Metric
              key={`${s.category}:${s.processor}`}
              label={`${METRIC_CATEGORY_TITLES[s.category]} · ${s.processor}`}
              value={s.latest * 1000}
              unit="ms"
            />
          ))}
        </div>
      )}

      {!noCharts &&
        categories.map((category) => {
          const categorySeries = visible.filter((s) => s.category === category);
          if (categorySeries.length === 0) return null;
          return (
            <div
              key={category}
              data-slot="metrics-chart"
              className="flex flex-col gap-2"
            >
              <h3 className="text-muted-foreground text-xs font-medium">
                {METRIC_CATEGORY_TITLES[category]}
                {METRIC_CATEGORY_UNITS[category]}
              </h3>
              <MetricSeriesChart series={categorySeries} scale={1000} />
            </div>
          );
        })}
    </div>
  );
}

export type MetricsPerformanceProps = Omit<
  MetricsPerformanceViewProps,
  "series"
>;

/**
 * Performance metrics wired to the shared RTVI metrics store — populates
 * live and resets with each new session. Must be rendered inside a
 * PipecatClientProvider.
 */
export function MetricsPerformance(props: MetricsPerformanceProps) {
  const { series } = usePipecatMetrics();
  return <MetricsPerformanceView series={series} {...props} />;
}
