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
  type TokenTotals,
} from "@/hooks/use-pipecat-metrics";
import { cn } from "@/lib/utils";

export type MetricsUsageCategory = "characters" | "stt_usage";

const USAGE_UNITS: Record<MetricsUsageCategory, string | undefined> = {
  characters: undefined,
  stt_usage: "s",
};

export interface MetricsUsageViewProps {
  /** Metric series; non-usage categories are ignored. */
  series: MetricSeries[];
  /** Running token totals; omit to hide the token tiles. */
  tokens?: TokenTotals;
  /** Which categories get charts (default both). */
  categories?: MetricsUsageCategory[];
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

/** Session usage totals. STT audio requires Pipecat server ≥1.7 with usage metrics enabled. */
export function MetricsUsageView({
  series,
  tokens,
  categories = ["characters", "stt_usage"],
  noTiles = false,
  noCharts = false,
  ignoreProcessors,
  empty,
  className,
}: MetricsUsageViewProps) {
  const visible = React.useMemo(() => {
    const bySelection = series.filter(
      (s) => s.category === "characters" || s.category === "stt_usage",
    );
    return ignoreProcessors?.length
      ? bySelection.filter((s) => !ignoreProcessors.includes(s.processor))
      : bySelection;
  }, [series, ignoreProcessors]);

  if (visible.length === 0 && !tokens) {
    return (
      <MetricsEmpty slot="metrics-usage" className={className}>
        {empty}
      </MetricsEmpty>
    );
  }

  return (
    <div
      data-slot="metrics-usage"
      data-state="live"
      className={cn("@container/metrics flex flex-col gap-6", className)}
    >
      {!noTiles && (
        <div
          data-slot="metrics-tiles"
          className="grid grid-cols-2 gap-4 @md/metrics:grid-cols-3 @xl/metrics:grid-cols-4"
        >
          {tokens ? (
            <>
              <Metric label="Prompt tokens" value={tokens.prompt} />
              <Metric label="Completion tokens" value={tokens.completion} />
              <Metric label="Total tokens" value={tokens.total} />
              {tokens.cacheRead > 0 && (
                <Metric label="Cache read tokens" value={tokens.cacheRead} />
              )}
              {tokens.reasoning > 0 && (
                <Metric label="Reasoning tokens" value={tokens.reasoning} />
              )}
            </>
          ) : null}
          {visible.map((s) => (
            <Metric
              key={`${s.category}:${s.processor}`}
              label={`${METRIC_CATEGORY_TITLES[s.category]} · ${s.processor}`}
              value={s.latest}
              unit={USAGE_UNITS[s.category as MetricsUsageCategory]}
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
              <MetricSeriesChart series={categorySeries} />
            </div>
          );
        })}
    </div>
  );
}

export type MetricsUsageProps = Omit<
  MetricsUsageViewProps,
  "series" | "tokens"
>;

/**
 * Usage metrics wired to the shared RTVI metrics store — running totals
 * populate live and reset with each new session. Must be rendered inside a
 * PipecatClientProvider.
 */
export function MetricsUsage(props: MetricsUsageProps) {
  const { series, tokens, hasTokens } = usePipecatMetrics();
  return (
    <MetricsUsageView
      series={series}
      tokens={hasTokens ? tokens : undefined}
      {...props}
    />
  );
}
