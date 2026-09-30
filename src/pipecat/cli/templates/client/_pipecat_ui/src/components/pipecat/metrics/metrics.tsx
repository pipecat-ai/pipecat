"use client";

import * as React from "react";

import {
  MetricsPerformance,
  MetricsPerformanceView,
  type MetricsPerformanceProps,
  type MetricsPerformanceViewProps,
} from "@/components/pipecat/metrics/performance";
import {
  MetricsUsage,
  MetricsUsageView,
  type MetricsUsageProps,
  type MetricsUsageViewProps,
} from "@/components/pipecat/metrics/usage";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { MetricSeries, TokenTotals } from "@/hooks/use-pipecat-metrics";
import { cn } from "@/lib/utils";

export type MetricsTab = "performance" | "usage";

export interface MetricsViewProps {
  /** Metric series for both sections (see use-pipecat-metrics). */
  series: MetricSeries[];
  /** Running token totals for the usage section. */
  tokens?: TokenTotals;
  /** Which tab is selected initially (default "performance"). */
  defaultTab?: MetricsTab;
  /** Overrides for the performance section. */
  performanceProps?: Omit<MetricsPerformanceViewProps, "series">;
  /** Overrides for the usage section. */
  usageProps?: Omit<MetricsUsageViewProps, "series" | "tokens">;
  className?: string;
}

function MetricsTabs({
  defaultTab = "performance",
  className,
  performance,
  usage,
}: {
  defaultTab?: MetricsTab;
  className?: string;
  performance: React.ReactNode;
  usage: React.ReactNode;
}) {
  return (
    <Tabs
      data-slot="metrics"
      defaultValue={defaultTab}
      className={cn("flex flex-col gap-4", className)}
    >
      <TabsList>
        <TabsTrigger value="performance">Performance</TabsTrigger>
        <TabsTrigger value="usage">Usage</TabsTrigger>
      </TabsList>
      <TabsContent value="performance">{performance}</TabsContent>
      <TabsContent value="usage">{usage}</TabsContent>
    </Tabs>
  );
}

/** Props-driven performance and usage tabs. */
export function MetricsView({
  series,
  tokens,
  defaultTab,
  performanceProps,
  usageProps,
  className,
}: MetricsViewProps) {
  return (
    <MetricsTabs
      defaultTab={defaultTab}
      className={className}
      performance={
        <MetricsPerformanceView series={series} {...performanceProps} />
      }
      usage={
        <MetricsUsageView series={series} tokens={tokens} {...usageProps} />
      }
    />
  );
}

export interface MetricsProps {
  /** Which tab is selected initially (default "performance"). */
  defaultTab?: MetricsTab;
  /** Overrides for the performance section. */
  performanceProps?: MetricsPerformanceProps;
  /** Overrides for the usage section. */
  usageProps?: MetricsUsageProps;
  className?: string;
}

/** Requires PipecatClientProvider. Keeps collection active while individual tabs are unmounted. */
export function Metrics({
  defaultTab,
  performanceProps,
  usageProps,
  className,
}: MetricsProps) {
  return (
    <MetricsTabs
      defaultTab={defaultTab}
      className={className}
      performance={<MetricsPerformance {...performanceProps} />}
      usage={<MetricsUsage {...usageProps} />}
    />
  );
}
