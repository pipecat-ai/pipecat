import { MeasurementUnit } from './measurement';
import { Primitive } from './misc';
/**
 * An abstract definition of the minimum required API
 * for a metric instance.
 */
export interface MetricInstance {
    /**
     * Returns the weight of the metric.
     */
    weight: number;
    /**
     * Adds a value to a metric.
     */
    add(value: number | string): void;
    /**
     * Serializes the metric into a statsd format string.
     */
    toString(): string;
}
export interface MetricBucketItem {
    metric: MetricInstance;
    timestamp: number;
    metricType: 'c' | 'g' | 's' | 'd';
    name: string;
    unit: MeasurementUnit;
    tags: Record<string, string>;
}
/**
 * A metrics aggregator that aggregates metrics in memory and flushes them periodically.
 */
export interface MetricsAggregator {
    /**
     * Add a metric to the aggregator.
     */
    add(metricType: 'c' | 'g' | 's' | 'd', name: string, value: number | string, unit?: MeasurementUnit, tags?: Record<string, Primitive>, timestamp?: number): void;
    /**
     * Flushes the current metrics to the transport via the transport.
     */
    flush(): void;
    /**
     * Shuts down metrics aggregator and clears all metrics.
     */
    close(): void;
    /**
     * Returns a string representation of the aggregator.
     */
    toString(): string;
}
//# sourceMappingURL=metrics.d.ts.map
