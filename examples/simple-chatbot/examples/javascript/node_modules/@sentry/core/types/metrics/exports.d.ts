import type { ClientOptions, MeasurementUnit, Primitive } from '@sentry/types';
interface MetricData {
    unit?: MeasurementUnit;
    tags?: Record<string, Primitive>;
    timestamp?: number;
}
/**
 * Adds a value to a counter metric
 *
 * @experimental This API is experimental and might have breaking changes in the future.
 */
export declare function increment(name: string, value?: number, data?: MetricData): void;
/**
 * Adds a value to a distribution metric
 *
 * @experimental This API is experimental and might have breaking changes in the future.
 */
export declare function distribution(name: string, value: number, data?: MetricData): void;
/**
 * Adds a value to a set metric. Value must be a string or integer.
 *
 * @experimental This API is experimental and might have breaking changes in the future.
 */
export declare function set(name: string, value: number | string, data?: MetricData): void;
/**
 * Adds a value to a gauge metric
 *
 * @experimental This API is experimental and might have breaking changes in the future.
 */
export declare function gauge(name: string, value: number, data?: MetricData): void;
export declare const metrics: {
    increment: typeof increment;
    distribution: typeof distribution;
    set: typeof set;
    gauge: typeof gauge;
    /** @deprecated Use `metrics.metricsAggregratorIntegration()` instead. */
    MetricsAggregator: import("@sentry/types").IntegrationClass<import("@sentry/types").Integration & {
        setup: (client: import("@sentry/types").Client<ClientOptions<import("@sentry/types").BaseTransportOptions>>) => void;
    }>;
    metricsAggregatorIntegration: () => import("@sentry/types").IntegrationFnResult;
};
export {};
//# sourceMappingURL=exports.d.ts.map