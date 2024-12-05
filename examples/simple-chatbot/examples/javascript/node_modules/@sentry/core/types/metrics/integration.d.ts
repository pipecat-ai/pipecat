import type { Client, Integration, IntegrationClass } from '@sentry/types';
export declare const metricsAggregatorIntegration: () => import("@sentry/types").IntegrationFnResult;
/**
 * Enables Sentry metrics monitoring.
 *
 * @experimental This API is experimental and might having breaking changes in the future.
 * @deprecated Use `metricsAggegratorIntegration()` instead.
 */
export declare const MetricsAggregator: IntegrationClass<Integration & {
    setup: (client: Client) => void;
}>;
//# sourceMappingURL=integration.d.ts.map