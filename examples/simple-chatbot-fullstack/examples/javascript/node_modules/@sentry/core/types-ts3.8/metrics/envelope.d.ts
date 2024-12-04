import { DsnComponents, MetricBucketItem, SdkMetadata, StatsdEnvelope } from '@sentry/types';
/**
 * Create envelope from a metric aggregate.
 */
export declare function createMetricEnvelope(metricBucketItems: Array<MetricBucketItem>, dsn?: DsnComponents, metadata?: SdkMetadata, tunnel?: string): StatsdEnvelope;
//# sourceMappingURL=envelope.d.ts.map
