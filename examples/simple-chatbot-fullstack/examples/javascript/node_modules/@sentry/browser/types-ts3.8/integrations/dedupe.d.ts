import { Event, Integration, IntegrationClass } from '@sentry/types';
export declare const dedupeIntegration: () => import("@sentry/types").IntegrationFnResult;
/**
 * Deduplication filter.
 * @deprecated Use `dedupeIntegration()` instead.
 */
export declare const Dedupe: IntegrationClass<Integration & {
    processEvent: (event: Event) => Event;
}>;
//# sourceMappingURL=dedupe.d.ts.map
