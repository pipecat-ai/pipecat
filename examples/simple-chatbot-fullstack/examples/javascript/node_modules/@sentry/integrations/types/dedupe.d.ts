import type { Event, Integration, IntegrationClass } from '@sentry/types';
export declare const dedupeIntegration: () => import("@sentry/types").IntegrationFnResult;
/**
 * Deduplication filter.
 * @deprecated Use `dedupeIntegration()` instead.
 */
export declare const Dedupe: IntegrationClass<Integration & {
    processEvent: (event: Event) => Event;
}>;
/** only exported for tests. */
export declare function _shouldDropEvent(currentEvent: Event, previousEvent?: Event): boolean;
//# sourceMappingURL=dedupe.d.ts.map