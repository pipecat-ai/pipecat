import type { Event, Integration, IntegrationClass } from '@sentry/types';
export declare const sessionTimingIntegration: () => import("@sentry/types").IntegrationFnResult;
/**
 * This function adds duration since Sentry was initialized till the time event was sent.
 * @deprecated Use `sessionTimingIntegration()` instead.
 */
export declare const SessionTiming: IntegrationClass<Integration & {
    processEvent: (event: Event) => Event;
}>;
//# sourceMappingURL=sessiontiming.d.ts.map