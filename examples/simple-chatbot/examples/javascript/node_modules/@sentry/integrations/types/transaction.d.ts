import type { Event, Integration, IntegrationClass } from '@sentry/types';
/**
 * Add node transaction to the event.
 * @deprecated This integration will be removed in v8.
 */
export declare const Transaction: IntegrationClass<Integration & {
    processEvent: (event: Event) => Event;
}>;
//# sourceMappingURL=transaction.d.ts.map