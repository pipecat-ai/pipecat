import type { Client, Event, EventHint, Integration, IntegrationClass } from '@sentry/types';
interface LinkedErrorsOptions {
    key?: string;
    limit?: number;
}
export declare const linkedErrorsIntegration: (options?: LinkedErrorsOptions | undefined) => import("@sentry/types").IntegrationFnResult;
/**
 * Adds SDK info to an event.
 * @deprecated Use `linkedErrorsIntegration()` instead.
 */
export declare const LinkedErrors: IntegrationClass<Integration & {
    preprocessEvent: (event: Event, hint: EventHint, client: Client) => void;
}> & (new (options?: {
    key?: string;
    limit?: number;
}) => Integration);
export {};
//# sourceMappingURL=linkederrors.d.ts.map