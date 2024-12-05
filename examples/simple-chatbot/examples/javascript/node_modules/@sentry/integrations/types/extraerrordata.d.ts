import type { Event, EventHint, Integration, IntegrationClass } from '@sentry/types';
interface ExtraErrorDataOptions {
    /**
     * The object depth up to which to capture data on error objects.
     */
    depth: number;
    /**
     * Whether to capture error causes.
     *
     * More information: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Error/cause
     */
    captureErrorCause: boolean;
}
export declare const extraErrorDataIntegration: (options?: Partial<ExtraErrorDataOptions> | undefined) => import("@sentry/types").IntegrationFnResult;
/**
 * Extract additional data for from original exceptions.
 * @deprecated Use `extraErrorDataIntegration()` instead.
 */
export declare const ExtraErrorData: IntegrationClass<Integration & {
    processEvent: (event: Event, hint: EventHint) => Event;
}> & (new (options?: Partial<{
    depth: number;
    captureErrorCause: boolean;
}>) => Integration);
export {};
//# sourceMappingURL=extraerrordata.d.ts.map