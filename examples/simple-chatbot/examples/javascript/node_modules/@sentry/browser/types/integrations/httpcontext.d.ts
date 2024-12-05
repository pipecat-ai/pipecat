import type { Event, Integration, IntegrationClass } from '@sentry/types';
export declare const httpContextIntegration: () => import("@sentry/types").IntegrationFnResult;
/**
 * HttpContext integration collects information about HTTP request headers.
 * @deprecated Use `httpContextIntegration()` instead.
 */
export declare const HttpContext: IntegrationClass<Integration & {
    preprocessEvent: (event: Event) => void;
}>;
//# sourceMappingURL=httpcontext.d.ts.map