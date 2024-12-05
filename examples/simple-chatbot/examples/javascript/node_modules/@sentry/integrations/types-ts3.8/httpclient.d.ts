import { Client, Integration, IntegrationClass } from '@sentry/types';
export type HttpStatusCodeRange = [
    number,
    number
] | number;
export type HttpRequestTarget = string | RegExp;
interface HttpClientOptions {
    /**
     * HTTP status codes that should be considered failed.
     * This array can contain tuples of `[begin, end]` (both inclusive),
     * single status codes, or a combinations of both
     *
     * Example: [[500, 505], 507]
     * Default: [[500, 599]]
     */
    failedRequestStatusCodes: HttpStatusCodeRange[];
    /**
     * Targets to track for failed requests.
     * This array can contain strings or regular expressions.
     *
     * Example: ['http://localhost', /api\/.*\/]
     * Default: [/.*\/]
     */
    failedRequestTargets: HttpRequestTarget[];
}
export declare const httpClientIntegration: (options?: Partial<HttpClientOptions> | undefined) => import("@sentry/types").IntegrationFnResult;
/**
 * Create events for failed client side HTTP requests.
 * @deprecated Use `httpClientIntegration()` instead.
 */
export declare const HttpClient: IntegrationClass<Integration & {
    setup: (client: Client) => void;
}> & (new (options?: {
    failedRequestStatusCodes: HttpStatusCodeRange[];
    failedRequestTargets: HttpRequestTarget[];
}) => Integration);
export {};
//# sourceMappingURL=httpclient.d.ts.map
