import type { Client, Event, EventHint, Integration, IntegrationClass } from '@sentry/types';
import type { TransactionNamingScheme } from '@sentry/utils';
export type RequestDataIntegrationOptions = {
    /**
     * Controls what data is pulled from the request and added to the event
     */
    include?: {
        cookies?: boolean;
        data?: boolean;
        headers?: boolean;
        ip?: boolean;
        query_string?: boolean;
        url?: boolean;
        user?: boolean | {
            id?: boolean;
            username?: boolean;
            email?: boolean;
        };
    };
    /** Whether to identify transactions by parameterized path, parameterized path with method, or handler name */
    transactionNamingScheme?: TransactionNamingScheme;
};
export declare const requestDataIntegration: (options?: RequestDataIntegrationOptions | undefined) => import("@sentry/types").IntegrationFnResult;
/**
 * Add data about a request to an event. Primarily for use in Node-based SDKs, but included in `@sentry/integrations`
 * so it can be used in cross-platform SDKs like `@sentry/nextjs`.
 * @deprecated Use `requestDataIntegration()` instead.
 */
export declare const RequestData: IntegrationClass<Integration & {
    processEvent: (event: Event, hint: EventHint, client: Client) => Event;
}> & (new (options?: {
    /**
     * Controls what data is pulled from the request and added to the event
     */
    include?: {
        cookies?: boolean;
        data?: boolean;
        headers?: boolean;
        ip?: boolean;
        query_string?: boolean;
        url?: boolean;
        user?: boolean | {
            id?: boolean;
            username?: boolean;
            email?: boolean;
        };
    };
    /** Whether to identify transactions by parameterized path, parameterized path with method, or handler name */
    transactionNamingScheme?: TransactionNamingScheme;
}) => Integration);
//# sourceMappingURL=requestdata.d.ts.map