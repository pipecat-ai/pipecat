import type { Client, Event, EventHint, Integration, IntegrationClass } from '@sentry/types';
/** Options for the InboundFilters integration */
export interface InboundFiltersOptions {
    allowUrls: Array<string | RegExp>;
    denyUrls: Array<string | RegExp>;
    ignoreErrors: Array<string | RegExp>;
    ignoreTransactions: Array<string | RegExp>;
    ignoreInternal: boolean;
    disableErrorDefaults: boolean;
    disableTransactionDefaults: boolean;
}
export declare const inboundFiltersIntegration: (options?: Partial<InboundFiltersOptions> | undefined) => import("@sentry/types").IntegrationFnResult;
/**
 * Inbound filters configurable by the user.
 * @deprecated Use `inboundFiltersIntegration()` instead.
 */
export declare const InboundFilters: IntegrationClass<Integration & {
    preprocessEvent: (event: Event, hint: EventHint, client: Client) => void;
}> & (new (options?: Partial<{
    allowUrls: Array<string | RegExp>;
    denyUrls: Array<string | RegExp>;
    ignoreErrors: Array<string | RegExp>;
    ignoreTransactions: Array<string | RegExp>;
    ignoreInternal: boolean;
    disableErrorDefaults: boolean;
    disableTransactionDefaults: boolean;
}>) => Integration);
//# sourceMappingURL=inboundfilters.d.ts.map