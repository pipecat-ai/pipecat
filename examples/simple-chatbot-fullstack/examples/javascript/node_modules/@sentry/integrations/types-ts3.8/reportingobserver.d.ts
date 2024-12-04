import { Client, Integration, IntegrationClass } from '@sentry/types';
type ReportTypes = 'crash' | 'deprecation' | 'intervention';
interface ReportingObserverOptions {
    types?: ReportTypes[];
}
export declare const reportingObserverIntegration: (options?: ReportingObserverOptions | undefined) => import("@sentry/types").IntegrationFnResult;
/**
 * Reporting API integration - https://w3c.github.io/reporting/
 * @deprecated Use `reportingObserverIntegration()` instead.
 */
export declare const ReportingObserver: IntegrationClass<Integration & {
    setup: (client: Client) => void;
}> & (new (options?: {
    types?: ReportTypes[];
}) => Integration);
export {};
//# sourceMappingURL=reportingobserver.d.ts.map
