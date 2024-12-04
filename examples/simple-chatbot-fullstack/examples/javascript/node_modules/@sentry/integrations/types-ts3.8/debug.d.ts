import { Client, Integration, IntegrationClass } from '@sentry/types';
interface DebugOptions {
    /** Controls whether console output created by this integration should be stringified. Default: `false` */
    stringify?: boolean;
    /** Controls whether a debugger should be launched before an event is sent. Default: `false` */
    debugger?: boolean;
}
export declare const debugIntegration: (options?: DebugOptions | undefined) => import("@sentry/types").IntegrationFnResult;
/**
 * Integration to debug sent Sentry events.
 * This integration should not be used in production.
 *
 * @deprecated Use `debugIntegration()` instead.
 */
export declare const Debug: IntegrationClass<Integration & {
    setup: (client: Client) => void;
}> & (new (options?: {
    stringify?: boolean;
    debugger?: boolean;
}) => Integration);
export {};
//# sourceMappingURL=debug.d.ts.map
