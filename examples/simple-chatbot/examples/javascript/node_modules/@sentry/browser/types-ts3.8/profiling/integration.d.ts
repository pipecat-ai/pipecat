import { Client, Integration, IntegrationClass } from '@sentry/types';
export declare const browserProfilingIntegration: () => import("@sentry/types").IntegrationFnResult;
/**
 * Browser profiling integration. Stores any event that has contexts["profile"]["profile_id"]
 * This exists because we do not want to await async profiler.stop calls as transaction.finish is called
 * in a synchronous context. Instead, we handle sending the profile async from the promise callback and
 * rely on being able to pull the event from the cache when we need to construct the envelope. This makes the
 * integration less reliable as we might be dropping profiles when the cache is full.
 *
 * @experimental
 * @deprecated Use `browserProfilingIntegration()` instead.
 */
export declare const BrowserProfilingIntegration: IntegrationClass<Integration & {
    setup: (client: Client) => void;
}>;
export type BrowserProfilingIntegration = typeof BrowserProfilingIntegration;
//# sourceMappingURL=integration.d.ts.map
