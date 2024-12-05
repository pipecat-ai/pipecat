import type { Integration, IntegrationClass } from '@sentry/types';
interface TryCatchOptions {
    setTimeout: boolean;
    setInterval: boolean;
    requestAnimationFrame: boolean;
    XMLHttpRequest: boolean;
    eventTarget: boolean | string[];
}
export declare const browserApiErrorsIntegration: (options?: Partial<TryCatchOptions> | undefined) => import("@sentry/types").IntegrationFnResult;
/**
 * Wrap timer functions and event targets to catch errors and provide better meta data.
 * @deprecated Use `browserApiErrorsIntegration()` instead.
 */
export declare const TryCatch: IntegrationClass<Integration> & (new (options?: {
    setTimeout: boolean;
    setInterval: boolean;
    requestAnimationFrame: boolean;
    XMLHttpRequest: boolean;
    eventTarget: boolean | string[];
}) => Integration);
export {};
//# sourceMappingURL=trycatch.d.ts.map