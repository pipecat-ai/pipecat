import type { Integration, IntegrationClass } from '@sentry/types';
/**
 * Patch toString calls to return proper name for wrapped functions.
 *
 * ```js
 * Sentry.init({
 *   integrations: [
 *     functionToStringIntegration(),
 *   ],
 * });
 * ```
 */
export declare const functionToStringIntegration: () => import("@sentry/types").IntegrationFnResult;
/**
 * Patch toString calls to return proper name for wrapped functions.
 *
 * @deprecated Use `functionToStringIntegration()` instead.
 */
export declare const FunctionToString: IntegrationClass<Integration & {
    setupOnce: () => void;
}>;
export type FunctionToString = typeof FunctionToString;
//# sourceMappingURL=functiontostring.d.ts.map