import type { Transaction } from '@sentry/types';
/**
 * Safety wrapper for startTransaction for the unlikely case that transaction starts before tracing is imported -
 * if that happens we want to avoid throwing an error from profiling code.
 * see https://github.com/getsentry/sentry-javascript/issues/4731.
 *
 * @experimental
 */
export declare function onProfilingStartRouteTransaction(transaction: Transaction | undefined): Transaction | undefined;
/**
 * Wraps startTransaction and stopTransaction with profiling related logic.
 * startProfileForTransaction is called after the call to startTransaction in order to avoid our own code from
 * being profiled. Because of that same reason, stopProfiling is called before the call to stopTransaction.
 */
export declare function startProfileForTransaction(transaction: Transaction): Transaction;
//# sourceMappingURL=hubextensions.d.ts.map