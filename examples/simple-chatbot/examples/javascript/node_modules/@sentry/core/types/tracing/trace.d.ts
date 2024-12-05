import type { Scope, Span, StartSpanOptions, TransactionContext } from '@sentry/types';
import { tracingContextFromHeaders } from '@sentry/utils';
/**
 * Wraps a function with a transaction/span and finishes the span after the function is done.
 *
 * Note that if you have not enabled tracing extensions via `addTracingExtensions`
 * or you didn't set `tracesSampleRate`, this function will not generate spans
 * and the `span` returned from the callback will be undefined.
 *
 * This function is meant to be used internally and may break at any time. Use at your own risk.
 *
 * @internal
 * @private
 *
 * @deprecated Use `startSpan` instead.
 */
export declare function trace<T>(context: TransactionContext, callback: (span?: Span) => T, onError?: (error: unknown, span?: Span) => void, afterFinish?: () => void): T;
/**
 * Wraps a function with a transaction/span and finishes the span after the function is done.
 * The created span is the active span and will be used as parent by other spans created inside the function
 * and can be accessed via `Sentry.getSpan()`, as long as the function is executed while the scope is active.
 *
 * If you want to create a span that is not set as active, use {@link startInactiveSpan}.
 *
 * Note that if you have not enabled tracing extensions via `addTracingExtensions`
 * or you didn't set `tracesSampleRate`, this function will not generate spans
 * and the `span` returned from the callback will be undefined.
 */
export declare function startSpan<T>(context: StartSpanOptions, callback: (span: Span | undefined) => T): T;
/**
 * @deprecated Use {@link startSpan} instead.
 */
export declare const startActiveSpan: typeof startSpan;
/**
 * Similar to `Sentry.startSpan`. Wraps a function with a transaction/span, but does not finish the span
 * after the function is done automatically. You'll have to call `span.end()` manually.
 *
 * The created span is the active span and will be used as parent by other spans created inside the function
 * and can be accessed via `Sentry.getActiveSpan()`, as long as the function is executed while the scope is active.
 *
 * Note that if you have not enabled tracing extensions via `addTracingExtensions`
 * or you didn't set `tracesSampleRate`, this function will not generate spans
 * and the `span` returned from the callback will be undefined.
 */
export declare function startSpanManual<T>(context: StartSpanOptions, callback: (span: Span | undefined, finish: () => void) => T): T;
/**
 * Creates a span. This span is not set as active, so will not get automatic instrumentation spans
 * as children or be able to be accessed via `Sentry.getSpan()`.
 *
 * If you want to create a span that is set as active, use {@link startSpan}.
 *
 * Note that if you have not enabled tracing extensions via `addTracingExtensions`
 * or you didn't set `tracesSampleRate` or `tracesSampler`, this function will not generate spans
 * and the `span` returned from the callback will be undefined.
 */
export declare function startInactiveSpan(context: StartSpanOptions): Span | undefined;
/**
 * Returns the currently active span.
 */
export declare function getActiveSpan(): Span | undefined;
interface ContinueTrace {
    /**
     * Continue a trace from `sentry-trace` and `baggage` values.
     * These values can be obtained from incoming request headers,
     * or in the browser from `<meta name="sentry-trace">` and `<meta name="baggage">` HTML tags.
     *
     * @deprecated Use the version of this function taking a callback as second parameter instead:
     *
     * ```
     * Sentry.continueTrace(sentryTrace: '...', baggage: '...' }, () => {
     *    // ...
     * })
     * ```
     *
     */
    ({ sentryTrace, baggage, }: {
        sentryTrace: Parameters<typeof tracingContextFromHeaders>[0];
        baggage: Parameters<typeof tracingContextFromHeaders>[1];
    }): Partial<TransactionContext>;
    /**
     * Continue a trace from `sentry-trace` and `baggage` values.
     * These values can be obtained from incoming request headers, or in the browser from `<meta name="sentry-trace">`
     * and `<meta name="baggage">` HTML tags.
     *
     * Spans started with `startSpan`, `startSpanManual` and `startInactiveSpan`, within the callback will automatically
     * be attached to the incoming trace.
     *
     * Deprecation notice: In the next major version of the SDK the provided callback will not receive a transaction
     * context argument.
     */
    <V>({ sentryTrace, baggage, }: {
        sentryTrace: Parameters<typeof tracingContextFromHeaders>[0];
        baggage: Parameters<typeof tracingContextFromHeaders>[1];
    }, callback: (transactionContext: Partial<TransactionContext>) => V): V;
}
export declare const continueTrace: ContinueTrace;
/**
 * Grabs the scope and isolation scope off a span that were active when the span was started.
 */
export declare function getCapturedScopesOnSpan(span: Span): {
    scope?: Scope;
    isolationScope?: Scope;
};
export {};
//# sourceMappingURL=trace.d.ts.map