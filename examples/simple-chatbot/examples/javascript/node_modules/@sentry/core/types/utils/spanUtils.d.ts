import type { Span, SpanJSON, SpanTimeInput, TraceContext } from '@sentry/types';
export declare const TRACE_FLAG_NONE = 0;
export declare const TRACE_FLAG_SAMPLED = 1;
/**
 * Convert a span to a trace context, which can be sent as the `trace` context in an event.
 */
export declare function spanToTraceContext(span: Span): TraceContext;
/**
 * Convert a Span to a Sentry trace header.
 */
export declare function spanToTraceHeader(span: Span): string;
/**
 * Convert a span time input intp a timestamp in seconds.
 */
export declare function spanTimeInputToSeconds(input: SpanTimeInput | undefined): number;
/**
 * Convert a span to a JSON representation.
 * Note that all fields returned here are optional and need to be guarded against.
 *
 * Note: Because of this, we currently have a circular type dependency (which we opted out of in package.json).
 * This is not avoidable as we need `spanToJSON` in `spanUtils.ts`, which in turn is needed by `span.ts` for backwards compatibility.
 * And `spanToJSON` needs the Span class from `span.ts` to check here.
 * TODO v8: When we remove the deprecated stuff from `span.ts`, we can remove the circular dependency again.
 */
export declare function spanToJSON(span: Span): Partial<SpanJSON>;
/**
 * Returns true if a span is sampled.
 * In most cases, you should just use `span.isRecording()` instead.
 * However, this has a slightly different semantic, as it also returns false if the span is finished.
 * So in the case where this distinction is important, use this method.
 */
export declare function spanIsSampled(span: Span): boolean;
//# sourceMappingURL=spanUtils.d.ts.map