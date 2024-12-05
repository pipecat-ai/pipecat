import { PropagationContext, TraceparentData } from '@sentry/types';
import { baggageHeaderToDynamicSamplingContext } from './baggage';
export declare const TRACEPARENT_REGEXP: RegExp;
/**
 * Extract transaction context data from a `sentry-trace` header.
 *
 * @param traceparent Traceparent string
 *
 * @returns Object containing data from the header, or undefined if traceparent string is malformed
 */
export declare function extractTraceparentData(traceparent?: string): TraceparentData | undefined;
/**
 * Create tracing context from incoming headers.
 *
 * @deprecated Use `propagationContextFromHeaders` instead.
 */
export declare function tracingContextFromHeaders(sentryTrace: Parameters<typeof extractTraceparentData>[0], baggage: Parameters<typeof baggageHeaderToDynamicSamplingContext>[0]): {
    traceparentData: ReturnType<typeof extractTraceparentData>;
    dynamicSamplingContext: ReturnType<typeof baggageHeaderToDynamicSamplingContext>;
    propagationContext: PropagationContext;
};
/**
 * Create a propagation context from incoming headers.
 */
export declare function propagationContextFromHeaders(sentryTrace: string | undefined, baggage: string | number | boolean | string[] | null | undefined): PropagationContext;
/**
 * Create sentry-trace header from span context values.
 */
export declare function generateSentryTraceHeader(traceId?: string, spanId?: string, sampled?: boolean): string;
//# sourceMappingURL=tracing.d.ts.map
