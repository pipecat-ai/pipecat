import { Client, DynamicSamplingContext, Scope, Span } from '@sentry/types';
/**
 * Creates a dynamic sampling context from a client.
 *
 * Dispatches the `createDsc` lifecycle hook as a side effect.
 */
export declare function getDynamicSamplingContextFromClient(trace_id: string, client: Client, scope?: Scope): DynamicSamplingContext;
/**
 * Creates a dynamic sampling context from a span (and client and scope)
 *
 * @param span the span from which a few values like the root span name and sample rate are extracted.
 *
 * @returns a dynamic sampling context
 */
export declare function getDynamicSamplingContextFromSpan(span: Span): Readonly<Partial<DynamicSamplingContext>>;
//# sourceMappingURL=dynamicSamplingContext.d.ts.map
