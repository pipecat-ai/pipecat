import type { HandlerDataXhr, Span } from '@sentry/types';
export declare const DEFAULT_TRACE_PROPAGATION_TARGETS: (string | RegExp)[];
/** Options for Request Instrumentation */
export interface RequestInstrumentationOptions {
    /**
     * @deprecated Will be removed in v8.
     * Use `shouldCreateSpanForRequest` to control span creation and `tracePropagationTargets` to control
     * trace header attachment.
     */
    tracingOrigins: Array<string | RegExp>;
    /**
     * List of strings and/or regexes used to determine which outgoing requests will have `sentry-trace` and `baggage`
     * headers attached.
     *
     * @deprecated Use the top-level `tracePropagationTargets` option in `Sentry.init` instead.
     * This option will be removed in v8.
     *
     * Default: ['localhost', /^\//] @see {DEFAULT_TRACE_PROPAGATION_TARGETS}
     */
    tracePropagationTargets: Array<string | RegExp>;
    /**
     * Flag to disable patching all together for fetch requests.
     *
     * Default: true
     */
    traceFetch: boolean;
    /**
     * Flag to disable patching all together for xhr requests.
     *
     * Default: true
     */
    traceXHR: boolean;
    /**
     * If true, Sentry will capture http timings and add them to the corresponding http spans.
     *
     * Default: true
     */
    enableHTTPTimings: boolean;
    /**
     * This function will be called before creating a span for a request with the given url.
     * Return false if you don't want a span for the given url.
     *
     * Default: (url: string) => true
     */
    shouldCreateSpanForRequest?(this: void, url: string): boolean;
}
export declare const defaultRequestInstrumentationOptions: RequestInstrumentationOptions;
/** Registers span creators for xhr and fetch requests  */
export declare function instrumentOutgoingRequests(_options?: Partial<RequestInstrumentationOptions>): void;
/**
 * Converts ALPN protocol ids to name and version.
 *
 * (https://www.iana.org/assignments/tls-extensiontype-values/tls-extensiontype-values.xhtml#alpn-protocol-ids)
 * @param nextHopProtocol PerformanceResourceTiming.nextHopProtocol
 */
export declare function extractNetworkProtocol(nextHopProtocol: string): {
    name: string;
    version: string;
};
/**
 * A function that determines whether to attach tracing headers to a request.
 * This was extracted from `instrumentOutgoingRequests` to make it easier to test shouldAttachHeaders.
 * We only export this fuction for testing purposes.
 */
export declare function shouldAttachHeaders(url: string, tracePropagationTargets: (string | RegExp)[] | undefined): boolean;
/**
 * Create and track xhr request spans
 *
 * @returns Span if a span was created, otherwise void.
 */
export declare function xhrCallback(handlerData: HandlerDataXhr, shouldCreateSpan: (url: string) => boolean, shouldAttachHeaders: (url: string) => boolean, spans: Record<string, Span>): Span | undefined;
//# sourceMappingURL=request.d.ts.map