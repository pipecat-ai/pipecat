import type { Client, StartSpanOptions } from '@sentry/types';
import type { Span } from '@sentry/types';
import type { RequestInstrumentationOptions } from './request';
export declare const BROWSER_TRACING_INTEGRATION_ID = "BrowserTracing";
/** Options for Browser Tracing integration */
export interface BrowserTracingOptions extends RequestInstrumentationOptions {
    /**
     * The time to wait in ms until the transaction will be finished during an idle state. An idle state is defined
     * by a moment where there are no in-progress spans.
     *
     * The transaction will use the end timestamp of the last finished span as the endtime for the transaction.
     * If there are still active spans when this the `idleTimeout` is set, the `idleTimeout` will get reset.
     * Time is in ms.
     *
     * Default: 1000
     */
    idleTimeout: number;
    /**
     * The max duration for a transaction. If a transaction duration hits the `finalTimeout` value, it
     * will be finished.
     * Time is in ms.
     *
     * Default: 30000
     */
    finalTimeout: number;
    /**
     * The heartbeat interval. If no new spans are started or open spans are finished within 3 heartbeats,
     * the transaction will be finished.
     * Time is in ms.
     *
     * Default: 5000
     */
    heartbeatInterval: number;
    /**
     * If a span should be created on page load.
     * If this is set to `false`, this integration will not start the default page load span.
     * Default: true
     */
    instrumentPageLoad: boolean;
    /**
     * If a span should be created on navigation (history change).
     * If this is set to `false`, this integration will not start the default navigation spans.
     * Default: true
     */
    instrumentNavigation: boolean;
    /**
     * Flag spans where tabs moved to background with "cancelled". Browser background tab timing is
     * not suited towards doing precise measurements of operations. By default, we recommend that this option
     * be enabled as background transactions can mess up your statistics in nondeterministic ways.
     *
     * Default: true
     */
    markBackgroundSpan: boolean;
    /**
     * If true, Sentry will capture long tasks and add them to the corresponding transaction.
     *
     * Default: true
     */
    enableLongTask: boolean;
    /**
     * If true, Sentry will capture INP web vitals as standalone spans .
     *
     * Default: false
     */
    enableInp: boolean;
    /**
     * Sample rate to determine interaction span sampling.
     * interactionsSampleRate is applied on top of the global tracesSampleRate.
     * ie a tracesSampleRate of 0.1 and interactionsSampleRate of 0.5 will result in a 0.05 sample rate for interactions.
     *
     * Default: 1
     */
    interactionsSampleRate: number;
    /**
     * _metricOptions allows the user to send options to change how metrics are collected.
     *
     * _metricOptions is currently experimental.
     *
     * Default: undefined
     */
    _metricOptions?: Partial<{
        /**
         * @deprecated This property no longer has any effect and will be removed in v8.
         */
        _reportAllChanges: boolean;
    }>;
    /**
     * _experiments allows the user to send options to define how this integration works.
     * Note that the `enableLongTask` options is deprecated in favor of the option at the top level, and will be removed in v8.
     *
     * TODO (v8): Remove enableLongTask
     *
     * Default: undefined
     */
    _experiments: Partial<{
        enableInteractions: boolean;
    }>;
    /**
     * A callback which is called before a span for a pageload or navigation is started.
     * It receives the options passed to `startSpan`, and expects to return an updated options object.
     */
    beforeStartSpan?: (options: StartSpanOptions) => StartSpanOptions;
}
/**
 * The Browser Tracing integration automatically instruments browser pageload/navigation
 * actions as transactions, and captures requests, metrics and errors as spans.
 *
 * The integration can be configured with a variety of options, and can be extended to use
 * any routing library. This integration uses {@see IdleTransaction} to create transactions.
 *
 * We explicitly export the proper type here, as this has to be extended in some cases.
 */
export declare const browserTracingIntegration: (_options?: Partial<BrowserTracingOptions>) => {
    name: string;
    setupOnce: () => void;
    afterAllSetup(client: Client<import("@sentry/types").ClientOptions<import("@sentry/types").BaseTransportOptions>>): void;
    options: {
        idleTimeout: number;
        finalTimeout: number;
        heartbeatInterval: number;
        instrumentPageLoad: boolean;
        instrumentNavigation: boolean;
        markBackgroundSpan: boolean;
        enableLongTask: boolean;
        enableInp: boolean;
        interactionsSampleRate: number;
        _metricOptions?: Partial<{
            /**
             * @deprecated This property no longer has any effect and will be removed in v8.
             */
            _reportAllChanges: boolean;
        }> | undefined;
        _experiments: Partial<{
            enableInteractions: boolean;
        }>;
        beforeStartSpan?: ((options: StartSpanOptions) => StartSpanOptions) | undefined;
        tracingOrigins: (string | RegExp)[];
        tracePropagationTargets: (string | RegExp)[];
        traceFetch: boolean;
        traceXHR: boolean;
        enableHTTPTimings: boolean;
        shouldCreateSpanForRequest?: ((this: void, url: string) => boolean) | undefined;
    };
};
/**
 * Manually start a page load span.
 * This will only do something if the BrowserTracing integration has been setup.
 */
export declare function startBrowserTracingPageLoadSpan(client: Client, spanOptions: StartSpanOptions): Span | undefined;
/**
 * Manually start a navigation span.
 * This will only do something if the BrowserTracing integration has been setup.
 */
export declare function startBrowserTracingNavigationSpan(client: Client, spanOptions: StartSpanOptions): Span | undefined;
/** Returns the value of a meta tag */
export declare function getMetaContent(metaName: string): string | undefined;
//# sourceMappingURL=browserTracingIntegration.d.ts.map