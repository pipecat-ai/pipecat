export * from './exports';
/** @deprecated Import the integration function directly, e.g. `inboundFiltersIntegration()` instead of `new Integrations.InboundFilter(). */
declare const INTEGRATIONS: {
    GlobalHandlers: import("@sentry/types").IntegrationClass<import("@sentry/types").Integration & {
        setup: (client: import("@sentry/types").Client<import("@sentry/types").ClientOptions<import("@sentry/types").BaseTransportOptions>>) => void;
    }> & (new (options?: Partial<{
        onerror: boolean;
        onunhandledrejection: boolean;
    }> | undefined) => import("@sentry/types").Integration);
    TryCatch: import("@sentry/types").IntegrationClass<import("@sentry/types").Integration> & (new (options?: {
        setTimeout: boolean;
        setInterval: boolean;
        requestAnimationFrame: boolean;
        XMLHttpRequest: boolean;
        eventTarget: boolean | string[];
    } | undefined) => import("@sentry/types").Integration);
    Breadcrumbs: import("@sentry/types").IntegrationClass<import("@sentry/types").Integration & {
        setup: (client: import("@sentry/types").Client<import("@sentry/types").ClientOptions<import("@sentry/types").BaseTransportOptions>>) => void;
    }> & (new (options?: Partial<{
        console: boolean;
        dom: boolean | {
            serializeAttribute?: string | string[] | undefined;
            maxStringLength?: number | undefined;
        };
        fetch: boolean;
        history: boolean;
        sentry: boolean;
        xhr: boolean;
    }> | undefined) => import("@sentry/types").Integration);
    LinkedErrors: import("@sentry/types").IntegrationClass<import("@sentry/types").Integration & {
        preprocessEvent: (event: import("@sentry/types").Event, hint: import("@sentry/types").EventHint, client: import("@sentry/types").Client<import("@sentry/types").ClientOptions<import("@sentry/types").BaseTransportOptions>>) => void;
    }> & (new (options?: {
        key?: string | undefined;
        limit?: number | undefined;
    } | undefined) => import("@sentry/types").Integration);
    HttpContext: import("@sentry/types").IntegrationClass<import("@sentry/types").Integration & {
        preprocessEvent: (event: import("@sentry/types").Event) => void;
    }>;
    Dedupe: import("@sentry/types").IntegrationClass<import("@sentry/types").Integration & {
        processEvent: (event: import("@sentry/types").Event) => import("@sentry/types").Event;
    }>;
    FunctionToString: import("@sentry/types").IntegrationClass<import("@sentry/types").Integration & {
        setupOnce: () => void;
    }>;
    InboundFilters: import("@sentry/types").IntegrationClass<import("@sentry/types").Integration & {
        preprocessEvent: (event: import("@sentry/types").Event, hint: import("@sentry/types").EventHint, client: import("@sentry/types").Client<import("@sentry/types").ClientOptions<import("@sentry/types").BaseTransportOptions>>) => void;
    }> & (new (options?: Partial<{
        allowUrls: (string | RegExp)[];
        denyUrls: (string | RegExp)[];
        ignoreErrors: (string | RegExp)[];
        ignoreTransactions: (string | RegExp)[];
        ignoreInternal: boolean;
        disableErrorDefaults: boolean;
        disableTransactionDefaults: boolean;
    }> | undefined) => import("@sentry/types").Integration);
};
export { INTEGRATIONS as Integrations };
export { InternalReplay as Replay, internalReplayIntegration as replayIntegration, internalGetReplay as getReplay, } from '@sentry/replay';
export type { InternalReplayEventType as ReplayEventType, InternalReplayEventWithTime as ReplayEventWithTime, InternalReplayBreadcrumbFrame as ReplayBreadcrumbFrame, InternalReplayBreadcrumbFrameEvent as ReplayBreadcrumbFrameEvent, InternalReplayOptionFrameEvent as ReplayOptionFrameEvent, InternalReplayFrame as ReplayFrame, InternalReplayFrameEvent as ReplayFrameEvent, InternalReplaySpanFrame as ReplaySpanFrame, InternalReplaySpanFrameEvent as ReplaySpanFrameEvent, } from '@sentry/replay';
export { ReplayCanvas, replayCanvasIntegration, } from '@sentry-internal/replay-canvas';
export { Feedback, feedbackIntegration, sendFeedback, } from '@sentry-internal/feedback';
export { captureConsoleIntegration, dedupeIntegration, debugIntegration, extraErrorDataIntegration, reportingObserverIntegration, rewriteFramesIntegration, sessionTimingIntegration, httpClientIntegration, contextLinesIntegration, } from '@sentry/integrations';
export { BrowserTracing, defaultRequestInstrumentationOptions, instrumentOutgoingRequests, browserTracingIntegration, startBrowserTracingNavigationSpan, startBrowserTracingPageLoadSpan, } from '@sentry-internal/tracing';
export type { RequestInstrumentationOptions } from '@sentry-internal/tracing';
export { addTracingExtensions, setMeasurement, extractTraceparentData, getActiveTransaction, spanStatusfromHttpCode, getSpanStatusFromHttpCode, setHttpStatus, trace, makeMultiplexedTransport, ModuleMetadata, moduleMetadataIntegration, } from '@sentry/core';
export type { SpanStatusType } from '@sentry/core';
export type { Span } from '@sentry/types';
export { makeBrowserOfflineTransport } from './transports/offline';
export { onProfilingStartRouteTransaction } from './profiling/hubextensions';
export { BrowserProfilingIntegration, browserProfilingIntegration, } from './profiling/integration';
//# sourceMappingURL=index.d.ts.map