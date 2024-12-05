import { Integrations } from '@sentry/core';
export { FunctionToString, Hub, InboundFilters, ModuleMetadata, SDK_VERSION, SEMANTIC_ATTRIBUTE_SENTRY_OP, SEMANTIC_ATTRIBUTE_SENTRY_ORIGIN, SEMANTIC_ATTRIBUTE_SENTRY_SAMPLE_RATE, SEMANTIC_ATTRIBUTE_SENTRY_SOURCE, Scope, addBreadcrumb, addEventProcessor, addGlobalEventProcessor, addIntegration, addTracingExtensions, captureEvent, captureException, captureMessage, captureSession, close, configureScope, continueTrace, createTransport, endSession, extractTraceparentData, flush, functionToStringIntegration, getActiveSpan, getActiveTransaction, getClient, getCurrentHub, getCurrentScope, getHubFromCarrier, getSpanStatusFromHttpCode, inboundFiltersIntegration, isInitialized, lastEventId, makeMain, makeMultiplexedTransport, metrics, moduleMetadataIntegration, parameterize, setContext, setCurrentClient, setExtra, setExtras, setHttpStatus, setMeasurement, setTag, setTags, setUser, spanStatusfromHttpCode, startInactiveSpan, startSession, startSpan, startSpanManual, startTransaction, trace, withActiveSpan, withIsolationScope, withScope } from '@sentry/core';
import { WINDOW } from './helpers.js';
export { WINDOW } from './helpers.js';
export { BrowserClient } from './client.js';
export { makeFetchTransport } from './transports/fetch.js';
export { makeXHRTransport } from './transports/xhr.js';
export { chromeStackLineParser, defaultStackLineParsers, defaultStackParser, geckoStackLineParser, opera10StackLineParser, opera11StackLineParser, winjsStackLineParser } from './stack-parsers.js';
export { eventFromException, eventFromMessage, exceptionFromError } from './eventbuilder.js';
export { createUserFeedbackEnvelope } from './userfeedback.js';
export { captureUserFeedback, defaultIntegrations, forceLoad, getDefaultIntegrations, init, onLoad, showReportDialog, wrap } from './sdk.js';
export { Breadcrumbs, breadcrumbsIntegration } from './integrations/breadcrumbs.js';
export { Dedupe } from './integrations/dedupe.js';
export { GlobalHandlers, globalHandlersIntegration } from './integrations/globalhandlers.js';
export { HttpContext, httpContextIntegration } from './integrations/httpcontext.js';
export { LinkedErrors, linkedErrorsIntegration } from './integrations/linkederrors.js';
export { TryCatch, browserApiErrorsIntegration } from './integrations/trycatch.js';
import * as index from './integrations/index.js';
export { InternalReplay as Replay, internalGetReplay as getReplay, internalReplayIntegration as replayIntegration } from '@sentry/replay';
export { ReplayCanvas, replayCanvasIntegration } from '@sentry-internal/replay-canvas';
export { Feedback, feedbackIntegration, sendFeedback } from '@sentry-internal/feedback';
export { captureConsoleIntegration, contextLinesIntegration, debugIntegration, dedupeIntegration, extraErrorDataIntegration, httpClientIntegration, reportingObserverIntegration, rewriteFramesIntegration, sessionTimingIntegration } from '@sentry/integrations';
export { BrowserTracing, browserTracingIntegration, defaultRequestInstrumentationOptions, instrumentOutgoingRequests, startBrowserTracingNavigationSpan, startBrowserTracingPageLoadSpan } from '@sentry-internal/tracing';
export { makeBrowserOfflineTransport } from './transports/offline.js';
export { onProfilingStartRouteTransaction } from './profiling/hubextensions.js';
export { BrowserProfilingIntegration, browserProfilingIntegration } from './profiling/integration.js';

let windowIntegrations = {};

// This block is needed to add compatibility with the integrations packages when used with a CDN
if (WINDOW.Sentry && WINDOW.Sentry.Integrations) {
  windowIntegrations = WINDOW.Sentry.Integrations;
}

/** @deprecated Import the integration function directly, e.g. `inboundFiltersIntegration()` instead of `new Integrations.InboundFilter(). */
const INTEGRATIONS = {
  ...windowIntegrations,
  // eslint-disable-next-line deprecation/deprecation
  ...Integrations,
  ...index,
};

export { INTEGRATIONS as Integrations };
//# sourceMappingURL=index.js.map
