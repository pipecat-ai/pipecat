import { inboundFiltersIntegration, functionToStringIntegration, getIntegrationsToSetup, initAndBind, getReportDialogEndpoint, getCurrentHub, startSession, captureSession, getClient } from '@sentry/core';
import { stackParserFromStackParserOptions, supportsFetch, logger, addHistoryInstrumentationHandler } from '@sentry/utils';
import { BrowserClient } from './client.js';
import { DEBUG_BUILD } from './debug-build.js';
import { WINDOW, wrap as wrap$1 } from './helpers.js';
import { breadcrumbsIntegration } from './integrations/breadcrumbs.js';
import { dedupeIntegration } from './integrations/dedupe.js';
import { globalHandlersIntegration } from './integrations/globalhandlers.js';
import { httpContextIntegration } from './integrations/httpcontext.js';
import { linkedErrorsIntegration } from './integrations/linkederrors.js';
import { browserApiErrorsIntegration } from './integrations/trycatch.js';
import { defaultStackParser } from './stack-parsers.js';
import { makeFetchTransport } from './transports/fetch.js';
import { makeXHRTransport } from './transports/xhr.js';

/** @deprecated Use `getDefaultIntegrations(options)` instead. */
const defaultIntegrations = [
  inboundFiltersIntegration(),
  functionToStringIntegration(),
  browserApiErrorsIntegration(),
  breadcrumbsIntegration(),
  globalHandlersIntegration(),
  linkedErrorsIntegration(),
  dedupeIntegration(),
  httpContextIntegration(),
];

/** Get the default integrations for the browser SDK. */
function getDefaultIntegrations(_options) {
  // We return a copy of the defaultIntegrations here to avoid mutating this
  return [
    // eslint-disable-next-line deprecation/deprecation
    ...defaultIntegrations,
  ];
}

/**
 * A magic string that build tooling can leverage in order to inject a release value into the SDK.
 */

/**
 * The Sentry Browser SDK Client.
 *
 * To use this SDK, call the {@link init} function as early as possible when
 * loading the web page. To set context information or send manual events, use
 * the provided methods.
 *
 * @example
 *
 * ```
 *
 * import { init } from '@sentry/browser';
 *
 * init({
 *   dsn: '__DSN__',
 *   // ...
 * });
 * ```
 *
 * @example
 * ```
 *
 * import { configureScope } from '@sentry/browser';
 * configureScope((scope: Scope) => {
 *   scope.setExtra({ battery: 0.7 });
 *   scope.setTag({ user_mode: 'admin' });
 *   scope.setUser({ id: '4711' });
 * });
 * ```
 *
 * @example
 * ```
 *
 * import { addBreadcrumb } from '@sentry/browser';
 * addBreadcrumb({
 *   message: 'My Breadcrumb',
 *   // ...
 * });
 * ```
 *
 * @example
 *
 * ```
 *
 * import * as Sentry from '@sentry/browser';
 * Sentry.captureMessage('Hello, world!');
 * Sentry.captureException(new Error('Good bye'));
 * Sentry.captureEvent({
 *   message: 'Manual',
 *   stacktrace: [
 *     // ...
 *   ],
 * });
 * ```
 *
 * @see {@link BrowserOptions} for documentation on configuration options.
 */
function init(options = {}) {
  if (options.defaultIntegrations === undefined) {
    options.defaultIntegrations = getDefaultIntegrations();
  }
  if (options.release === undefined) {
    // This allows build tooling to find-and-replace __SENTRY_RELEASE__ to inject a release value
    if (typeof __SENTRY_RELEASE__ === 'string') {
      options.release = __SENTRY_RELEASE__;
    }

    // This supports the variable that sentry-webpack-plugin injects
    if (WINDOW.SENTRY_RELEASE && WINDOW.SENTRY_RELEASE.id) {
      options.release = WINDOW.SENTRY_RELEASE.id;
    }
  }
  if (options.autoSessionTracking === undefined) {
    options.autoSessionTracking = true;
  }
  if (options.sendClientReports === undefined) {
    options.sendClientReports = true;
  }

  const clientOptions = {
    ...options,
    stackParser: stackParserFromStackParserOptions(options.stackParser || defaultStackParser),
    integrations: getIntegrationsToSetup(options),
    transport: options.transport || (supportsFetch() ? makeFetchTransport : makeXHRTransport),
  };

  initAndBind(BrowserClient, clientOptions);

  if (options.autoSessionTracking) {
    startSessionTracking();
  }
}

const showReportDialog = (
  // eslint-disable-next-line deprecation/deprecation
  options = {},
  // eslint-disable-next-line deprecation/deprecation
  hub = getCurrentHub(),
) => {
  // doesn't work without a document (React Native)
  if (!WINDOW.document) {
    DEBUG_BUILD && logger.error('Global document not defined in showReportDialog call');
    return;
  }

  // eslint-disable-next-line deprecation/deprecation
  const { client, scope } = hub.getStackTop();
  const dsn = options.dsn || (client && client.getDsn());
  if (!dsn) {
    DEBUG_BUILD && logger.error('DSN not configured for showReportDialog call');
    return;
  }

  if (scope) {
    options.user = {
      ...scope.getUser(),
      ...options.user,
    };
  }

  if (!options.eventId) {
    // eslint-disable-next-line deprecation/deprecation
    options.eventId = hub.lastEventId();
  }

  const script = WINDOW.document.createElement('script');
  script.async = true;
  script.crossOrigin = 'anonymous';
  script.src = getReportDialogEndpoint(dsn, options);

  if (options.onLoad) {
    script.onload = options.onLoad;
  }

  const { onClose } = options;
  if (onClose) {
    const reportDialogClosedMessageHandler = (event) => {
      if (event.data === '__sentry_reportdialog_closed__') {
        try {
          onClose();
        } finally {
          WINDOW.removeEventListener('message', reportDialogClosedMessageHandler);
        }
      }
    };
    WINDOW.addEventListener('message', reportDialogClosedMessageHandler);
  }

  const injectionPoint = WINDOW.document.head || WINDOW.document.body;
  if (injectionPoint) {
    injectionPoint.appendChild(script);
  } else {
    DEBUG_BUILD && logger.error('Not injecting report dialog. No injection point found in HTML');
  }
};

/**
 * This function is here to be API compatible with the loader.
 * @hidden
 */
function forceLoad() {
  // Noop
}

/**
 * This function is here to be API compatible with the loader.
 * @hidden
 */
function onLoad(callback) {
  callback();
}

/**
 * Wrap code within a try/catch block so the SDK is able to capture errors.
 *
 * @deprecated This function will be removed in v8.
 * It is not part of Sentry's official API and it's easily replaceable by using a try/catch block
 * and calling Sentry.captureException.
 *
 * @param fn A function to wrap.
 *
 * @returns The result of wrapped function call.
 */
// TODO(v8): Remove this function
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function wrap(fn) {
  return wrap$1(fn)();
}

/**
 * Enable automatic Session Tracking for the initial page load.
 */
function startSessionTracking() {
  if (typeof WINDOW.document === 'undefined') {
    DEBUG_BUILD && logger.warn('Session tracking in non-browser environment with @sentry/browser is not supported.');
    return;
  }

  // The session duration for browser sessions does not track a meaningful
  // concept that can be used as a metric.
  // Automatically captured sessions are akin to page views, and thus we
  // discard their duration.
  startSession({ ignoreDuration: true });
  captureSession();

  // We want to create a session for every navigation as well
  addHistoryInstrumentationHandler(({ from, to }) => {
    // Don't create an additional session for the initial route or if the location did not change
    if (from !== undefined && from !== to) {
      startSession({ ignoreDuration: true });
      captureSession();
    }
  });
}

/**
 * Captures user feedback and sends it to Sentry.
 */
function captureUserFeedback(feedback) {
  const client = getClient();
  if (client) {
    client.captureUserFeedback(feedback);
  }
}

export { captureUserFeedback, defaultIntegrations, forceLoad, getDefaultIntegrations, init, onLoad, showReportDialog, wrap };
//# sourceMappingURL=sdk.js.map
