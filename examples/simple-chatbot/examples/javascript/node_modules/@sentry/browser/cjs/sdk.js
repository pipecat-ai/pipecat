Object.defineProperty(exports, '__esModule', { value: true });

const core = require('@sentry/core');
const utils = require('@sentry/utils');
const client = require('./client.js');
const debugBuild = require('./debug-build.js');
const helpers = require('./helpers.js');
const breadcrumbs = require('./integrations/breadcrumbs.js');
const dedupe = require('./integrations/dedupe.js');
const globalhandlers = require('./integrations/globalhandlers.js');
const httpcontext = require('./integrations/httpcontext.js');
const linkederrors = require('./integrations/linkederrors.js');
const trycatch = require('./integrations/trycatch.js');
const stackParsers = require('./stack-parsers.js');
const fetch = require('./transports/fetch.js');
const xhr = require('./transports/xhr.js');

/** @deprecated Use `getDefaultIntegrations(options)` instead. */
const defaultIntegrations = [
  core.inboundFiltersIntegration(),
  core.functionToStringIntegration(),
  trycatch.browserApiErrorsIntegration(),
  breadcrumbs.breadcrumbsIntegration(),
  globalhandlers.globalHandlersIntegration(),
  linkederrors.linkedErrorsIntegration(),
  dedupe.dedupeIntegration(),
  httpcontext.httpContextIntegration(),
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
    if (helpers.WINDOW.SENTRY_RELEASE && helpers.WINDOW.SENTRY_RELEASE.id) {
      options.release = helpers.WINDOW.SENTRY_RELEASE.id;
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
    stackParser: utils.stackParserFromStackParserOptions(options.stackParser || stackParsers.defaultStackParser),
    integrations: core.getIntegrationsToSetup(options),
    transport: options.transport || (utils.supportsFetch() ? fetch.makeFetchTransport : xhr.makeXHRTransport),
  };

  core.initAndBind(client.BrowserClient, clientOptions);

  if (options.autoSessionTracking) {
    startSessionTracking();
  }
}

const showReportDialog = (
  // eslint-disable-next-line deprecation/deprecation
  options = {},
  // eslint-disable-next-line deprecation/deprecation
  hub = core.getCurrentHub(),
) => {
  // doesn't work without a document (React Native)
  if (!helpers.WINDOW.document) {
    debugBuild.DEBUG_BUILD && utils.logger.error('Global document not defined in showReportDialog call');
    return;
  }

  // eslint-disable-next-line deprecation/deprecation
  const { client, scope } = hub.getStackTop();
  const dsn = options.dsn || (client && client.getDsn());
  if (!dsn) {
    debugBuild.DEBUG_BUILD && utils.logger.error('DSN not configured for showReportDialog call');
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

  const script = helpers.WINDOW.document.createElement('script');
  script.async = true;
  script.crossOrigin = 'anonymous';
  script.src = core.getReportDialogEndpoint(dsn, options);

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
          helpers.WINDOW.removeEventListener('message', reportDialogClosedMessageHandler);
        }
      }
    };
    helpers.WINDOW.addEventListener('message', reportDialogClosedMessageHandler);
  }

  const injectionPoint = helpers.WINDOW.document.head || helpers.WINDOW.document.body;
  if (injectionPoint) {
    injectionPoint.appendChild(script);
  } else {
    debugBuild.DEBUG_BUILD && utils.logger.error('Not injecting report dialog. No injection point found in HTML');
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
  return helpers.wrap(fn)();
}

/**
 * Enable automatic Session Tracking for the initial page load.
 */
function startSessionTracking() {
  if (typeof helpers.WINDOW.document === 'undefined') {
    debugBuild.DEBUG_BUILD && utils.logger.warn('Session tracking in non-browser environment with @sentry/browser is not supported.');
    return;
  }

  // The session duration for browser sessions does not track a meaningful
  // concept that can be used as a metric.
  // Automatically captured sessions are akin to page views, and thus we
  // discard their duration.
  core.startSession({ ignoreDuration: true });
  core.captureSession();

  // We want to create a session for every navigation as well
  utils.addHistoryInstrumentationHandler(({ from, to }) => {
    // Don't create an additional session for the initial route or if the location did not change
    if (from !== undefined && from !== to) {
      core.startSession({ ignoreDuration: true });
      core.captureSession();
    }
  });
}

/**
 * Captures user feedback and sends it to Sentry.
 */
function captureUserFeedback(feedback) {
  const client = core.getClient();
  if (client) {
    client.captureUserFeedback(feedback);
  }
}

exports.captureUserFeedback = captureUserFeedback;
exports.defaultIntegrations = defaultIntegrations;
exports.forceLoad = forceLoad;
exports.getDefaultIntegrations = getDefaultIntegrations;
exports.init = init;
exports.onLoad = onLoad;
exports.showReportDialog = showReportDialog;
exports.wrap = wrap;
//# sourceMappingURL=sdk.js.map
