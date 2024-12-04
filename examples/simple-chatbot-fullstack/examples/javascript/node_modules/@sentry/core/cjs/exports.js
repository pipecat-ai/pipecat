Object.defineProperty(exports, '__esModule', { value: true });

const utils = require('@sentry/utils');
const constants = require('./constants.js');
const debugBuild = require('./debug-build.js');
const hub = require('./hub.js');
const session = require('./session.js');
const prepareEvent = require('./utils/prepareEvent.js');

/**
 * Captures an exception event and sends it to Sentry.
 *
 * @param exception The exception to capture.
 * @param hint Optional additional data to attach to the Sentry event.
 * @returns the id of the captured Sentry event.
 */
function captureException(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  exception,
  hint,
) {
  // eslint-disable-next-line deprecation/deprecation
  return hub.getCurrentHub().captureException(exception, prepareEvent.parseEventHintOrCaptureContext(hint));
}

/**
 * Captures a message event and sends it to Sentry.
 *
 * @param exception The exception to capture.
 * @param captureContext Define the level of the message or pass in additional data to attach to the message.
 * @returns the id of the captured message.
 */
function captureMessage(
  message,
  // eslint-disable-next-line deprecation/deprecation
  captureContext,
) {
  // This is necessary to provide explicit scopes upgrade, without changing the original
  // arity of the `captureMessage(message, level)` method.
  const level = typeof captureContext === 'string' ? captureContext : undefined;
  const context = typeof captureContext !== 'string' ? { captureContext } : undefined;
  // eslint-disable-next-line deprecation/deprecation
  return hub.getCurrentHub().captureMessage(message, level, context);
}

/**
 * Captures a manually created event and sends it to Sentry.
 *
 * @param exception The event to send to Sentry.
 * @param hint Optional additional data to attach to the Sentry event.
 * @returns the id of the captured event.
 */
function captureEvent(event, hint) {
  // eslint-disable-next-line deprecation/deprecation
  return hub.getCurrentHub().captureEvent(event, hint);
}

/**
 * Callback to set context information onto the scope.
 * @param callback Callback function that receives Scope.
 *
 * @deprecated Use getCurrentScope() directly.
 */
// eslint-disable-next-line deprecation/deprecation
function configureScope(callback) {
  // eslint-disable-next-line deprecation/deprecation
  hub.getCurrentHub().configureScope(callback);
}

/**
 * Records a new breadcrumb which will be attached to future events.
 *
 * Breadcrumbs will be added to subsequent events to provide more context on
 * user's actions prior to an error or crash.
 *
 * @param breadcrumb The breadcrumb to record.
 */
// eslint-disable-next-line deprecation/deprecation
function addBreadcrumb(breadcrumb, hint) {
  // eslint-disable-next-line deprecation/deprecation
  hub.getCurrentHub().addBreadcrumb(breadcrumb, hint);
}

/**
 * Sets context data with the given name.
 * @param name of the context
 * @param context Any kind of data. This data will be normalized.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any, deprecation/deprecation
function setContext(name, context) {
  // eslint-disable-next-line deprecation/deprecation
  hub.getCurrentHub().setContext(name, context);
}

/**
 * Set an object that will be merged sent as extra data with the event.
 * @param extras Extras object to merge into current context.
 */
// eslint-disable-next-line deprecation/deprecation
function setExtras(extras) {
  // eslint-disable-next-line deprecation/deprecation
  hub.getCurrentHub().setExtras(extras);
}

/**
 * Set key:value that will be sent as extra data with the event.
 * @param key String of extra
 * @param extra Any kind of data. This data will be normalized.
 */
// eslint-disable-next-line deprecation/deprecation
function setExtra(key, extra) {
  // eslint-disable-next-line deprecation/deprecation
  hub.getCurrentHub().setExtra(key, extra);
}

/**
 * Set an object that will be merged sent as tags data with the event.
 * @param tags Tags context object to merge into current context.
 */
// eslint-disable-next-line deprecation/deprecation
function setTags(tags) {
  // eslint-disable-next-line deprecation/deprecation
  hub.getCurrentHub().setTags(tags);
}

/**
 * Set key:value that will be sent as tags data with the event.
 *
 * Can also be used to unset a tag, by passing `undefined`.
 *
 * @param key String key of tag
 * @param value Value of tag
 */
// eslint-disable-next-line deprecation/deprecation
function setTag(key, value) {
  // eslint-disable-next-line deprecation/deprecation
  hub.getCurrentHub().setTag(key, value);
}

/**
 * Updates user context information for future events.
 *
 * @param user User context object to be set in the current context. Pass `null` to unset the user.
 */
// eslint-disable-next-line deprecation/deprecation
function setUser(user) {
  // eslint-disable-next-line deprecation/deprecation
  hub.getCurrentHub().setUser(user);
}

/**
 * Creates a new scope with and executes the given operation within.
 * The scope is automatically removed once the operation
 * finishes or throws.
 *
 * This is essentially a convenience function for:
 *
 *     pushScope();
 *     callback();
 *     popScope();
 */

/**
 * Either creates a new active scope, or sets the given scope as active scope in the given callback.
 */
function withScope(
  ...rest
) {
  // eslint-disable-next-line deprecation/deprecation
  const hub$1 = hub.getCurrentHub();

  // If a scope is defined, we want to make this the active scope instead of the default one
  if (rest.length === 2) {
    const [scope, callback] = rest;
    if (!scope) {
      // eslint-disable-next-line deprecation/deprecation
      return hub$1.withScope(callback);
    }

    // eslint-disable-next-line deprecation/deprecation
    return hub$1.withScope(() => {
      // eslint-disable-next-line deprecation/deprecation
      hub$1.getStackTop().scope = scope ;
      return callback(scope );
    });
  }

  // eslint-disable-next-line deprecation/deprecation
  return hub$1.withScope(rest[0]);
}

/**
 * Attempts to fork the current isolation scope and the current scope based on the current async context strategy. If no
 * async context strategy is set, the isolation scope and the current scope will not be forked (this is currently the
 * case, for example, in the browser).
 *
 * Usage of this function in environments without async context strategy is discouraged and may lead to unexpected behaviour.
 *
 * This function is intended for Sentry SDK and SDK integration development. It is not recommended to be used in "normal"
 * applications directly because it comes with pitfalls. Use at your own risk!
 *
 * @param callback The callback in which the passed isolation scope is active. (Note: In environments without async
 * context strategy, the currently active isolation scope may change within execution of the callback.)
 * @returns The same value that `callback` returns.
 */
function withIsolationScope(callback) {
  return hub.runWithAsyncContext(() => {
    return callback(hub.getIsolationScope());
  });
}

/**
 * Forks the current scope and sets the provided span as active span in the context of the provided callback.
 *
 * @param span Spans started in the context of the provided callback will be children of this span.
 * @param callback Execution context in which the provided span will be active. Is passed the newly forked scope.
 * @returns the value returned from the provided callback function.
 */
function withActiveSpan(span, callback) {
  return withScope(scope => {
    // eslint-disable-next-line deprecation/deprecation
    scope.setSpan(span);
    return callback(scope);
  });
}

/**
 * Starts a new `Transaction` and returns it. This is the entry point to manual tracing instrumentation.
 *
 * A tree structure can be built by adding child spans to the transaction, and child spans to other spans. To start a
 * new child span within the transaction or any span, call the respective `.startChild()` method.
 *
 * Every child span must be finished before the transaction is finished, otherwise the unfinished spans are discarded.
 *
 * The transaction must be finished with a call to its `.end()` method, at which point the transaction with all its
 * finished child spans will be sent to Sentry.
 *
 * NOTE: This function should only be used for *manual* instrumentation. Auto-instrumentation should call
 * `startTransaction` directly on the hub.
 *
 * @param context Properties of the new `Transaction`.
 * @param customSamplingContext Information given to the transaction sampling function (along with context-dependent
 * default values). See {@link Options.tracesSampler}.
 *
 * @returns The transaction which was just started
 *
 * @deprecated Use `startSpan()`, `startSpanManual()` or `startInactiveSpan()` instead.
 */
function startTransaction(
  context,
  customSamplingContext,
  // eslint-disable-next-line deprecation/deprecation
) {
  // eslint-disable-next-line deprecation/deprecation
  return hub.getCurrentHub().startTransaction({ ...context }, customSamplingContext);
}

/**
 * Create a cron monitor check in and send it to Sentry.
 *
 * @param checkIn An object that describes a check in.
 * @param upsertMonitorConfig An optional object that describes a monitor config. Use this if you want
 * to create a monitor automatically when sending a check in.
 */
function captureCheckIn(checkIn, upsertMonitorConfig) {
  const scope = getCurrentScope();
  const client = getClient();
  if (!client) {
    debugBuild.DEBUG_BUILD && utils.logger.warn('Cannot capture check-in. No client defined.');
  } else if (!client.captureCheckIn) {
    debugBuild.DEBUG_BUILD && utils.logger.warn('Cannot capture check-in. Client does not support sending check-ins.');
  } else {
    return client.captureCheckIn(checkIn, upsertMonitorConfig, scope);
  }

  return utils.uuid4();
}

/**
 * Wraps a callback with a cron monitor check in. The check in will be sent to Sentry when the callback finishes.
 *
 * @param monitorSlug The distinct slug of the monitor.
 * @param upsertMonitorConfig An optional object that describes a monitor config. Use this if you want
 * to create a monitor automatically when sending a check in.
 */
function withMonitor(
  monitorSlug,
  callback,
  upsertMonitorConfig,
) {
  const checkInId = captureCheckIn({ monitorSlug, status: 'in_progress' }, upsertMonitorConfig);
  const now = utils.timestampInSeconds();

  function finishCheckIn(status) {
    captureCheckIn({ monitorSlug, status, checkInId, duration: utils.timestampInSeconds() - now });
  }

  let maybePromiseResult;
  try {
    maybePromiseResult = callback();
  } catch (e) {
    finishCheckIn('error');
    throw e;
  }

  if (utils.isThenable(maybePromiseResult)) {
    Promise.resolve(maybePromiseResult).then(
      () => {
        finishCheckIn('ok');
      },
      () => {
        finishCheckIn('error');
      },
    );
  } else {
    finishCheckIn('ok');
  }

  return maybePromiseResult;
}

/**
 * Call `flush()` on the current client, if there is one. See {@link Client.flush}.
 *
 * @param timeout Maximum time in ms the client should wait to flush its event queue. Omitting this parameter will cause
 * the client to wait until all events are sent before resolving the promise.
 * @returns A promise which resolves to `true` if the queue successfully drains before the timeout, or `false` if it
 * doesn't (or if there's no client defined).
 */
async function flush(timeout) {
  const client = getClient();
  if (client) {
    return client.flush(timeout);
  }
  debugBuild.DEBUG_BUILD && utils.logger.warn('Cannot flush events. No client defined.');
  return Promise.resolve(false);
}

/**
 * Call `close()` on the current client, if there is one. See {@link Client.close}.
 *
 * @param timeout Maximum time in ms the client should wait to flush its event queue before shutting down. Omitting this
 * parameter will cause the client to wait until all events are sent before disabling itself.
 * @returns A promise which resolves to `true` if the queue successfully drains before the timeout, or `false` if it
 * doesn't (or if there's no client defined).
 */
async function close(timeout) {
  const client = getClient();
  if (client) {
    return client.close(timeout);
  }
  debugBuild.DEBUG_BUILD && utils.logger.warn('Cannot flush events and disable SDK. No client defined.');
  return Promise.resolve(false);
}

/**
 * This is the getter for lastEventId.
 *
 * @returns The last event id of a captured event.
 */
function lastEventId() {
  // eslint-disable-next-line deprecation/deprecation
  return hub.getCurrentHub().lastEventId();
}

/**
 * Get the currently active client.
 */
function getClient() {
  // eslint-disable-next-line deprecation/deprecation
  return hub.getCurrentHub().getClient();
}

/**
 * Returns true if Sentry has been properly initialized.
 */
function isInitialized() {
  return !!getClient();
}

/**
 * Get the currently active scope.
 */
function getCurrentScope() {
  // eslint-disable-next-line deprecation/deprecation
  return hub.getCurrentHub().getScope();
}

/**
 * Start a session on the current isolation scope.
 *
 * @param context (optional) additional properties to be applied to the returned session object
 *
 * @returns the new active session
 */
function startSession(context) {
  const client = getClient();
  const isolationScope = hub.getIsolationScope();
  const currentScope = getCurrentScope();

  const { release, environment = constants.DEFAULT_ENVIRONMENT } = (client && client.getOptions()) || {};

  // Will fetch userAgent if called from browser sdk
  const { userAgent } = utils.GLOBAL_OBJ.navigator || {};

  const session$1 = session.makeSession({
    release,
    environment,
    user: currentScope.getUser() || isolationScope.getUser(),
    ...(userAgent && { userAgent }),
    ...context,
  });

  // End existing session if there's one
  const currentSession = isolationScope.getSession();
  if (currentSession && currentSession.status === 'ok') {
    session.updateSession(currentSession, { status: 'exited' });
  }

  endSession();

  // Afterwards we set the new session on the scope
  isolationScope.setSession(session$1);

  // TODO (v8): Remove this and only use the isolation scope(?).
  // For v7 though, we can't "soft-break" people using getCurrentHub().getScope().setSession()
  currentScope.setSession(session$1);

  return session$1;
}

/**
 * End the session on the current isolation scope.
 */
function endSession() {
  const isolationScope = hub.getIsolationScope();
  const currentScope = getCurrentScope();

  const session$1 = currentScope.getSession() || isolationScope.getSession();
  if (session$1) {
    session.closeSession(session$1);
  }
  _sendSessionUpdate();

  // the session is over; take it off of the scope
  isolationScope.setSession();

  // TODO (v8): Remove this and only use the isolation scope(?).
  // For v7 though, we can't "soft-break" people using getCurrentHub().getScope().setSession()
  currentScope.setSession();
}

/**
 * Sends the current Session on the scope
 */
function _sendSessionUpdate() {
  const isolationScope = hub.getIsolationScope();
  const currentScope = getCurrentScope();
  const client = getClient();
  // TODO (v8): Remove currentScope and only use the isolation scope(?).
  // For v7 though, we can't "soft-break" people using getCurrentHub().getScope().setSession()
  const session = currentScope.getSession() || isolationScope.getSession();
  if (session && client && client.captureSession) {
    client.captureSession(session);
  }
}

/**
 * Sends the current session on the scope to Sentry
 *
 * @param end If set the session will be marked as exited and removed from the scope.
 *            Defaults to `false`.
 */
function captureSession(end = false) {
  // both send the update and pull the session from the scope
  if (end) {
    endSession();
    return;
  }

  // only send the update
  _sendSessionUpdate();
}

exports.addBreadcrumb = addBreadcrumb;
exports.captureCheckIn = captureCheckIn;
exports.captureEvent = captureEvent;
exports.captureException = captureException;
exports.captureMessage = captureMessage;
exports.captureSession = captureSession;
exports.close = close;
exports.configureScope = configureScope;
exports.endSession = endSession;
exports.flush = flush;
exports.getClient = getClient;
exports.getCurrentScope = getCurrentScope;
exports.isInitialized = isInitialized;
exports.lastEventId = lastEventId;
exports.setContext = setContext;
exports.setExtra = setExtra;
exports.setExtras = setExtras;
exports.setTag = setTag;
exports.setTags = setTags;
exports.setUser = setUser;
exports.startSession = startSession;
exports.startTransaction = startTransaction;
exports.withActiveSpan = withActiveSpan;
exports.withIsolationScope = withIsolationScope;
exports.withMonitor = withMonitor;
exports.withScope = withScope;
//# sourceMappingURL=exports.js.map
