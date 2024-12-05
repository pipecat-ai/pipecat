Object.defineProperty(exports, '__esModule', { value: true });

const utils = require('@sentry/utils');
const hub = require('../hub.js');

/**
 * Grabs active transaction off scope.
 *
 * @deprecated You should not rely on the transaction, but just use `startSpan()` APIs instead.
 */
// eslint-disable-next-line deprecation/deprecation
function getActiveTransaction(maybeHub) {
  // eslint-disable-next-line deprecation/deprecation
  const hub$1 = maybeHub || hub.getCurrentHub();
  // eslint-disable-next-line deprecation/deprecation
  const scope = hub$1.getScope();
  // eslint-disable-next-line deprecation/deprecation
  return scope.getTransaction() ;
}

/**
 * The `extractTraceparentData` function and `TRACEPARENT_REGEXP` constant used
 * to be declared in this file. It was later moved into `@sentry/utils` as part of a
 * move to remove `@sentry/tracing` dependencies from `@sentry/node` (`extractTraceparentData`
 * is the only tracing function used by `@sentry/node`).
 *
 * These exports are kept here for backwards compatability's sake.
 *
 * See https://github.com/getsentry/sentry-javascript/issues/4642 for more details.
 *
 * @deprecated Import this function from `@sentry/utils` instead
 */
const extractTraceparentData = utils.extractTraceparentData;

exports.stripUrlQueryAndFragment = utils.stripUrlQueryAndFragment;
exports.extractTraceparentData = extractTraceparentData;
exports.getActiveTransaction = getActiveTransaction;
//# sourceMappingURL=utils.js.map
