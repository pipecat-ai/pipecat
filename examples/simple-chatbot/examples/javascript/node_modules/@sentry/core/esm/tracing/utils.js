import { extractTraceparentData as extractTraceparentData$1 } from '@sentry/utils';
export { stripUrlQueryAndFragment } from '@sentry/utils';
import { getCurrentHub } from '../hub.js';

/**
 * Grabs active transaction off scope.
 *
 * @deprecated You should not rely on the transaction, but just use `startSpan()` APIs instead.
 */
// eslint-disable-next-line deprecation/deprecation
function getActiveTransaction(maybeHub) {
  // eslint-disable-next-line deprecation/deprecation
  const hub = maybeHub || getCurrentHub();
  // eslint-disable-next-line deprecation/deprecation
  const scope = hub.getScope();
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
const extractTraceparentData = extractTraceparentData$1;

export { extractTraceparentData, getActiveTransaction };
//# sourceMappingURL=utils.js.map
