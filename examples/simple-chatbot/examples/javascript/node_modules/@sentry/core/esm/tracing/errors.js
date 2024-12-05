import { addGlobalErrorInstrumentationHandler, addGlobalUnhandledRejectionInstrumentationHandler, logger } from '@sentry/utils';
import { DEBUG_BUILD } from '../debug-build.js';
import { getActiveTransaction } from './utils.js';

let errorsInstrumented = false;

/**
 * Configures global error listeners
 */
function registerErrorInstrumentation() {
  if (errorsInstrumented) {
    return;
  }

  errorsInstrumented = true;
  addGlobalErrorInstrumentationHandler(errorCallback);
  addGlobalUnhandledRejectionInstrumentationHandler(errorCallback);
}

/**
 * If an error or unhandled promise occurs, we mark the active transaction as failed
 */
function errorCallback() {
  // eslint-disable-next-line deprecation/deprecation
  const activeTransaction = getActiveTransaction();
  if (activeTransaction) {
    const status = 'internal_error';
    DEBUG_BUILD && logger.log(`[Tracing] Transaction: ${status} -> Global error occured`);
    activeTransaction.setStatus(status);
  }
}

// The function name will be lost when bundling but we need to be able to identify this listener later to maintain the
// node.js default exit behaviour
errorCallback.tag = 'sentry_tracingErrorCallback';

export { registerErrorInstrumentation };
//# sourceMappingURL=errors.js.map
