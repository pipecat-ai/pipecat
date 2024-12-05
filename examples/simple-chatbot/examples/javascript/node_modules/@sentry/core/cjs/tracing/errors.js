Object.defineProperty(exports, '__esModule', { value: true });

const utils = require('@sentry/utils');
const debugBuild = require('../debug-build.js');
const utils$1 = require('./utils.js');

let errorsInstrumented = false;

/**
 * Configures global error listeners
 */
function registerErrorInstrumentation() {
  if (errorsInstrumented) {
    return;
  }

  errorsInstrumented = true;
  utils.addGlobalErrorInstrumentationHandler(errorCallback);
  utils.addGlobalUnhandledRejectionInstrumentationHandler(errorCallback);
}

/**
 * If an error or unhandled promise occurs, we mark the active transaction as failed
 */
function errorCallback() {
  // eslint-disable-next-line deprecation/deprecation
  const activeTransaction = utils$1.getActiveTransaction();
  if (activeTransaction) {
    const status = 'internal_error';
    debugBuild.DEBUG_BUILD && utils.logger.log(`[Tracing] Transaction: ${status} -> Global error occured`);
    activeTransaction.setStatus(status);
  }
}

// The function name will be lost when bundling but we need to be able to identify this listener later to maintain the
// node.js default exit behaviour
errorCallback.tag = 'sentry_tracingErrorCallback';

exports.registerErrorInstrumentation = registerErrorInstrumentation;
//# sourceMappingURL=errors.js.map
