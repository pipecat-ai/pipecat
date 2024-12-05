Object.defineProperty(exports, '__esModule', { value: true });

const core = require('@sentry/core');
const utils = require('@sentry/utils');
const debugBuild = require('../common/debug-build.js');
const types = require('./types.js');

/**
 * Add a listener that cancels and finishes a transaction when the global
 * document is hidden.
 */
function registerBackgroundTabDetection() {
  if (types.WINDOW.document) {
    types.WINDOW.document.addEventListener('visibilitychange', () => {
      // eslint-disable-next-line deprecation/deprecation
      const activeTransaction = core.getActiveTransaction() ;
      if (types.WINDOW.document.hidden && activeTransaction) {
        const statusType = 'cancelled';

        const { op, status } = core.spanToJSON(activeTransaction);

        debugBuild.DEBUG_BUILD &&
          utils.logger.log(`[Tracing] Transaction: ${statusType} -> since tab moved to the background, op: ${op}`);
        // We should not set status if it is already set, this prevent important statuses like
        // error or data loss from being overwritten on transaction.
        if (!status) {
          activeTransaction.setStatus(statusType);
        }
        // TODO: Can we rewrite this to an attribute?
        // eslint-disable-next-line deprecation/deprecation
        activeTransaction.setTag('visibilitychange', 'document.hidden');
        activeTransaction.end();
      }
    });
  } else {
    debugBuild.DEBUG_BUILD && utils.logger.warn('[Tracing] Could not set up background tab detection due to lack of global document');
  }
}

exports.registerBackgroundTabDetection = registerBackgroundTabDetection;
//# sourceMappingURL=backgroundtab.js.map
