import { logger, browserPerformanceTimeOrigin, addHistoryInstrumentationHandler } from '@sentry/utils';
import { DEBUG_BUILD } from '../common/debug-build.js';
import { WINDOW } from './types.js';

/**
 * Default function implementing pageload and navigation transactions
 */
function instrumentRoutingWithDefaults(
  customStartTransaction,
  startTransactionOnPageLoad = true,
  startTransactionOnLocationChange = true,
) {
  if (!WINDOW || !WINDOW.location) {
    DEBUG_BUILD && logger.warn('Could not initialize routing instrumentation due to invalid location');
    return;
  }

  let startingUrl = WINDOW.location.href;

  let activeTransaction;
  if (startTransactionOnPageLoad) {
    activeTransaction = customStartTransaction({
      name: WINDOW.location.pathname,
      // pageload should always start at timeOrigin (and needs to be in s, not ms)
      startTimestamp: browserPerformanceTimeOrigin ? browserPerformanceTimeOrigin / 1000 : undefined,
      op: 'pageload',
      origin: 'auto.pageload.browser',
      metadata: { source: 'url' },
    });
  }

  if (startTransactionOnLocationChange) {
    addHistoryInstrumentationHandler(({ to, from }) => {
      /**
       * This early return is there to account for some cases where a navigation transaction starts right after
       * long-running pageload. We make sure that if `from` is undefined and a valid `startingURL` exists, we don't
       * create an uneccessary navigation transaction.
       *
       * This was hard to duplicate, but this behavior stopped as soon as this fix was applied. This issue might also
       * only be caused in certain development environments where the usage of a hot module reloader is causing
       * errors.
       */
      if (from === undefined && startingUrl && startingUrl.indexOf(to) !== -1) {
        startingUrl = undefined;
        return;
      }

      if (from !== to) {
        startingUrl = undefined;
        if (activeTransaction) {
          DEBUG_BUILD && logger.log(`[Tracing] Finishing current transaction with op: ${activeTransaction.op}`);
          // If there's an open transaction on the scope, we need to finish it before creating an new one.
          activeTransaction.end();
        }
        activeTransaction = customStartTransaction({
          name: WINDOW.location.pathname,
          op: 'navigation',
          origin: 'auto.navigation.browser',
          metadata: { source: 'url' },
        });
      }
    });
  }
}

export { instrumentRoutingWithDefaults };
//# sourceMappingURL=router.js.map
