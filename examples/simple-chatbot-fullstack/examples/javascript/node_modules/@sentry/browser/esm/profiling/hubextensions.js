import { spanToJSON } from '@sentry/core';
import { logger, timestampInSeconds, uuid4 } from '@sentry/utils';
import { DEBUG_BUILD } from '../debug-build.js';
import { WINDOW } from '../helpers.js';
import { shouldProfileTransaction, isAutomatedPageLoadTransaction, startJSSelfProfile, MAX_PROFILE_DURATION_MS, addProfileToGlobalCache } from './utils.js';

/* eslint-disable complexity */

/**
 * Safety wrapper for startTransaction for the unlikely case that transaction starts before tracing is imported -
 * if that happens we want to avoid throwing an error from profiling code.
 * see https://github.com/getsentry/sentry-javascript/issues/4731.
 *
 * @experimental
 */
function onProfilingStartRouteTransaction(transaction) {
  if (!transaction) {
    if (DEBUG_BUILD) {
      logger.log('[Profiling] Transaction is undefined, skipping profiling');
    }
    return transaction;
  }

  if (shouldProfileTransaction(transaction)) {
    return startProfileForTransaction(transaction);
  }

  return transaction;
}

/**
 * Wraps startTransaction and stopTransaction with profiling related logic.
 * startProfileForTransaction is called after the call to startTransaction in order to avoid our own code from
 * being profiled. Because of that same reason, stopProfiling is called before the call to stopTransaction.
 */
function startProfileForTransaction(transaction) {
  // Start the profiler and get the profiler instance.
  let startTimestamp;
  if (isAutomatedPageLoadTransaction(transaction)) {
    startTimestamp = timestampInSeconds() * 1000;
  }

  const profiler = startJSSelfProfile();

  // We failed to construct the profiler, fallback to original transaction.
  // No need to log anything as this has already been logged in startProfile.
  if (!profiler) {
    return transaction;
  }

  if (DEBUG_BUILD) {
    logger.log(`[Profiling] started profiling transaction: ${spanToJSON(transaction).description}`);
  }

  // We create "unique" transaction names to avoid concurrent transactions with same names
  // from being ignored by the profiler. From here on, only this transaction name should be used when
  // calling the profiler methods. Note: we log the original name to the user to avoid confusion.
  const profileId = uuid4();

  /**
   * Idempotent handler for profile stop
   */
  async function onProfileHandler() {
    // Check if the profile exists and return it the behavior has to be idempotent as users may call transaction.finish multiple times.
    if (!transaction) {
      return null;
    }
    // Satisfy the type checker, but profiler will always be defined here.
    if (!profiler) {
      return null;
    }

    return profiler
      .stop()
      .then((profile) => {
        if (maxDurationTimeoutID) {
          WINDOW.clearTimeout(maxDurationTimeoutID);
          maxDurationTimeoutID = undefined;
        }

        if (DEBUG_BUILD) {
          logger.log(`[Profiling] stopped profiling of transaction: ${spanToJSON(transaction).description}`);
        }

        // In case of an overlapping transaction, stopProfiling may return null and silently ignore the overlapping profile.
        if (!profile) {
          if (DEBUG_BUILD) {
            logger.log(
              `[Profiling] profiler returned null profile for: ${spanToJSON(transaction).description}`,
              'this may indicate an overlapping transaction or a call to stopProfiling with a profile title that was never started',
            );
          }
          return null;
        }

        addProfileToGlobalCache(profileId, profile);
        return null;
      })
      .catch(error => {
        if (DEBUG_BUILD) {
          logger.log('[Profiling] error while stopping profiler:', error);
        }
        return null;
      });
  }

  // Enqueue a timeout to prevent profiles from running over max duration.
  let maxDurationTimeoutID = WINDOW.setTimeout(() => {
    if (DEBUG_BUILD) {
      logger.log(
        '[Profiling] max profile duration elapsed, stopping profiling for:',
        spanToJSON(transaction).description,
      );
    }
    // If the timeout exceeds, we want to stop profiling, but not finish the transaction
    // eslint-disable-next-line @typescript-eslint/no-floating-promises
    onProfileHandler();
  }, MAX_PROFILE_DURATION_MS);

  // We need to reference the original end call to avoid creating an infinite loop
  const originalEnd = transaction.end.bind(transaction);

  /**
   * Wraps startTransaction and stopTransaction with profiling related logic.
   * startProfiling is called after the call to startTransaction in order to avoid our own code from
   * being profiled. Because of that same reason, stopProfiling is called before the call to stopTransaction.
   */
  function profilingWrappedTransactionEnd() {
    if (!transaction) {
      return originalEnd();
    }
    // onProfileHandler should always return the same profile even if this is called multiple times.
    // Always call onProfileHandler to ensure stopProfiling is called and the timeout is cleared.
    void onProfileHandler().then(
      () => {
        // TODO: Can we rewrite this to use attributes?
        // eslint-disable-next-line deprecation/deprecation
        transaction.setContext('profile', { profile_id: profileId, start_timestamp: startTimestamp });
        originalEnd();
      },
      () => {
        // If onProfileHandler fails, we still want to call the original finish method.
        originalEnd();
      },
    );

    return transaction;
  }

  transaction.end = profilingWrappedTransactionEnd;
  return transaction;
}

export { onProfilingStartRouteTransaction, startProfileForTransaction };
//# sourceMappingURL=hubextensions.js.map
