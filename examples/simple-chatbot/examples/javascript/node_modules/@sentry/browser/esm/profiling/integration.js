import { defineIntegration, convertIntegrationFnToClass, getCurrentScope } from '@sentry/core';
import { logger } from '@sentry/utils';
import { DEBUG_BUILD } from '../debug-build.js';
import { startProfileForTransaction } from './hubextensions.js';
import { isAutomatedPageLoadTransaction, shouldProfileTransaction, getActiveProfilesCount, findProfiledTransactionsFromEnvelope, takeProfileFromGlobalCache, createProfilingEvent, addProfilesToEnvelope } from './utils.js';

const INTEGRATION_NAME = 'BrowserProfiling';

const _browserProfilingIntegration = (() => {
  return {
    name: INTEGRATION_NAME,
    // TODO v8: Remove this
    setupOnce() {}, // eslint-disable-line @typescript-eslint/no-empty-function
    setup(client) {
      const scope = getCurrentScope();

      // eslint-disable-next-line deprecation/deprecation
      const transaction = scope.getTransaction();

      if (transaction && isAutomatedPageLoadTransaction(transaction)) {
        if (shouldProfileTransaction(transaction)) {
          startProfileForTransaction(transaction);
        }
      }

      if (typeof client.on !== 'function') {
        logger.warn('[Profiling] Client does not support hooks, profiling will be disabled');
        return;
      }

      client.on('startTransaction', (transaction) => {
        if (shouldProfileTransaction(transaction)) {
          startProfileForTransaction(transaction);
        }
      });

      client.on('beforeEnvelope', (envelope) => {
        // if not profiles are in queue, there is nothing to add to the envelope.
        if (!getActiveProfilesCount()) {
          return;
        }

        const profiledTransactionEvents = findProfiledTransactionsFromEnvelope(envelope);
        if (!profiledTransactionEvents.length) {
          return;
        }

        const profilesToAddToEnvelope = [];

        for (const profiledTransaction of profiledTransactionEvents) {
          const context = profiledTransaction && profiledTransaction.contexts;
          const profile_id = context && context['profile'] && context['profile']['profile_id'];
          const start_timestamp = context && context['profile'] && context['profile']['start_timestamp'];

          if (typeof profile_id !== 'string') {
            DEBUG_BUILD && logger.log('[Profiling] cannot find profile for a transaction without a profile context');
            continue;
          }

          if (!profile_id) {
            DEBUG_BUILD && logger.log('[Profiling] cannot find profile for a transaction without a profile context');
            continue;
          }

          // Remove the profile from the transaction context before sending, relay will take care of the rest.
          if (context && context['profile']) {
            delete context.profile;
          }

          const profile = takeProfileFromGlobalCache(profile_id);
          if (!profile) {
            DEBUG_BUILD && logger.log(`[Profiling] Could not retrieve profile for transaction: ${profile_id}`);
            continue;
          }

          const profileEvent = createProfilingEvent(
            profile_id,
            start_timestamp ,
            profile,
            profiledTransaction ,
          );
          if (profileEvent) {
            profilesToAddToEnvelope.push(profileEvent);
          }
        }

        addProfilesToEnvelope(envelope , profilesToAddToEnvelope);
      });
    },
  };
}) ;

const browserProfilingIntegration = defineIntegration(_browserProfilingIntegration);

/**
 * Browser profiling integration. Stores any event that has contexts["profile"]["profile_id"]
 * This exists because we do not want to await async profiler.stop calls as transaction.finish is called
 * in a synchronous context. Instead, we handle sending the profile async from the promise callback and
 * rely on being able to pull the event from the cache when we need to construct the envelope. This makes the
 * integration less reliable as we might be dropping profiles when the cache is full.
 *
 * @experimental
 * @deprecated Use `browserProfilingIntegration()` instead.
 */
// eslint-disable-next-line deprecation/deprecation
const BrowserProfilingIntegration = convertIntegrationFnToClass(
  INTEGRATION_NAME,
  browserProfilingIntegration,
) ;

// eslint-disable-next-line deprecation/deprecation

export { BrowserProfilingIntegration, browserProfilingIntegration };
//# sourceMappingURL=integration.js.map
