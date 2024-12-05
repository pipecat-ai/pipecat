Object.defineProperty(exports, '__esModule', { value: true });

const utils = require('@sentry/utils');
const debugBuild = require('../debug-build.js');

const MIN_DELAY = 100; // 100 ms
const START_DELAY = 5000; // 5 seconds
const MAX_DELAY = 3.6e6; // 1 hour

function log(msg, error) {
  debugBuild.DEBUG_BUILD && utils.logger.info(`[Offline]: ${msg}`, error);
}

/**
 * Wraps a transport and stores and retries events when they fail to send.
 *
 * @param createTransport The transport to wrap.
 */
function makeOfflineTransport(
  createTransport,
) {
  return options => {
    const transport = createTransport(options);
    const store = options.createStore ? options.createStore(options) : undefined;

    let retryDelay = START_DELAY;
    let flushTimer;

    function shouldQueue(env, error, retryDelay) {
      // We don't queue Session Replay envelopes because they are:
      // - Ordered and Replay relies on the response status to know when they're successfully sent.
      // - Likely to fill the queue quickly and block other events from being sent.
      // We also want to drop client reports because they can be generated when we retry sending events while offline.
      if (utils.envelopeContainsItemType(env, ['replay_event', 'replay_recording', 'client_report'])) {
        return false;
      }

      if (options.shouldStore) {
        return options.shouldStore(env, error, retryDelay);
      }

      return true;
    }

    function flushIn(delay) {
      if (!store) {
        return;
      }

      if (flushTimer) {
        clearTimeout(flushTimer );
      }

      flushTimer = setTimeout(async () => {
        flushTimer = undefined;

        const found = await store.pop();
        if (found) {
          log('Attempting to send previously queued event');
          void send(found).catch(e => {
            log('Failed to retry sending', e);
          });
        }
      }, delay) ;

      // We need to unref the timer in node.js, otherwise the node process never exit.
      if (typeof flushTimer !== 'number' && flushTimer.unref) {
        flushTimer.unref();
      }
    }

    function flushWithBackOff() {
      if (flushTimer) {
        return;
      }

      flushIn(retryDelay);

      retryDelay = Math.min(retryDelay * 2, MAX_DELAY);
    }

    async function send(envelope) {
      try {
        const result = await transport.send(envelope);

        let delay = MIN_DELAY;

        if (result) {
          // If there's a retry-after header, use that as the next delay.
          if (result.headers && result.headers['retry-after']) {
            delay = utils.parseRetryAfterHeader(result.headers['retry-after']);
          } // If we have a server error, return now so we don't flush the queue.
          else if ((result.statusCode || 0) >= 400) {
            return result;
          }
        }

        flushIn(delay);
        retryDelay = START_DELAY;
        return result;
      } catch (e) {
        if (store && (await shouldQueue(envelope, e , retryDelay))) {
          await store.insert(envelope);
          flushWithBackOff();
          log('Error sending. Event queued', e );
          return {};
        } else {
          throw e;
        }
      }
    }

    if (options.flushAtStartup) {
      flushWithBackOff();
    }

    return {
      send,
      flush: t => transport.flush(t),
    };
  };
}

exports.MIN_DELAY = MIN_DELAY;
exports.START_DELAY = START_DELAY;
exports.makeOfflineTransport = makeOfflineTransport;
//# sourceMappingURL=offline.js.map
