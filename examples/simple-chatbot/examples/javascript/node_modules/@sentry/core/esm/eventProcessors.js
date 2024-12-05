import { SyncPromise, logger, isThenable, getGlobalSingleton } from '@sentry/utils';
import { DEBUG_BUILD } from './debug-build.js';

/**
 * Returns the global event processors.
 * @deprecated Global event processors will be removed in v8.
 */
function getGlobalEventProcessors() {
  return getGlobalSingleton('globalEventProcessors', () => []);
}

/**
 * Add a EventProcessor to be kept globally.
 * @deprecated Use `addEventProcessor` instead. Global event processors will be removed in v8.
 */
function addGlobalEventProcessor(callback) {
  // eslint-disable-next-line deprecation/deprecation
  getGlobalEventProcessors().push(callback);
}

/**
 * Process an array of event processors, returning the processed event (or `null` if the event was dropped).
 */
function notifyEventProcessors(
  processors,
  event,
  hint,
  index = 0,
) {
  return new SyncPromise((resolve, reject) => {
    const processor = processors[index];
    if (event === null || typeof processor !== 'function') {
      resolve(event);
    } else {
      const result = processor({ ...event }, hint) ;

      DEBUG_BUILD && processor.id && result === null && logger.log(`Event processor "${processor.id}" dropped event`);

      if (isThenable(result)) {
        void result
          .then(final => notifyEventProcessors(processors, final, hint, index + 1).then(resolve))
          .then(null, reject);
      } else {
        void notifyEventProcessors(processors, result, hint, index + 1)
          .then(resolve)
          .then(null, reject);
      }
    }
  });
}

export { addGlobalEventProcessor, getGlobalEventProcessors, notifyEventProcessors };
//# sourceMappingURL=eventProcessors.js.map
