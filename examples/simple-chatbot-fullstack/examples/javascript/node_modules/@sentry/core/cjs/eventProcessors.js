Object.defineProperty(exports, '__esModule', { value: true });

const utils = require('@sentry/utils');
const debugBuild = require('./debug-build.js');

/**
 * Returns the global event processors.
 * @deprecated Global event processors will be removed in v8.
 */
function getGlobalEventProcessors() {
  return utils.getGlobalSingleton('globalEventProcessors', () => []);
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
  return new utils.SyncPromise((resolve, reject) => {
    const processor = processors[index];
    if (event === null || typeof processor !== 'function') {
      resolve(event);
    } else {
      const result = processor({ ...event }, hint) ;

      debugBuild.DEBUG_BUILD && processor.id && result === null && utils.logger.log(`Event processor "${processor.id}" dropped event`);

      if (utils.isThenable(result)) {
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

exports.addGlobalEventProcessor = addGlobalEventProcessor;
exports.getGlobalEventProcessors = getGlobalEventProcessors;
exports.notifyEventProcessors = notifyEventProcessors;
//# sourceMappingURL=eventProcessors.js.map
