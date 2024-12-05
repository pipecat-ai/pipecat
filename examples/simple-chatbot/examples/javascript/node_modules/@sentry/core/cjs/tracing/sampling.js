Object.defineProperty(exports, '__esModule', { value: true });

const utils = require('@sentry/utils');
const debugBuild = require('../debug-build.js');
const semanticAttributes = require('../semanticAttributes.js');
const hasTracingEnabled = require('../utils/hasTracingEnabled.js');
const spanUtils = require('../utils/spanUtils.js');

/**
 * Makes a sampling decision for the given transaction and stores it on the transaction.
 *
 * Called every time a transaction is created. Only transactions which emerge with a `sampled` value of `true` will be
 * sent to Sentry.
 *
 * This method muttes the given `transaction` and will set the `sampled` value on it.
 * It returns the same transaction, for convenience.
 */
function sampleTransaction(
  transaction,
  options,
  samplingContext,
) {
  // nothing to do if tracing is not enabled
  if (!hasTracingEnabled.hasTracingEnabled(options)) {
    // eslint-disable-next-line deprecation/deprecation
    transaction.sampled = false;
    return transaction;
  }

  // if the user has forced a sampling decision by passing a `sampled` value in their transaction context, go with that
  // eslint-disable-next-line deprecation/deprecation
  if (transaction.sampled !== undefined) {
    // eslint-disable-next-line deprecation/deprecation
    transaction.setAttribute(semanticAttributes.SEMANTIC_ATTRIBUTE_SENTRY_SAMPLE_RATE, Number(transaction.sampled));
    return transaction;
  }

  // we would have bailed already if neither `tracesSampler` nor `tracesSampleRate` nor `enableTracing` were defined, so one of these should
  // work; prefer the hook if so
  let sampleRate;
  if (typeof options.tracesSampler === 'function') {
    sampleRate = options.tracesSampler(samplingContext);
    transaction.setAttribute(semanticAttributes.SEMANTIC_ATTRIBUTE_SENTRY_SAMPLE_RATE, Number(sampleRate));
  } else if (samplingContext.parentSampled !== undefined) {
    sampleRate = samplingContext.parentSampled;
  } else if (typeof options.tracesSampleRate !== 'undefined') {
    sampleRate = options.tracesSampleRate;
    transaction.setAttribute(semanticAttributes.SEMANTIC_ATTRIBUTE_SENTRY_SAMPLE_RATE, Number(sampleRate));
  } else {
    // When `enableTracing === true`, we use a sample rate of 100%
    sampleRate = 1;
    transaction.setAttribute(semanticAttributes.SEMANTIC_ATTRIBUTE_SENTRY_SAMPLE_RATE, sampleRate);
  }

  // Since this is coming from the user (or from a function provided by the user), who knows what we might get. (The
  // only valid values are booleans or numbers between 0 and 1.)
  if (!isValidSampleRate(sampleRate)) {
    debugBuild.DEBUG_BUILD && utils.logger.warn('[Tracing] Discarding transaction because of invalid sample rate.');
    // eslint-disable-next-line deprecation/deprecation
    transaction.sampled = false;
    return transaction;
  }

  // if the function returned 0 (or false), or if `tracesSampleRate` is 0, it's a sign the transaction should be dropped
  if (!sampleRate) {
    debugBuild.DEBUG_BUILD &&
      utils.logger.log(
        `[Tracing] Discarding transaction because ${
          typeof options.tracesSampler === 'function'
            ? 'tracesSampler returned 0 or false'
            : 'a negative sampling decision was inherited or tracesSampleRate is set to 0'
        }`,
      );
    // eslint-disable-next-line deprecation/deprecation
    transaction.sampled = false;
    return transaction;
  }

  // Now we roll the dice. Math.random is inclusive of 0, but not of 1, so strict < is safe here. In case sampleRate is
  // a boolean, the < comparison will cause it to be automatically cast to 1 if it's true and 0 if it's false.
  // eslint-disable-next-line deprecation/deprecation
  transaction.sampled = Math.random() < (sampleRate );

  // if we're not going to keep it, we're done
  // eslint-disable-next-line deprecation/deprecation
  if (!transaction.sampled) {
    debugBuild.DEBUG_BUILD &&
      utils.logger.log(
        `[Tracing] Discarding transaction because it's not included in the random sample (sampling rate = ${Number(
          sampleRate,
        )})`,
      );
    return transaction;
  }

  debugBuild.DEBUG_BUILD &&
    // eslint-disable-next-line deprecation/deprecation
    utils.logger.log(`[Tracing] starting ${transaction.op} transaction - ${spanUtils.spanToJSON(transaction).description}`);
  return transaction;
}

/**
 * Checks the given sample rate to make sure it is valid type and value (a boolean, or a number between 0 and 1).
 */
function isValidSampleRate(rate) {
  // we need to check NaN explicitly because it's of type 'number' and therefore wouldn't get caught by this typecheck
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  if (utils.isNaN(rate) || !(typeof rate === 'number' || typeof rate === 'boolean')) {
    debugBuild.DEBUG_BUILD &&
      utils.logger.warn(
        `[Tracing] Given sample rate is invalid. Sample rate must be a boolean or a number between 0 and 1. Got ${JSON.stringify(
          rate,
        )} of type ${JSON.stringify(typeof rate)}.`,
      );
    return false;
  }

  // in case sampleRate is a boolean, it will get automatically cast to 1 if it's true and 0 if it's false
  if (rate < 0 || rate > 1) {
    debugBuild.DEBUG_BUILD &&
      utils.logger.warn(`[Tracing] Given sample rate is invalid. Sample rate must be between 0 and 1. Got ${rate}.`);
    return false;
  }
  return true;
}

exports.isValidSampleRate = isValidSampleRate;
exports.sampleTransaction = sampleTransaction;
//# sourceMappingURL=sampling.js.map
