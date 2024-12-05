import { logger } from '@sentry/utils';
import { DEBUG_BUILD } from '../debug-build.js';
import { getMainCarrier } from '../hub.js';
import { spanToTraceHeader } from '../utils/spanUtils.js';
import { registerErrorInstrumentation } from './errors.js';
import { IdleTransaction } from './idletransaction.js';
import { sampleTransaction } from './sampling.js';
import { Transaction } from './transaction.js';

/** Returns all trace headers that are currently on the top scope. */
// eslint-disable-next-line deprecation/deprecation
function traceHeaders() {
  // eslint-disable-next-line deprecation/deprecation
  const scope = this.getScope();
  // eslint-disable-next-line deprecation/deprecation
  const span = scope.getSpan();

  return span
    ? {
        'sentry-trace': spanToTraceHeader(span),
      }
    : {};
}

/**
 * Creates a new transaction and adds a sampling decision if it doesn't yet have one.
 *
 * The Hub.startTransaction method delegates to this method to do its work, passing the Hub instance in as `this`, as if
 * it had been called on the hub directly. Exists as a separate function so that it can be injected into the class as an
 * "extension method."
 *
 * @param this: The Hub starting the transaction
 * @param transactionContext: Data used to configure the transaction
 * @param CustomSamplingContext: Optional data to be provided to the `tracesSampler` function (if any)
 *
 * @returns The new transaction
 *
 * @see {@link Hub.startTransaction}
 */
function _startTransaction(
  // eslint-disable-next-line deprecation/deprecation

  transactionContext,
  customSamplingContext,
) {
  // eslint-disable-next-line deprecation/deprecation
  const client = this.getClient();
  const options = (client && client.getOptions()) || {};

  const configInstrumenter = options.instrumenter || 'sentry';
  const transactionInstrumenter = transactionContext.instrumenter || 'sentry';

  if (configInstrumenter !== transactionInstrumenter) {
    DEBUG_BUILD &&
      logger.error(
        `A transaction was started with instrumenter=\`${transactionInstrumenter}\`, but the SDK is configured with the \`${configInstrumenter}\` instrumenter.
The transaction will not be sampled. Please use the ${configInstrumenter} instrumentation to start transactions.`,
      );

    // eslint-disable-next-line deprecation/deprecation
    transactionContext.sampled = false;
  }

  // eslint-disable-next-line deprecation/deprecation
  let transaction = new Transaction(transactionContext, this);
  transaction = sampleTransaction(transaction, options, {
    name: transactionContext.name,
    parentSampled: transactionContext.parentSampled,
    transactionContext,
    attributes: {
      // eslint-disable-next-line deprecation/deprecation
      ...transactionContext.data,
      ...transactionContext.attributes,
    },
    ...customSamplingContext,
  });
  if (transaction.isRecording()) {
    transaction.initSpanRecorder(options._experiments && (options._experiments.maxSpans ));
  }
  if (client && client.emit) {
    client.emit('startTransaction', transaction);
  }
  return transaction;
}

/**
 * Create new idle transaction.
 */
function startIdleTransaction(
  // eslint-disable-next-line deprecation/deprecation
  hub,
  transactionContext,
  idleTimeout,
  finalTimeout,
  onScope,
  customSamplingContext,
  heartbeatInterval,
  delayAutoFinishUntilSignal = false,
) {
  // eslint-disable-next-line deprecation/deprecation
  const client = hub.getClient();
  const options = (client && client.getOptions()) || {};

  // eslint-disable-next-line deprecation/deprecation
  let transaction = new IdleTransaction(
    transactionContext,
    hub,
    idleTimeout,
    finalTimeout,
    heartbeatInterval,
    onScope,
    delayAutoFinishUntilSignal,
  );
  transaction = sampleTransaction(transaction, options, {
    name: transactionContext.name,
    parentSampled: transactionContext.parentSampled,
    transactionContext,
    attributes: {
      // eslint-disable-next-line deprecation/deprecation
      ...transactionContext.data,
      ...transactionContext.attributes,
    },
    ...customSamplingContext,
  });
  if (transaction.isRecording()) {
    transaction.initSpanRecorder(options._experiments && (options._experiments.maxSpans ));
  }
  if (client && client.emit) {
    client.emit('startTransaction', transaction);
  }
  return transaction;
}

/**
 * Adds tracing extensions to the global hub.
 */
function addTracingExtensions() {
  const carrier = getMainCarrier();
  if (!carrier.__SENTRY__) {
    return;
  }
  carrier.__SENTRY__.extensions = carrier.__SENTRY__.extensions || {};
  if (!carrier.__SENTRY__.extensions.startTransaction) {
    carrier.__SENTRY__.extensions.startTransaction = _startTransaction;
  }
  if (!carrier.__SENTRY__.extensions.traceHeaders) {
    carrier.__SENTRY__.extensions.traceHeaders = traceHeaders;
  }

  registerErrorInstrumentation();
}

export { addTracingExtensions, startIdleTransaction };
//# sourceMappingURL=hubextensions.js.map
