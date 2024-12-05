Object.defineProperty(exports, '__esModule', { value: true });

const utils = require('@sentry/utils');
const debugBuild = require('../debug-build.js');
const hub = require('../hub.js');
const metricSummary = require('../metrics/metric-summary.js');
const semanticAttributes = require('../semanticAttributes.js');
const spanUtils = require('../utils/spanUtils.js');
const dynamicSamplingContext = require('./dynamicSamplingContext.js');
const span = require('./span.js');
const trace = require('./trace.js');

/** JSDoc */
class Transaction extends span.Span  {
  /**
   * The reference to the current hub.
   */
  // eslint-disable-next-line deprecation/deprecation

  // DO NOT yet remove this property, it is used in a hack for v7 backwards compatibility.

  /**
   * This constructor should never be called manually. Those instrumenting tracing should use
   * `Sentry.startTransaction()`, and internal methods should use `hub.startTransaction()`.
   * @internal
   * @hideconstructor
   * @hidden
   *
   * @deprecated Transactions will be removed in v8. Use spans instead.
   */
  // eslint-disable-next-line deprecation/deprecation
   constructor(transactionContext, hub$1) {
    super(transactionContext);
    this._contexts = {};

    // eslint-disable-next-line deprecation/deprecation
    this._hub = hub$1 || hub.getCurrentHub();

    this._name = transactionContext.name || '';

    this._metadata = {
      // eslint-disable-next-line deprecation/deprecation
      ...transactionContext.metadata,
    };

    this._trimEnd = transactionContext.trimEnd;

    // this is because transactions are also spans, and spans have a transaction pointer
    // TODO (v8): Replace this with another way to set the root span
    // eslint-disable-next-line deprecation/deprecation
    this.transaction = this;

    // If Dynamic Sampling Context is provided during the creation of the transaction, we freeze it as it usually means
    // there is incoming Dynamic Sampling Context. (Either through an incoming request, a baggage meta-tag, or other means)
    const incomingDynamicSamplingContext = this._metadata.dynamicSamplingContext;
    if (incomingDynamicSamplingContext) {
      // We shallow copy this in case anything writes to the original reference of the passed in `dynamicSamplingContext`
      this._frozenDynamicSamplingContext = { ...incomingDynamicSamplingContext };
    }
  }

  // This sadly conflicts with the getter/setter ordering :(
  /* eslint-disable @typescript-eslint/member-ordering */

  /**
   * Getter for `name` property.
   * @deprecated Use `spanToJSON(span).description` instead.
   */
   get name() {
    return this._name;
  }

  /**
   * Setter for `name` property, which also sets `source` as custom.
   * @deprecated Use `updateName()` and `setMetadata()` instead.
   */
   set name(newName) {
    // eslint-disable-next-line deprecation/deprecation
    this.setName(newName);
  }

  /**
   * Get the metadata for this transaction.
   * @deprecated Use `spanGetMetadata(transaction)` instead.
   */
   get metadata() {
    // We merge attributes in for backwards compatibility
    return {
      // Defaults
      // eslint-disable-next-line deprecation/deprecation
      source: 'custom',
      spanMetadata: {},

      // Legacy metadata
      ...this._metadata,

      // From attributes
      ...(this._attributes[semanticAttributes.SEMANTIC_ATTRIBUTE_SENTRY_SOURCE] && {
        source: this._attributes[semanticAttributes.SEMANTIC_ATTRIBUTE_SENTRY_SOURCE] ,
      }),
      ...(this._attributes[semanticAttributes.SEMANTIC_ATTRIBUTE_SENTRY_SAMPLE_RATE] && {
        sampleRate: this._attributes[semanticAttributes.SEMANTIC_ATTRIBUTE_SENTRY_SAMPLE_RATE] ,
      }),
    };
  }

  /**
   * Update the metadata for this transaction.
   * @deprecated Use `spanGetMetadata(transaction)` instead.
   */
   set metadata(metadata) {
    this._metadata = metadata;
  }

  /* eslint-enable @typescript-eslint/member-ordering */

  /**
   * Setter for `name` property, which also sets `source` on the metadata.
   *
   * @deprecated Use `.updateName()` and `.setAttribute()` instead.
   */
   setName(name, source = 'custom') {
    this._name = name;
    this.setAttribute(semanticAttributes.SEMANTIC_ATTRIBUTE_SENTRY_SOURCE, source);
  }

  /** @inheritdoc */
   updateName(name) {
    this._name = name;
    return this;
  }

  /**
   * Attaches SpanRecorder to the span itself
   * @param maxlen maximum number of spans that can be recorded
   */
   initSpanRecorder(maxlen = 1000) {
    // eslint-disable-next-line deprecation/deprecation
    if (!this.spanRecorder) {
      // eslint-disable-next-line deprecation/deprecation
      this.spanRecorder = new span.SpanRecorder(maxlen);
    }
    // eslint-disable-next-line deprecation/deprecation
    this.spanRecorder.add(this);
  }

  /**
   * Set the context of a transaction event.
   * @deprecated Use either `.setAttribute()`, or set the context on the scope before creating the transaction.
   */
   setContext(key, context) {
    if (context === null) {
      // eslint-disable-next-line @typescript-eslint/no-dynamic-delete
      delete this._contexts[key];
    } else {
      this._contexts[key] = context;
    }
  }

  /**
   * @inheritDoc
   *
   * @deprecated Use top-level `setMeasurement()` instead.
   */
   setMeasurement(name, value, unit = '') {
    this._measurements[name] = { value, unit };
  }

  /**
   * Store metadata on this transaction.
   * @deprecated Use attributes or store data on the scope instead.
   */
   setMetadata(newMetadata) {
    this._metadata = { ...this._metadata, ...newMetadata };
  }

  /**
   * @inheritDoc
   */
   end(endTimestamp) {
    const timestampInS = spanUtils.spanTimeInputToSeconds(endTimestamp);
    const transaction = this._finishTransaction(timestampInS);
    if (!transaction) {
      return undefined;
    }
    // eslint-disable-next-line deprecation/deprecation
    return this._hub.captureEvent(transaction);
  }

  /**
   * @inheritDoc
   */
   toContext() {
    // eslint-disable-next-line deprecation/deprecation
    const spanContext = super.toContext();

    return utils.dropUndefinedKeys({
      ...spanContext,
      name: this._name,
      trimEnd: this._trimEnd,
    });
  }

  /**
   * @inheritDoc
   */
   updateWithContext(transactionContext) {
    // eslint-disable-next-line deprecation/deprecation
    super.updateWithContext(transactionContext);

    this._name = transactionContext.name || '';
    this._trimEnd = transactionContext.trimEnd;

    return this;
  }

  /**
   * @inheritdoc
   *
   * @experimental
   *
   * @deprecated Use top-level `getDynamicSamplingContextFromSpan` instead.
   */
   getDynamicSamplingContext() {
    return dynamicSamplingContext.getDynamicSamplingContextFromSpan(this);
  }

  /**
   * Override the current hub with a new one.
   * Used if you want another hub to finish the transaction.
   *
   * @internal
   */
  // eslint-disable-next-line deprecation/deprecation
   setHub(hub) {
    this._hub = hub;
  }

  /**
   * Get the profile id of the transaction.
   */
   getProfileId() {
    if (this._contexts !== undefined && this._contexts['profile'] !== undefined) {
      return this._contexts['profile'].profile_id ;
    }
    return undefined;
  }

  /**
   * Finish the transaction & prepare the event to send to Sentry.
   */
   _finishTransaction(endTimestamp) {
    // This transaction is already finished, so we should not flush it again.
    if (this._endTime !== undefined) {
      return undefined;
    }

    if (!this._name) {
      debugBuild.DEBUG_BUILD && utils.logger.warn('Transaction has no name, falling back to `<unlabeled transaction>`.');
      this._name = '<unlabeled transaction>';
    }

    // just sets the end timestamp
    super.end(endTimestamp);

    // eslint-disable-next-line deprecation/deprecation
    const client = this._hub.getClient();
    if (client && client.emit) {
      client.emit('finishTransaction', this);
    }

    if (this._sampled !== true) {
      // At this point if `sampled !== true` we want to discard the transaction.
      debugBuild.DEBUG_BUILD && utils.logger.log('[Tracing] Discarding transaction because its trace was not chosen to be sampled.');

      if (client) {
        client.recordDroppedEvent('sample_rate', 'transaction');
      }

      return undefined;
    }

    // eslint-disable-next-line deprecation/deprecation
    const finishedSpans = this.spanRecorder
      ? // eslint-disable-next-line deprecation/deprecation
        this.spanRecorder.spans.filter(span => span !== this && spanUtils.spanToJSON(span).timestamp)
      : [];

    if (this._trimEnd && finishedSpans.length > 0) {
      const endTimes = finishedSpans.map(span => spanUtils.spanToJSON(span).timestamp).filter(Boolean) ;
      this._endTime = endTimes.reduce((prev, current) => {
        return prev > current ? prev : current;
      });
    }

    const { scope: capturedSpanScope, isolationScope: capturedSpanIsolationScope } = trace.getCapturedScopesOnSpan(this);

    // eslint-disable-next-line deprecation/deprecation
    const { metadata } = this;
    // eslint-disable-next-line deprecation/deprecation
    const { source } = metadata;

    const transaction = {
      contexts: {
        ...this._contexts,
        // We don't want to override trace context
        trace: spanUtils.spanToTraceContext(this),
      },
      // TODO: Pass spans serialized via `spanToJSON()` here instead in v8.
      spans: finishedSpans,
      start_timestamp: this._startTime,
      // eslint-disable-next-line deprecation/deprecation
      tags: this.tags,
      timestamp: this._endTime,
      transaction: this._name,
      type: 'transaction',
      sdkProcessingMetadata: {
        ...metadata,
        capturedSpanScope,
        capturedSpanIsolationScope,
        ...utils.dropUndefinedKeys({
          dynamicSamplingContext: dynamicSamplingContext.getDynamicSamplingContextFromSpan(this),
        }),
      },
      _metrics_summary: metricSummary.getMetricSummaryJsonForSpan(this),
      ...(source && {
        transaction_info: {
          source,
        },
      }),
    };

    const hasMeasurements = Object.keys(this._measurements).length > 0;

    if (hasMeasurements) {
      debugBuild.DEBUG_BUILD &&
        utils.logger.log(
          '[Measurements] Adding measurements to transaction',
          JSON.stringify(this._measurements, undefined, 2),
        );
      transaction.measurements = this._measurements;
    }

    // eslint-disable-next-line deprecation/deprecation
    debugBuild.DEBUG_BUILD && utils.logger.log(`[Tracing] Finishing ${this.op} transaction: ${this._name}.`);

    return transaction;
  }
}

exports.Transaction = Transaction;
//# sourceMappingURL=transaction.js.map
