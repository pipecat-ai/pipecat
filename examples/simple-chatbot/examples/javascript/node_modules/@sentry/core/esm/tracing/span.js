import { uuid4, timestampInSeconds, logger, dropUndefinedKeys } from '@sentry/utils';
import { DEBUG_BUILD } from '../debug-build.js';
import { getMetricSummaryJsonForSpan } from '../metrics/metric-summary.js';
import { SEMANTIC_ATTRIBUTE_SENTRY_ORIGIN, SEMANTIC_ATTRIBUTE_SENTRY_OP, SEMANTIC_ATTRIBUTE_PROFILE_ID } from '../semanticAttributes.js';
import { getRootSpan } from '../utils/getRootSpan.js';
import { TRACE_FLAG_SAMPLED, TRACE_FLAG_NONE, spanToJSON, spanTimeInputToSeconds, spanToTraceHeader, spanToTraceContext } from '../utils/spanUtils.js';
import { setHttpStatus } from './spanstatus.js';

/**
 * Keeps track of finished spans for a given transaction
 * @internal
 * @hideconstructor
 * @hidden
 */
class SpanRecorder {

   constructor(maxlen = 1000) {
    this._maxlen = maxlen;
    this.spans = [];
  }

  /**
   * This is just so that we don't run out of memory while recording a lot
   * of spans. At some point we just stop and flush out the start of the
   * trace tree (i.e.the first n spans with the smallest
   * start_timestamp).
   */
   add(span) {
    if (this.spans.length > this._maxlen) {
      // eslint-disable-next-line deprecation/deprecation
      span.spanRecorder = undefined;
    } else {
      this.spans.push(span);
    }
  }
}

/**
 * Span contains all data about a span
 */
class Span  {
  /**
   * Tags for the span.
   * @deprecated Use `spanToJSON(span).atttributes` instead.
   */

  /**
   * Data for the span.
   * @deprecated Use `spanToJSON(span).atttributes` instead.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any

  /**
   * List of spans that were finalized
   *
   * @deprecated This property will no longer be public. Span recording will be handled internally.
   */

  /**
   * @inheritDoc
   * @deprecated Use top level `Sentry.getRootSpan()` instead
   */

  /**
   * The instrumenter that created this span.
   *
   * TODO (v8): This can probably be replaced by an `instanceOf` check of the span class.
   *            the instrumenter can only be sentry or otel so we can check the span instance
   *            to verify which one it is and remove this field entirely.
   *
   * @deprecated This field will be removed.
   */

  /** Epoch timestamp in seconds when the span started. */

  /** Epoch timestamp in seconds when the span ended. */

  /** Internal keeper of the status */

  /**
   * You should never call the constructor manually, always use `Sentry.startTransaction()`
   * or call `startChild()` on an existing span.
   * @internal
   * @hideconstructor
   * @hidden
   */
   constructor(spanContext = {}) {
    this._traceId = spanContext.traceId || uuid4();
    this._spanId = spanContext.spanId || uuid4().substring(16);
    this._startTime = spanContext.startTimestamp || timestampInSeconds();
    // eslint-disable-next-line deprecation/deprecation
    this.tags = spanContext.tags ? { ...spanContext.tags } : {};
    // eslint-disable-next-line deprecation/deprecation
    this.data = spanContext.data ? { ...spanContext.data } : {};
    // eslint-disable-next-line deprecation/deprecation
    this.instrumenter = spanContext.instrumenter || 'sentry';

    this._attributes = {};
    this.setAttributes({
      [SEMANTIC_ATTRIBUTE_SENTRY_ORIGIN]: spanContext.origin || 'manual',
      [SEMANTIC_ATTRIBUTE_SENTRY_OP]: spanContext.op,
      ...spanContext.attributes,
    });

    // eslint-disable-next-line deprecation/deprecation
    this._name = spanContext.name || spanContext.description;

    if (spanContext.parentSpanId) {
      this._parentSpanId = spanContext.parentSpanId;
    }
    // We want to include booleans as well here
    if ('sampled' in spanContext) {
      this._sampled = spanContext.sampled;
    }
    if (spanContext.status) {
      this._status = spanContext.status;
    }
    if (spanContext.endTimestamp) {
      this._endTime = spanContext.endTimestamp;
    }
    if (spanContext.exclusiveTime !== undefined) {
      this._exclusiveTime = spanContext.exclusiveTime;
    }
    this._measurements = spanContext.measurements ? { ...spanContext.measurements } : {};
  }

  // This rule conflicts with another eslint rule :(
  /* eslint-disable @typescript-eslint/member-ordering */

  /**
   * An alias for `description` of the Span.
   * @deprecated Use `spanToJSON(span).description` instead.
   */
   get name() {
    return this._name || '';
  }

  /**
   * Update the name of the span.
   * @deprecated Use `spanToJSON(span).description` instead.
   */
   set name(name) {
    this.updateName(name);
  }

  /**
   * Get the description of the Span.
   * @deprecated Use `spanToJSON(span).description` instead.
   */
   get description() {
    return this._name;
  }

  /**
   * Get the description of the Span.
   * @deprecated Use `spanToJSON(span).description` instead.
   */
   set description(description) {
    this._name = description;
  }

  /**
   * The ID of the trace.
   * @deprecated Use `spanContext().traceId` instead.
   */
   get traceId() {
    return this._traceId;
  }

  /**
   * The ID of the trace.
   * @deprecated You cannot update the traceId of a span after span creation.
   */
   set traceId(traceId) {
    this._traceId = traceId;
  }

  /**
   * The ID of the span.
   * @deprecated Use `spanContext().spanId` instead.
   */
   get spanId() {
    return this._spanId;
  }

  /**
   * The ID of the span.
   * @deprecated You cannot update the spanId of a span after span creation.
   */
   set spanId(spanId) {
    this._spanId = spanId;
  }

  /**
   * @inheritDoc
   *
   * @deprecated Use `startSpan` functions instead.
   */
   set parentSpanId(string) {
    this._parentSpanId = string;
  }

  /**
   * @inheritDoc
   *
   * @deprecated Use `spanToJSON(span).parent_span_id` instead.
   */
   get parentSpanId() {
    return this._parentSpanId;
  }

  /**
   * Was this span chosen to be sent as part of the sample?
   * @deprecated Use `isRecording()` instead.
   */
   get sampled() {
    return this._sampled;
  }

  /**
   * Was this span chosen to be sent as part of the sample?
   * @deprecated You cannot update the sampling decision of a span after span creation.
   */
   set sampled(sampled) {
    this._sampled = sampled;
  }

  /**
   * Attributes for the span.
   * @deprecated Use `spanToJSON(span).atttributes` instead.
   */
   get attributes() {
    return this._attributes;
  }

  /**
   * Attributes for the span.
   * @deprecated Use `setAttributes()` instead.
   */
   set attributes(attributes) {
    this._attributes = attributes;
  }

  /**
   * Timestamp in seconds (epoch time) indicating when the span started.
   * @deprecated Use `spanToJSON()` instead.
   */
   get startTimestamp() {
    return this._startTime;
  }

  /**
   * Timestamp in seconds (epoch time) indicating when the span started.
   * @deprecated In v8, you will not be able to update the span start time after creation.
   */
   set startTimestamp(startTime) {
    this._startTime = startTime;
  }

  /**
   * Timestamp in seconds when the span ended.
   * @deprecated Use `spanToJSON()` instead.
   */
   get endTimestamp() {
    return this._endTime;
  }

  /**
   * Timestamp in seconds when the span ended.
   * @deprecated Set the end time via `span.end()` instead.
   */
   set endTimestamp(endTime) {
    this._endTime = endTime;
  }

  /**
   * The status of the span.
   *
   * @deprecated Use `spanToJSON().status` instead to get the status.
   */
   get status() {
    return this._status;
  }

  /**
   * The status of the span.
   *
   * @deprecated Use `.setStatus()` instead to set or update the status.
   */
   set status(status) {
    this._status = status;
  }

  /**
   * Operation of the span
   *
   * @deprecated Use `spanToJSON().op` to read the op instead.
   */
   get op() {
    return this._attributes[SEMANTIC_ATTRIBUTE_SENTRY_OP] ;
  }

  /**
   * Operation of the span
   *
   * @deprecated Use `startSpan()` functions to set or `span.setAttribute(SEMANTIC_ATTRIBUTE_SENTRY_OP, 'op')
   *             to update the span instead.
   */
   set op(op) {
    this.setAttribute(SEMANTIC_ATTRIBUTE_SENTRY_OP, op);
  }

  /**
   * The origin of the span, giving context about what created the span.
   *
   * @deprecated Use `spanToJSON().origin` to read the origin instead.
   */
   get origin() {
    return this._attributes[SEMANTIC_ATTRIBUTE_SENTRY_ORIGIN] ;
  }

  /**
   * The origin of the span, giving context about what created the span.
   *
   * @deprecated Use `startSpan()` functions to set the origin instead.
   */
   set origin(origin) {
    this.setAttribute(SEMANTIC_ATTRIBUTE_SENTRY_ORIGIN, origin);
  }

  /* eslint-enable @typescript-eslint/member-ordering */

  /** @inheritdoc */
   spanContext() {
    const { _spanId: spanId, _traceId: traceId, _sampled: sampled } = this;
    return {
      spanId,
      traceId,
      traceFlags: sampled ? TRACE_FLAG_SAMPLED : TRACE_FLAG_NONE,
    };
  }

  /**
   * Creates a new `Span` while setting the current `Span.id` as `parentSpanId`.
   * Also the `sampled` decision will be inherited.
   *
   * @deprecated Use `startSpan()`, `startSpanManual()` or `startInactiveSpan()` instead.
   */
   startChild(
    spanContext,
  ) {
    const childSpan = new Span({
      ...spanContext,
      parentSpanId: this._spanId,
      sampled: this._sampled,
      traceId: this._traceId,
    });

    // eslint-disable-next-line deprecation/deprecation
    childSpan.spanRecorder = this.spanRecorder;
    // eslint-disable-next-line deprecation/deprecation
    if (childSpan.spanRecorder) {
      // eslint-disable-next-line deprecation/deprecation
      childSpan.spanRecorder.add(childSpan);
    }

    const rootSpan = getRootSpan(this);
    // TODO: still set span.transaction here until we have a more permanent solution
    // Probably similarly to the weakmap we hold in node-experimental
    // eslint-disable-next-line deprecation/deprecation
    childSpan.transaction = rootSpan ;

    if (DEBUG_BUILD && rootSpan) {
      const opStr = (spanContext && spanContext.op) || '< unknown op >';
      const nameStr = spanToJSON(childSpan).description || '< unknown name >';
      const idStr = rootSpan.spanContext().spanId;

      const logMessage = `[Tracing] Starting '${opStr}' span on transaction '${nameStr}' (${idStr}).`;
      logger.log(logMessage);
      this._logMessage = logMessage;
    }

    return childSpan;
  }

  /**
   * Sets the tag attribute on the current span.
   *
   * Can also be used to unset a tag, by passing `undefined`.
   *
   * @param key Tag key
   * @param value Tag value
   * @deprecated Use `setAttribute()` instead.
   */
   setTag(key, value) {
    // eslint-disable-next-line deprecation/deprecation
    this.tags = { ...this.tags, [key]: value };
    return this;
  }

  /**
   * Sets the data attribute on the current span
   * @param key Data key
   * @param value Data value
   * @deprecated Use `setAttribute()` instead.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
   setData(key, value) {
    // eslint-disable-next-line deprecation/deprecation
    this.data = { ...this.data, [key]: value };
    return this;
  }

  /** @inheritdoc */
   setAttribute(key, value) {
    if (value === undefined) {
      // eslint-disable-next-line @typescript-eslint/no-dynamic-delete
      delete this._attributes[key];
    } else {
      this._attributes[key] = value;
    }
  }

  /** @inheritdoc */
   setAttributes(attributes) {
    Object.keys(attributes).forEach(key => this.setAttribute(key, attributes[key]));
  }

  /**
   * @inheritDoc
   */
   setStatus(value) {
    this._status = value;
    return this;
  }

  /**
   * @inheritDoc
   * @deprecated Use top-level `setHttpStatus()` instead.
   */
   setHttpStatus(httpStatus) {
    setHttpStatus(this, httpStatus);
    return this;
  }

  /**
   * @inheritdoc
   *
   * @deprecated Use `.updateName()` instead.
   */
   setName(name) {
    this.updateName(name);
  }

  /**
   * @inheritDoc
   */
   updateName(name) {
    this._name = name;
    return this;
  }

  /**
   * @inheritDoc
   *
   * @deprecated Use `spanToJSON(span).status === 'ok'` instead.
   */
   isSuccess() {
    return this._status === 'ok';
  }

  /**
   * @inheritDoc
   *
   * @deprecated Use `.end()` instead.
   */
   finish(endTimestamp) {
    return this.end(endTimestamp);
  }

  /** @inheritdoc */
   end(endTimestamp) {
    // If already ended, skip
    if (this._endTime) {
      return;
    }
    const rootSpan = getRootSpan(this);
    if (
      DEBUG_BUILD &&
      // Don't call this for transactions
      rootSpan &&
      rootSpan.spanContext().spanId !== this._spanId
    ) {
      const logMessage = this._logMessage;
      if (logMessage) {
        logger.log((logMessage ).replace('Starting', 'Finishing'));
      }
    }

    this._endTime = spanTimeInputToSeconds(endTimestamp);
  }

  /**
   * @inheritDoc
   *
   * @deprecated Use `spanToTraceHeader()` instead.
   */
   toTraceparent() {
    return spanToTraceHeader(this);
  }

  /**
   * @inheritDoc
   *
   * @deprecated Use `spanToJSON()` or access the fields directly instead.
   */
   toContext() {
    return dropUndefinedKeys({
      data: this._getData(),
      description: this._name,
      endTimestamp: this._endTime,
      // eslint-disable-next-line deprecation/deprecation
      op: this.op,
      parentSpanId: this._parentSpanId,
      sampled: this._sampled,
      spanId: this._spanId,
      startTimestamp: this._startTime,
      status: this._status,
      // eslint-disable-next-line deprecation/deprecation
      tags: this.tags,
      traceId: this._traceId,
    });
  }

  /**
   * @inheritDoc
   *
   * @deprecated Update the fields directly instead.
   */
   updateWithContext(spanContext) {
    // eslint-disable-next-line deprecation/deprecation
    this.data = spanContext.data || {};
    // eslint-disable-next-line deprecation/deprecation
    this._name = spanContext.name || spanContext.description;
    this._endTime = spanContext.endTimestamp;
    // eslint-disable-next-line deprecation/deprecation
    this.op = spanContext.op;
    this._parentSpanId = spanContext.parentSpanId;
    this._sampled = spanContext.sampled;
    this._spanId = spanContext.spanId || this._spanId;
    this._startTime = spanContext.startTimestamp || this._startTime;
    this._status = spanContext.status;
    // eslint-disable-next-line deprecation/deprecation
    this.tags = spanContext.tags || {};
    this._traceId = spanContext.traceId || this._traceId;

    return this;
  }

  /**
   * @inheritDoc
   *
   * @deprecated Use `spanToTraceContext()` util function instead.
   */
   getTraceContext() {
    return spanToTraceContext(this);
  }

  /**
   * Get JSON representation of this span.
   *
   * @hidden
   * @internal This method is purely for internal purposes and should not be used outside
   * of SDK code. If you need to get a JSON representation of a span,
   * use `spanToJSON(span)` instead.
   */
   getSpanJSON() {
    return dropUndefinedKeys({
      data: this._getData(),
      description: this._name,
      op: this._attributes[SEMANTIC_ATTRIBUTE_SENTRY_OP] ,
      parent_span_id: this._parentSpanId,
      span_id: this._spanId,
      start_timestamp: this._startTime,
      status: this._status,
      // eslint-disable-next-line deprecation/deprecation
      tags: Object.keys(this.tags).length > 0 ? this.tags : undefined,
      timestamp: this._endTime,
      trace_id: this._traceId,
      origin: this._attributes[SEMANTIC_ATTRIBUTE_SENTRY_ORIGIN] ,
      _metrics_summary: getMetricSummaryJsonForSpan(this),
      profile_id: this._attributes[SEMANTIC_ATTRIBUTE_PROFILE_ID] ,
      exclusive_time: this._exclusiveTime,
      measurements: Object.keys(this._measurements).length > 0 ? this._measurements : undefined,
    });
  }

  /** @inheritdoc */
   isRecording() {
    return !this._endTime && !!this._sampled;
  }

  /**
   * Convert the object to JSON.
   * @deprecated Use `spanToJSON(span)` instead.
   */
   toJSON() {
    return this.getSpanJSON();
  }

  /**
   * Get the merged data for this span.
   * For now, this combines `data` and `attributes` together,
   * until eventually we can ingest `attributes` directly.
   */
   _getData()

 {
    // eslint-disable-next-line deprecation/deprecation
    const { data, _attributes: attributes } = this;

    const hasData = Object.keys(data).length > 0;
    const hasAttributes = Object.keys(attributes).length > 0;

    if (!hasData && !hasAttributes) {
      return undefined;
    }

    if (hasData && hasAttributes) {
      return {
        ...data,
        ...attributes,
      };
    }

    return hasData ? data : attributes;
  }
}

export { Span, SpanRecorder };
//# sourceMappingURL=span.js.map
