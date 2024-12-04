import type { Instrumenter, Measurements, Primitive, Span as SpanInterface, SpanAttributeValue, SpanAttributes, SpanContext, SpanContextData, SpanJSON, SpanOrigin, SpanTimeInput, TraceContext, Transaction } from '@sentry/types';
import type { SpanStatusType } from './spanstatus';
/**
 * Keeps track of finished spans for a given transaction
 * @internal
 * @hideconstructor
 * @hidden
 */
export declare class SpanRecorder {
    spans: Span[];
    private readonly _maxlen;
    constructor(maxlen?: number);
    /**
     * This is just so that we don't run out of memory while recording a lot
     * of spans. At some point we just stop and flush out the start of the
     * trace tree (i.e.the first n spans with the smallest
     * start_timestamp).
     */
    add(span: Span): void;
}
/**
 * Span contains all data about a span
 */
export declare class Span implements SpanInterface {
    /**
     * Tags for the span.
     * @deprecated Use `spanToJSON(span).atttributes` instead.
     */
    tags: {
        [key: string]: Primitive;
    };
    /**
     * Data for the span.
     * @deprecated Use `spanToJSON(span).atttributes` instead.
     */
    data: {
        [key: string]: any;
    };
    /**
     * List of spans that were finalized
     *
     * @deprecated This property will no longer be public. Span recording will be handled internally.
     */
    spanRecorder?: SpanRecorder;
    /**
     * @inheritDoc
     * @deprecated Use top level `Sentry.getRootSpan()` instead
     */
    transaction?: Transaction;
    /**
     * The instrumenter that created this span.
     *
     * TODO (v8): This can probably be replaced by an `instanceOf` check of the span class.
     *            the instrumenter can only be sentry or otel so we can check the span instance
     *            to verify which one it is and remove this field entirely.
     *
     * @deprecated This field will be removed.
     */
    instrumenter: Instrumenter;
    protected _traceId: string;
    protected _spanId: string;
    protected _parentSpanId?: string | undefined;
    protected _sampled: boolean | undefined;
    protected _name?: string | undefined;
    protected _attributes: SpanAttributes;
    /** Epoch timestamp in seconds when the span started. */
    protected _startTime: number;
    /** Epoch timestamp in seconds when the span ended. */
    protected _endTime?: number | undefined;
    /** Internal keeper of the status */
    protected _status?: SpanStatusType | string | undefined;
    protected _exclusiveTime?: number;
    protected _measurements: Measurements;
    private _logMessage?;
    /**
     * You should never call the constructor manually, always use `Sentry.startTransaction()`
     * or call `startChild()` on an existing span.
     * @internal
     * @hideconstructor
     * @hidden
     */
    constructor(spanContext?: SpanContext);
    /**
     * An alias for `description` of the Span.
     * @deprecated Use `spanToJSON(span).description` instead.
     */
    get name(): string;
    /**
     * Update the name of the span.
     * @deprecated Use `spanToJSON(span).description` instead.
     */
    set name(name: string);
    /**
     * Get the description of the Span.
     * @deprecated Use `spanToJSON(span).description` instead.
     */
    get description(): string | undefined;
    /**
     * Get the description of the Span.
     * @deprecated Use `spanToJSON(span).description` instead.
     */
    set description(description: string | undefined);
    /**
     * The ID of the trace.
     * @deprecated Use `spanContext().traceId` instead.
     */
    get traceId(): string;
    /**
     * The ID of the trace.
     * @deprecated You cannot update the traceId of a span after span creation.
     */
    set traceId(traceId: string);
    /**
     * The ID of the span.
     * @deprecated Use `spanContext().spanId` instead.
     */
    get spanId(): string;
    /**
     * The ID of the span.
     * @deprecated You cannot update the spanId of a span after span creation.
     */
    set spanId(spanId: string);
    /**
     * @inheritDoc
     *
     * @deprecated Use `startSpan` functions instead.
     */
    set parentSpanId(string: string | undefined);
    /**
     * @inheritDoc
     *
     * @deprecated Use `spanToJSON(span).parent_span_id` instead.
     */
    get parentSpanId(): string | undefined;
    /**
     * Was this span chosen to be sent as part of the sample?
     * @deprecated Use `isRecording()` instead.
     */
    get sampled(): boolean | undefined;
    /**
     * Was this span chosen to be sent as part of the sample?
     * @deprecated You cannot update the sampling decision of a span after span creation.
     */
    set sampled(sampled: boolean | undefined);
    /**
     * Attributes for the span.
     * @deprecated Use `spanToJSON(span).atttributes` instead.
     */
    get attributes(): SpanAttributes;
    /**
     * Attributes for the span.
     * @deprecated Use `setAttributes()` instead.
     */
    set attributes(attributes: SpanAttributes);
    /**
     * Timestamp in seconds (epoch time) indicating when the span started.
     * @deprecated Use `spanToJSON()` instead.
     */
    get startTimestamp(): number;
    /**
     * Timestamp in seconds (epoch time) indicating when the span started.
     * @deprecated In v8, you will not be able to update the span start time after creation.
     */
    set startTimestamp(startTime: number);
    /**
     * Timestamp in seconds when the span ended.
     * @deprecated Use `spanToJSON()` instead.
     */
    get endTimestamp(): number | undefined;
    /**
     * Timestamp in seconds when the span ended.
     * @deprecated Set the end time via `span.end()` instead.
     */
    set endTimestamp(endTime: number | undefined);
    /**
     * The status of the span.
     *
     * @deprecated Use `spanToJSON().status` instead to get the status.
     */
    get status(): SpanStatusType | string | undefined;
    /**
     * The status of the span.
     *
     * @deprecated Use `.setStatus()` instead to set or update the status.
     */
    set status(status: SpanStatusType | string | undefined);
    /**
     * Operation of the span
     *
     * @deprecated Use `spanToJSON().op` to read the op instead.
     */
    get op(): string | undefined;
    /**
     * Operation of the span
     *
     * @deprecated Use `startSpan()` functions to set or `span.setAttribute(SEMANTIC_ATTRIBUTE_SENTRY_OP, 'op')
     *             to update the span instead.
     */
    set op(op: string | undefined);
    /**
     * The origin of the span, giving context about what created the span.
     *
     * @deprecated Use `spanToJSON().origin` to read the origin instead.
     */
    get origin(): SpanOrigin | undefined;
    /**
     * The origin of the span, giving context about what created the span.
     *
     * @deprecated Use `startSpan()` functions to set the origin instead.
     */
    set origin(origin: SpanOrigin | undefined);
    /** @inheritdoc */
    spanContext(): SpanContextData;
    /**
     * Creates a new `Span` while setting the current `Span.id` as `parentSpanId`.
     * Also the `sampled` decision will be inherited.
     *
     * @deprecated Use `startSpan()`, `startSpanManual()` or `startInactiveSpan()` instead.
     */
    startChild(spanContext?: Pick<SpanContext, Exclude<keyof SpanContext, 'sampled' | 'traceId' | 'parentSpanId'>>): SpanInterface;
    /**
     * Sets the tag attribute on the current span.
     *
     * Can also be used to unset a tag, by passing `undefined`.
     *
     * @param key Tag key
     * @param value Tag value
     * @deprecated Use `setAttribute()` instead.
     */
    setTag(key: string, value: Primitive): this;
    /**
     * Sets the data attribute on the current span
     * @param key Data key
     * @param value Data value
     * @deprecated Use `setAttribute()` instead.
     */
    setData(key: string, value: any): this;
    /** @inheritdoc */
    setAttribute(key: string, value: SpanAttributeValue | undefined): void;
    /** @inheritdoc */
    setAttributes(attributes: SpanAttributes): void;
    /**
     * @inheritDoc
     */
    setStatus(value: SpanStatusType): this;
    /**
     * @inheritDoc
     * @deprecated Use top-level `setHttpStatus()` instead.
     */
    setHttpStatus(httpStatus: number): this;
    /**
     * @inheritdoc
     *
     * @deprecated Use `.updateName()` instead.
     */
    setName(name: string): void;
    /**
     * @inheritDoc
     */
    updateName(name: string): this;
    /**
     * @inheritDoc
     *
     * @deprecated Use `spanToJSON(span).status === 'ok'` instead.
     */
    isSuccess(): boolean;
    /**
     * @inheritDoc
     *
     * @deprecated Use `.end()` instead.
     */
    finish(endTimestamp?: number): void;
    /** @inheritdoc */
    end(endTimestamp?: SpanTimeInput): void;
    /**
     * @inheritDoc
     *
     * @deprecated Use `spanToTraceHeader()` instead.
     */
    toTraceparent(): string;
    /**
     * @inheritDoc
     *
     * @deprecated Use `spanToJSON()` or access the fields directly instead.
     */
    toContext(): SpanContext;
    /**
     * @inheritDoc
     *
     * @deprecated Update the fields directly instead.
     */
    updateWithContext(spanContext: SpanContext): this;
    /**
     * @inheritDoc
     *
     * @deprecated Use `spanToTraceContext()` util function instead.
     */
    getTraceContext(): TraceContext;
    /**
     * Get JSON representation of this span.
     *
     * @hidden
     * @internal This method is purely for internal purposes and should not be used outside
     * of SDK code. If you need to get a JSON representation of a span,
     * use `spanToJSON(span)` instead.
     */
    getSpanJSON(): SpanJSON;
    /** @inheritdoc */
    isRecording(): boolean;
    /**
     * Convert the object to JSON.
     * @deprecated Use `spanToJSON(span)` instead.
     */
    toJSON(): SpanJSON;
    /**
     * Get the merged data for this span.
     * For now, this combines `data` and `attributes` together,
     * until eventually we can ingest `attributes` directly.
     */
    private _getData;
}
//# sourceMappingURL=span.d.ts.map