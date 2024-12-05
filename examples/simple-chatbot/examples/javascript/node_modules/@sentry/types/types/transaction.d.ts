import type { Context } from './context';
import type { DynamicSamplingContext } from './envelope';
import type { Instrumenter } from './instrumenter';
import type { MeasurementUnit } from './measurement';
import type { ExtractedNodeRequestData, Primitive, WorkerLocation } from './misc';
import type { PolymorphicRequest } from './polymorphics';
import type { Span, SpanAttributes, SpanContext } from './span';
/**
 * Interface holding Transaction-specific properties
 */
export interface TransactionContext extends SpanContext {
    /**
     * Human-readable identifier for the transaction
     */
    name: string;
    /**
     * If true, sets the end timestamp of the transaction to the highest timestamp of child spans, trimming
     * the duration of the transaction. This is useful to discard extra time in the transaction that is not
     * accounted for in child spans, like what happens in the idle transaction Tracing integration, where we finish the
     * transaction after a given "idle time" and we don't want this "idle time" to be part of the transaction.
     */
    trimEnd?: boolean | undefined;
    /**
     * If this transaction has a parent, the parent's sampling decision
     */
    parentSampled?: boolean | undefined;
    /**
     * Metadata associated with the transaction, for internal SDK use.
     * @deprecated Use attributes or store data on the scope instead.
     */
    metadata?: Partial<TransactionMetadata>;
}
/**
 * Data pulled from a `sentry-trace` header
 */
export type TraceparentData = Pick<TransactionContext, 'traceId' | 'parentSpanId' | 'parentSampled'>;
/**
 * Transaction "Class", inherits Span only has `setName`
 */
export interface Transaction extends TransactionContext, Omit<Span, 'setName' | 'name'> {
    /**
     * Human-readable identifier for the transaction.
     * @deprecated Use `spanToJSON(span).description` instead.
     */
    name: string;
    /**
     * The ID of the transaction.
     * @deprecated Use `spanContext().spanId` instead.
     */
    spanId: string;
    /**
     * The ID of the trace.
     * @deprecated Use `spanContext().traceId` instead.
     */
    traceId: string;
    /**
     * Was this transaction chosen to be sent as part of the sample?
     * @deprecated Use `spanIsSampled(transaction)` instead.
     */
    sampled?: boolean | undefined;
    /**
     * @inheritDoc
     */
    startTimestamp: number;
    /**
     * Tags for the transaction.
     * @deprecated Use `getSpanAttributes(transaction)` instead.
     */
    tags: {
        [key: string]: Primitive;
    };
    /**
     * Data for the transaction.
     * @deprecated Use `getSpanAttributes(transaction)` instead.
     */
    data: {
        [key: string]: any;
    };
    /**
     * Attributes for the transaction.
     * @deprecated Use `getSpanAttributes(transaction)` instead.
     */
    attributes: SpanAttributes;
    /**
     * Metadata about the transaction.
     * @deprecated Use attributes or store data on the scope instead.
     */
    metadata: TransactionMetadata;
    /**
     * The instrumenter that created this transaction.
     *
     * @deprecated This field will be removed in v8.
     */
    instrumenter: Instrumenter;
    /**
     * Set the name of the transaction
     *
     * @deprecated Use `.updateName()` and `.setAttribute()` instead.
     */
    setName(name: string, source?: TransactionMetadata['source']): void;
    /**
     * Set the context of a transaction event.
     * @deprecated Use either `.setAttribute()`, or set the context on the scope before creating the transaction.
     */
    setContext(key: string, context: Context): void;
    /**
     * Set observed measurement for this transaction.
     *
     * @param name Name of the measurement
     * @param value Value of the measurement
     * @param unit Unit of the measurement. (Defaults to an empty string)
     *
     * @deprecated Use top-level `setMeasurement()` instead.
     */
    setMeasurement(name: string, value: number, unit: MeasurementUnit): void;
    /**
     * Returns the current transaction properties as a `TransactionContext`.
     * @deprecated Use `toJSON()` or access the fields directly instead.
     */
    toContext(): TransactionContext;
    /**
     * Updates the current transaction with a new `TransactionContext`.
     * @deprecated Update the fields directly instead.
     */
    updateWithContext(transactionContext: TransactionContext): this;
    /**
     * Set metadata for this transaction.
     * @deprecated Use attributes or store data on the scope instead.
     */
    setMetadata(newMetadata: Partial<TransactionMetadata>): void;
    /**
     * Return the current Dynamic Sampling Context of this transaction
     *
     * @deprecated Use top-level `getDynamicSamplingContextFromSpan` instead.
     */
    getDynamicSamplingContext(): Partial<DynamicSamplingContext>;
    /**
     * Get the profile id from the transaction
     * @deprecated Use `toJSON()` or access the fields directly instead.
     */
    getProfileId(): string | undefined;
}
/**
 * Context data passed by the user when starting a transaction, to be used by the tracesSampler method.
 */
export interface CustomSamplingContext {
    [key: string]: any;
}
/**
 * Data passed to the `tracesSampler` function, which forms the basis for whatever decisions it might make.
 *
 * Adds default data to data provided by the user. See {@link Hub.startTransaction}
 */
export interface SamplingContext extends CustomSamplingContext {
    /**
     * Context data with which transaction being sampled was created
     */
    transactionContext: TransactionContext;
    /**
     * Sampling decision from the parent transaction, if any.
     */
    parentSampled?: boolean;
    /**
     * Object representing the URL of the current page or worker script. Passed by default when using the `BrowserTracing`
     * integration.
     */
    location?: WorkerLocation;
    /**
     * Object representing the incoming request to a node server. Passed by default when using the TracingHandler.
     */
    request?: ExtractedNodeRequestData;
}
export interface TransactionMetadata {
    /**
     * The sample rate used when sampling this transaction.
     * @deprecated Use `SEMANTIC_ATTRIBUTE_SENTRY_SAMPLE_RATE` attribute instead.
     */
    sampleRate?: number;
    /**
     * The Dynamic Sampling Context of a transaction. If provided during transaction creation, its Dynamic Sampling
     * Context Will be frozen
     */
    dynamicSamplingContext?: Partial<DynamicSamplingContext>;
    /** For transactions tracing server-side request handling, the request being tracked. */
    request?: PolymorphicRequest;
    /** Compatibility shim for transitioning to the `RequestData` integration. The options passed to our Express request
     * handler controlling what request data is added to the event.
     * TODO (v8): This should go away
     */
    requestDataOptionsFromExpressHandler?: {
        [key: string]: unknown;
    };
    /** For transactions tracing server-side request handling, the path of the request being tracked. */
    /** TODO: If we rm -rf `instrumentServer`, this can go, too */
    requestPath?: string;
    /**
     * Information on how a transaction name was generated.
     * @deprecated Use `SEMANTIC_ATTRIBUTE_SENTRY_SOURCE` attribute instead.
     */
    source: TransactionSource;
    /**
     * Metadata for the transaction's spans, keyed by spanId.
     * @deprecated This will be removed in v8.
     */
    spanMetadata: {
        [spanId: string]: {
            [key: string]: unknown;
        };
    };
}
/**
 * Contains information about how the name of the transaction was determined. This will be used by the server to decide
 * whether or not to scrub identifiers from the transaction name, or replace the entire name with a placeholder.
 */
export type TransactionSource = 
/** User-defined name */
'custom'
/** Raw URL, potentially containing identifiers */
 | 'url'
/** Parametrized URL / route */
 | 'route'
/** Name of the view handling the request */
 | 'view'
/** Named after a software component, such as a function or class name. */
 | 'component'
/** Name of a background task (e.g. a Celery task) */
 | 'task';
//# sourceMappingURL=transaction.d.ts.map