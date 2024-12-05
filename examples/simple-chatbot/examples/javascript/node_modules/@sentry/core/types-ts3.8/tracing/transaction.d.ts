import { Context, DynamicSamplingContext, MeasurementUnit, SpanTimeInput, Transaction as TransactionInterface, TransactionContext, TransactionEvent, TransactionMetadata } from '@sentry/types';
import { Hub } from '../hub';
import { Span as SpanClass } from './span';
/** JSDoc */
export declare class Transaction extends SpanClass implements TransactionInterface {
    /**
     * The reference to the current hub.
     */
    _hub: Hub;
    protected _name: string;
    private _contexts;
    private _trimEnd?;
    private _frozenDynamicSamplingContext;
    private _metadata;
    /**
     * This constructor should never be called manually. Those instrumenting tracing should use
     * `Sentry.startTransaction()`, and internal methods should use `hub.startTransaction()`.
     * @internal
     * @hideconstructor
     * @hidden
     *
     * @deprecated Transactions will be removed in v8. Use spans instead.
     */
    constructor(transactionContext: TransactionContext, hub?: Hub);
    /*
    * Getter for `name` property.
    * @deprecated Use `spanToJSON(span).description` instead.
    
    
    * Setter for `name` property, which also sets `source` as custom.
    * @deprecated Use `updateName()` and `setMetadata()` instead.
    */
    name: string;
    /*
    * Get the metadata for this transaction.
    * @deprecated Use `spanGetMetadata(transaction)` instead.
    
    
    * Update the metadata for this transaction.
    * @deprecated Use `spanGetMetadata(transaction)` instead.
    */
    metadata: TransactionMetadata;
    /**
     * Setter for `name` property, which also sets `source` on the metadata.
     *
     * @deprecated Use `.updateName()` and `.setAttribute()` instead.
     */
    setName(name: string, source?: TransactionMetadata['source']): void;
    /** @inheritdoc */
    updateName(name: string): this;
    /**
     * Attaches SpanRecorder to the span itself
     * @param maxlen maximum number of spans that can be recorded
     */
    initSpanRecorder(maxlen?: number): void;
    /**
     * Set the context of a transaction event.
     * @deprecated Use either `.setAttribute()`, or set the context on the scope before creating the transaction.
     */
    setContext(key: string, context: Context | null): void;
    /**
     * @inheritDoc
     *
     * @deprecated Use top-level `setMeasurement()` instead.
     */
    setMeasurement(name: string, value: number, unit?: MeasurementUnit): void;
    /**
     * Store metadata on this transaction.
     * @deprecated Use attributes or store data on the scope instead.
     */
    setMetadata(newMetadata: Partial<TransactionMetadata>): void;
    /**
     * @inheritDoc
     */
    end(endTimestamp?: SpanTimeInput): string | undefined;
    /**
     * @inheritDoc
     */
    toContext(): TransactionContext;
    /**
     * @inheritDoc
     */
    updateWithContext(transactionContext: TransactionContext): this;
    /**
     * @inheritdoc
     *
     * @experimental
     *
     * @deprecated Use top-level `getDynamicSamplingContextFromSpan` instead.
     */
    getDynamicSamplingContext(): Readonly<Partial<DynamicSamplingContext>>;
    /**
     * Override the current hub with a new one.
     * Used if you want another hub to finish the transaction.
     *
     * @internal
     */
    setHub(hub: Hub): void;
    /**
     * Get the profile id of the transaction.
     */
    getProfileId(): string | undefined;
    /**
     * Finish the transaction & prepare the event to send to Sentry.
     */
    protected _finishTransaction(endTimestamp?: number): TransactionEvent | undefined;
}
//# sourceMappingURL=transaction.d.ts.map
