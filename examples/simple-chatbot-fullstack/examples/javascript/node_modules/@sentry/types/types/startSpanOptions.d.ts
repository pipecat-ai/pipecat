import type { Instrumenter } from './instrumenter';
import type { Primitive } from './misc';
import type { Scope } from './scope';
import type { SpanAttributes, SpanOrigin, SpanTimeInput } from './span';
import type { TransactionContext, TransactionMetadata, TransactionSource } from './transaction';
export interface StartSpanOptions extends TransactionContext {
    /** A manually specified start time for the created `Span` object. */
    startTime?: SpanTimeInput;
    /** If defined, start this span off this scope instead off the current scope. */
    scope?: Scope;
    /** The name of the span. */
    name: string;
    /** If set to true, only start a span if a parent span exists. */
    onlyIfParent?: boolean;
    /** An op for the span. This is a categorization for spans. */
    op?: string;
    /**
     * If set to true, this span will be forced to be treated as a transaction in the Sentry UI, if possible and applicable.
     * Note that it is up to the SDK to decide how exactly the span will be sent, which may change in future SDK versions.
     * It is not guaranteed that a span started with this flag set to `true` will be sent as a transaction.
     */
    forceTransaction?: boolean;
    /**
     * The origin of the span - if it comes from auto instrumentation or manual instrumentation.
     *
     * @deprecated Set `attributes[SEMANTIC_ATTRIBUTE_SENTRY_ORIGIN]` instead.
     */
    origin?: SpanOrigin;
    /** Attributes for the span. */
    attributes?: SpanAttributes;
    /**
     * @deprecated Manually set the end timestamp instead.
     */
    trimEnd?: boolean;
    /**
     * @deprecated This cannot be set manually anymore.
     */
    parentSampled?: boolean;
    /**
     * @deprecated Use attributes or set data on scopes instead.
     */
    metadata?: Partial<TransactionMetadata>;
    /**
     * The name thingy.
     * @deprecated Use `name` instead.
     */
    description?: string;
    /**
     * @deprecated Use `span.setStatus()` instead.
     */
    status?: string;
    /**
     * @deprecated Use `scope` instead.
     */
    parentSpanId?: string;
    /**
     * @deprecated You cannot manually set the span to sampled anymore.
     */
    sampled?: boolean;
    /**
     * @deprecated You cannot manually set the spanId anymore.
     */
    spanId?: string;
    /**
     * @deprecated You cannot manually set the traceId anymore.
     */
    traceId?: string;
    /**
     * @deprecated Use an attribute instead.
     */
    source?: TransactionSource;
    /**
     * @deprecated Use attributes or set tags on the scope instead.
     */
    tags?: {
        [key: string]: Primitive;
    };
    /**
     * @deprecated Use attributes instead.
     */
    data?: {
        [key: string]: any;
    };
    /**
     * @deprecated Use `startTime` instead.
     */
    startTimestamp?: number;
    /**
     * @deprecated Use `span.end()` instead.
     */
    endTimestamp?: number;
    /**
     * @deprecated You cannot set the instrumenter manually anymore.
     */
    instrumenter?: Instrumenter;
}
//# sourceMappingURL=startSpanOptions.d.ts.map