import { Attachment, Breadcrumb, CaptureContext, Client, Context, Contexts, Event, EventHint, EventProcessor, Extra, Extras, Primitive, PropagationContext, RequestSession, Scope as ScopeInterface, ScopeData, Session, Severity, SeverityLevel, Span, Transaction, User } from '@sentry/types';
/**
 * Holds additional event information. {@link Scope.applyToEvent} will be
 * called by the client before an event will be sent.
 */
export declare class Scope implements ScopeInterface {
    /** Flag if notifying is happening. */
    protected _notifyingListeners: boolean;
    /** Callback for client to receive scope changes. */
    protected _scopeListeners: Array<(scope: Scope) => void>;
    /** Callback list that will be called after {@link applyToEvent}. */
    protected _eventProcessors: EventProcessor[];
    /** Array of breadcrumbs. */
    protected _breadcrumbs: Breadcrumb[];
    /** User */
    protected _user: User;
    /** Tags */
    protected _tags: {
        [key: string]: Primitive;
    };
    /** Extra */
    protected _extra: Extras;
    /** Contexts */
    protected _contexts: Contexts;
    /** Attachments */
    protected _attachments: Attachment[];
    /** Propagation Context for distributed tracing */
    protected _propagationContext: PropagationContext;
    /**
     * A place to stash data which is needed at some point in the SDK's event processing pipeline but which shouldn't get
     * sent to Sentry
     */
    protected _sdkProcessingMetadata: {
        [key: string]: unknown;
    };
    /** Fingerprint */
    protected _fingerprint?: string[];
    /** Severity */
    protected _level?: Severity | SeverityLevel;
    /**
     * Transaction Name
     */
    protected _transactionName?: string;
    /** Span */
    protected _span?: Span;
    /** Session */
    protected _session?: Session;
    /** Request Mode Session Status */
    protected _requestSession?: RequestSession;
    /** The client on this scope */
    protected _client?: Client;
    constructor();
    /**
     * Inherit values from the parent scope.
     * @deprecated Use `scope.clone()` and `new Scope()` instead.
     */
    static clone(scope?: Scope): Scope;
    /**
     * Clone this scope instance.
     */
    clone(): Scope;
    /** Update the client on the scope. */
    setClient(client: Client | undefined): void;
    /**
     * Get the client assigned to this scope.
     *
     * It is generally recommended to use the global function `Sentry.getClient()` instead, unless you know what you are doing.
     */
    getClient(): Client | undefined;
    /**
     * Add internal on change listener. Used for sub SDKs that need to store the scope.
     * @hidden
     */
    addScopeListener(callback: (scope: Scope) => void): void;
    /**
     * @inheritDoc
     */
    addEventProcessor(callback: EventProcessor): this;
    /**
     * @inheritDoc
     */
    setUser(user: User | null): this;
    /**
     * @inheritDoc
     */
    getUser(): User | undefined;
    /**
     * @inheritDoc
     */
    getRequestSession(): RequestSession | undefined;
    /**
     * @inheritDoc
     */
    setRequestSession(requestSession?: RequestSession): this;
    /**
     * @inheritDoc
     */
    setTags(tags: {
        [key: string]: Primitive;
    }): this;
    /**
     * @inheritDoc
     */
    setTag(key: string, value: Primitive): this;
    /**
     * @inheritDoc
     */
    setExtras(extras: Extras): this;
    /**
     * @inheritDoc
     */
    setExtra(key: string, extra: Extra): this;
    /**
     * @inheritDoc
     */
    setFingerprint(fingerprint: string[]): this;
    /**
     * @inheritDoc
     */
    setLevel(level: Severity | SeverityLevel): this;
    /**
     * Sets the transaction name on the scope for future events.
     */
    setTransactionName(name?: string): this;
    /**
     * @inheritDoc
     */
    setContext(key: string, context: Context | null): this;
    /**
     * Sets the Span on the scope.
     * @param span Span
     * @deprecated Instead of setting a span on a scope, use `startSpan()`/`startSpanManual()` instead.
     */
    setSpan(span?: Span): this;
    /**
     * Returns the `Span` if there is one.
     * @deprecated Use `getActiveSpan()` instead.
     */
    getSpan(): Span | undefined;
    /**
     * Returns the `Transaction` attached to the scope (if there is one).
     * @deprecated You should not rely on the transaction, but just use `startSpan()` APIs instead.
     */
    getTransaction(): Transaction | undefined;
    /**
     * @inheritDoc
     */
    setSession(session?: Session): this;
    /**
     * @inheritDoc
     */
    getSession(): Session | undefined;
    /**
     * @inheritDoc
     */
    update(captureContext?: CaptureContext): this;
    /**
     * @inheritDoc
     */
    clear(): this;
    /**
     * @inheritDoc
     */
    addBreadcrumb(breadcrumb: Breadcrumb, maxBreadcrumbs?: number): this;
    /**
     * @inheritDoc
     */
    getLastBreadcrumb(): Breadcrumb | undefined;
    /**
     * @inheritDoc
     */
    clearBreadcrumbs(): this;
    /**
     * @inheritDoc
     */
    addAttachment(attachment: Attachment): this;
    /**
     * @inheritDoc
     * @deprecated Use `getScopeData()` instead.
     */
    getAttachments(): Attachment[];
    /**
     * @inheritDoc
     */
    clearAttachments(): this;
    /** @inheritDoc */
    getScopeData(): ScopeData;
    /**
     * Applies data from the scope to the event and runs all event processors on it.
     *
     * @param event Event
     * @param hint Object containing additional information about the original exception, for use by the event processors.
     * @hidden
     * @deprecated Use `applyScopeDataToEvent()` directly
     */
    applyToEvent(event: Event, hint?: EventHint, additionalEventProcessors?: EventProcessor[]): PromiseLike<Event | null>;
    /**
     * Add data which will be accessible during event processing but won't get sent to Sentry
     */
    setSDKProcessingMetadata(newData: {
        [key: string]: unknown;
    }): this;
    /**
     * @inheritDoc
     */
    setPropagationContext(context: PropagationContext): this;
    /**
     * @inheritDoc
     */
    getPropagationContext(): PropagationContext;
    /**
     * Capture an exception for this scope.
     *
     * @param exception The exception to capture.
     * @param hint Optinal additional data to attach to the Sentry event.
     * @returns the id of the captured Sentry event.
     */
    captureException(exception: unknown, hint?: EventHint): string;
    /**
     * Capture a message for this scope.
     *
     * @param message The message to capture.
     * @param level An optional severity level to report the message with.
     * @param hint Optional additional data to attach to the Sentry event.
     * @returns the id of the captured message.
     */
    captureMessage(message: string, level?: SeverityLevel, hint?: EventHint): string;
    /**
     * Captures a manually created event for this scope and sends it to Sentry.
     *
     * @param exception The event to capture.
     * @param hint Optional additional data to attach to the Sentry event.
     * @returns the id of the captured event.
     */
    captureEvent(event: Event, hint?: EventHint): string;
    /**
     * This will be called on every set call.
     */
    protected _notifyScopeListeners(): void;
}
/**
 * Get the global scope.
 * This scope is applied to _all_ events.
 */
export declare function getGlobalScope(): ScopeInterface;
/**
 * This is mainly needed for tests.
 * DO NOT USE this, as this is an internal API and subject to change.
 * @hidden
 */
export declare function setGlobalScope(scope: ScopeInterface | undefined): void;
//# sourceMappingURL=scope.d.ts.map
