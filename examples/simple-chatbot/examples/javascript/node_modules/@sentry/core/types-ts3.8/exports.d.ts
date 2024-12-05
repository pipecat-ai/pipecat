import { Breadcrumb, BreadcrumbHint, CaptureContext, CheckIn, Client, CustomSamplingContext, Event, EventHint, Extra, Extras, MonitorConfig, Primitive, Scope as ScopeInterface, Session, SessionContext, Severity, SeverityLevel, Span, TransactionContext, User } from '@sentry/types';
import { Hub } from './hub';
import { Scope } from './scope';
import { ExclusiveEventHintOrCaptureContext } from './utils/prepareEvent';
/**
 * Captures an exception event and sends it to Sentry.
 *
 * @param exception The exception to capture.
 * @param hint Optional additional data to attach to the Sentry event.
 * @returns the id of the captured Sentry event.
 */
export declare function captureException(exception: any, hint?: ExclusiveEventHintOrCaptureContext): string;
/**
 * Captures a message event and sends it to Sentry.
 *
 * @param exception The exception to capture.
 * @param captureContext Define the level of the message or pass in additional data to attach to the message.
 * @returns the id of the captured message.
 */
export declare function captureMessage(message: string, captureContext?: CaptureContext | Severity | SeverityLevel): string;
/**
 * Captures a manually created event and sends it to Sentry.
 *
 * @param exception The event to send to Sentry.
 * @param hint Optional additional data to attach to the Sentry event.
 * @returns the id of the captured event.
 */
export declare function captureEvent(event: Event, hint?: EventHint): string;
/**
 * Callback to set context information onto the scope.
 * @param callback Callback function that receives Scope.
 *
 * @deprecated Use getCurrentScope() directly.
 */
export declare function configureScope(callback: (scope: Scope) => void): ReturnType<Hub['configureScope']>;
/**
 * Records a new breadcrumb which will be attached to future events.
 *
 * Breadcrumbs will be added to subsequent events to provide more context on
 * user's actions prior to an error or crash.
 *
 * @param breadcrumb The breadcrumb to record.
 */
export declare function addBreadcrumb(breadcrumb: Breadcrumb, hint?: BreadcrumbHint): ReturnType<Hub['addBreadcrumb']>;
/**
 * Sets context data with the given name.
 * @param name of the context
 * @param context Any kind of data. This data will be normalized.
 */
export declare function setContext(name: string, context: {
    [key: string]: any;
} | null): ReturnType<Hub['setContext']>;
/**
 * Set an object that will be merged sent as extra data with the event.
 * @param extras Extras object to merge into current context.
 */
export declare function setExtras(extras: Extras): ReturnType<Hub['setExtras']>;
/**
 * Set key:value that will be sent as extra data with the event.
 * @param key String of extra
 * @param extra Any kind of data. This data will be normalized.
 */
export declare function setExtra(key: string, extra: Extra): ReturnType<Hub['setExtra']>;
/**
 * Set an object that will be merged sent as tags data with the event.
 * @param tags Tags context object to merge into current context.
 */
export declare function setTags(tags: {
    [key: string]: Primitive;
}): ReturnType<Hub['setTags']>;
/**
 * Set key:value that will be sent as tags data with the event.
 *
 * Can also be used to unset a tag, by passing `undefined`.
 *
 * @param key String key of tag
 * @param value Value of tag
 */
export declare function setTag(key: string, value: Primitive): ReturnType<Hub['setTag']>;
/**
 * Updates user context information for future events.
 *
 * @param user User context object to be set in the current context. Pass `null` to unset the user.
 */
export declare function setUser(user: User | null): ReturnType<Hub['setUser']>;
/**
 * Creates a new scope with and executes the given operation within.
 * The scope is automatically removed once the operation
 * finishes or throws.
 *
 * This is essentially a convenience function for:
 *
 *     pushScope();
 *     callback();
 *     popScope();
 */
export declare function withScope<T>(callback: (scope: Scope) => T): T;
/**
 * Set the given scope as the active scope in the callback.
 */
export declare function withScope<T>(scope: ScopeInterface | undefined, callback: (scope: Scope) => T): T;
/**
 * Attempts to fork the current isolation scope and the current scope based on the current async context strategy. If no
 * async context strategy is set, the isolation scope and the current scope will not be forked (this is currently the
 * case, for example, in the browser).
 *
 * Usage of this function in environments without async context strategy is discouraged and may lead to unexpected behaviour.
 *
 * This function is intended for Sentry SDK and SDK integration development. It is not recommended to be used in "normal"
 * applications directly because it comes with pitfalls. Use at your own risk!
 *
 * @param callback The callback in which the passed isolation scope is active. (Note: In environments without async
 * context strategy, the currently active isolation scope may change within execution of the callback.)
 * @returns The same value that `callback` returns.
 */
export declare function withIsolationScope<T>(callback: (isolationScope: Scope) => T): T;
/**
 * Forks the current scope and sets the provided span as active span in the context of the provided callback.
 *
 * @param span Spans started in the context of the provided callback will be children of this span.
 * @param callback Execution context in which the provided span will be active. Is passed the newly forked scope.
 * @returns the value returned from the provided callback function.
 */
export declare function withActiveSpan<T>(span: Span, callback: (scope: Scope) => T): T;
/**
 * Starts a new `Transaction` and returns it. This is the entry point to manual tracing instrumentation.
 *
 * A tree structure can be built by adding child spans to the transaction, and child spans to other spans. To start a
 * new child span within the transaction or any span, call the respective `.startChild()` method.
 *
 * Every child span must be finished before the transaction is finished, otherwise the unfinished spans are discarded.
 *
 * The transaction must be finished with a call to its `.end()` method, at which point the transaction with all its
 * finished child spans will be sent to Sentry.
 *
 * NOTE: This function should only be used for *manual* instrumentation. Auto-instrumentation should call
 * `startTransaction` directly on the hub.
 *
 * @param context Properties of the new `Transaction`.
 * @param customSamplingContext Information given to the transaction sampling function (along with context-dependent
 * default values). See {@link Options.tracesSampler}.
 *
 * @returns The transaction which was just started
 *
 * @deprecated Use `startSpan()`, `startSpanManual()` or `startInactiveSpan()` instead.
 */
export declare function startTransaction(context: TransactionContext, customSamplingContext?: CustomSamplingContext): ReturnType<Hub['startTransaction']>;
/**
 * Create a cron monitor check in and send it to Sentry.
 *
 * @param checkIn An object that describes a check in.
 * @param upsertMonitorConfig An optional object that describes a monitor config. Use this if you want
 * to create a monitor automatically when sending a check in.
 */
export declare function captureCheckIn(checkIn: CheckIn, upsertMonitorConfig?: MonitorConfig): string;
/**
 * Wraps a callback with a cron monitor check in. The check in will be sent to Sentry when the callback finishes.
 *
 * @param monitorSlug The distinct slug of the monitor.
 * @param upsertMonitorConfig An optional object that describes a monitor config. Use this if you want
 * to create a monitor automatically when sending a check in.
 */
export declare function withMonitor<T>(monitorSlug: CheckIn['monitorSlug'], callback: () => T, upsertMonitorConfig?: MonitorConfig): T;
/**
 * Call `flush()` on the current client, if there is one. See {@link Client.flush}.
 *
 * @param timeout Maximum time in ms the client should wait to flush its event queue. Omitting this parameter will cause
 * the client to wait until all events are sent before resolving the promise.
 * @returns A promise which resolves to `true` if the queue successfully drains before the timeout, or `false` if it
 * doesn't (or if there's no client defined).
 */
export declare function flush(timeout?: number): Promise<boolean>;
/**
 * Call `close()` on the current client, if there is one. See {@link Client.close}.
 *
 * @param timeout Maximum time in ms the client should wait to flush its event queue before shutting down. Omitting this
 * parameter will cause the client to wait until all events are sent before disabling itself.
 * @returns A promise which resolves to `true` if the queue successfully drains before the timeout, or `false` if it
 * doesn't (or if there's no client defined).
 */
export declare function close(timeout?: number): Promise<boolean>;
/**
 * This is the getter for lastEventId.
 *
 * @returns The last event id of a captured event.
 */
export declare function lastEventId(): string | undefined;
/**
 * Get the currently active client.
 */
export declare function getClient<C extends Client>(): C | undefined;
/**
 * Returns true if Sentry has been properly initialized.
 */
export declare function isInitialized(): boolean;
/**
 * Get the currently active scope.
 */
export declare function getCurrentScope(): Scope;
/**
 * Start a session on the current isolation scope.
 *
 * @param context (optional) additional properties to be applied to the returned session object
 *
 * @returns the new active session
 */
export declare function startSession(context?: SessionContext): Session;
/**
 * End the session on the current isolation scope.
 */
export declare function endSession(): void;
/**
 * Sends the current session on the scope to Sentry
 *
 * @param end If set the session will be marked as exited and removed from the scope.
 *            Defaults to `false`.
 */
export declare function captureSession(end?: boolean): void;
//# sourceMappingURL=exports.d.ts.map
