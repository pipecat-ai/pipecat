import type { Breadcrumb, BreadcrumbHint, Client, CustomSamplingContext, Event, EventHint, Extra, Extras, Hub as HubInterface, Integration, IntegrationClass, Primitive, Session, SessionContext, Severity, SeverityLevel, Transaction, TransactionContext, User } from '@sentry/types';
import { Scope } from './scope';
/**
 * API compatibility version of this hub.
 *
 * WARNING: This number should only be increased when the global interface
 * changes and new methods are introduced.
 *
 * @hidden
 */
export declare const API_VERSION: number;
export interface RunWithAsyncContextOptions {
    /** Whether to reuse an existing async context if one exists. Defaults to false. */
    reuseExisting?: boolean;
}
/**
 * @private Private API with no semver guarantees!
 *
 * Strategy used to track async context.
 */
export interface AsyncContextStrategy {
    /**
     * Gets the current async context. Returns undefined if there is no current async context.
     */
    getCurrentHub: () => Hub | undefined;
    /**
     * Runs the supplied callback in its own async context.
     */
    runWithAsyncContext<T>(callback: () => T, options: RunWithAsyncContextOptions): T;
}
/**
 * A layer in the process stack.
 * @hidden
 */
export interface Layer {
    client?: Client;
    scope: Scope;
}
/**
 * An object that contains a hub and maintains a scope stack.
 * @hidden
 */
export interface Carrier {
    __SENTRY__?: {
        hub?: Hub;
        acs?: AsyncContextStrategy;
        /**
         * Extra Hub properties injected by various SDKs
         */
        integrations?: Integration[];
        extensions?: {
            /** Extension methods for the hub, which are bound to the current Hub instance */
            [key: string]: Function;
        };
    };
}
/**
 * @deprecated The `Hub` class will be removed in version 8 of the SDK in favour of `Scope` and `Client` objects.
 *
 * If you previously used the `Hub` class directly, replace it with `Scope` and `Client` objects. More information:
 * - [Multiple Sentry Instances](https://docs.sentry.io/platforms/javascript/best-practices/multiple-sentry-instances/)
 * - [Browser Extensions](https://docs.sentry.io/platforms/javascript/best-practices/browser-extensions/)
 *
 * Some of our APIs are typed with the Hub class instead of the interface (e.g. `getCurrentHub`). Most of them are deprecated
 * themselves and will also be removed in version 8. More information:
 * - [Migration Guide](https://github.com/getsentry/sentry-javascript/blob/develop/MIGRATION.md#deprecate-hub)
 */
export declare class Hub implements HubInterface {
    private readonly _version;
    /** Is a {@link Layer}[] containing the client and scope */
    private readonly _stack;
    /** Contains the last event id of a captured event.  */
    private _lastEventId?;
    private _isolationScope;
    /**
     * Creates a new instance of the hub, will push one {@link Layer} into the
     * internal stack on creation.
     *
     * @param client bound to the hub.
     * @param scope bound to the hub.
     * @param version number, higher number means higher priority.
     *
     * @deprecated Instantiation of Hub objects is deprecated and the constructor will be removed in version 8 of the SDK.
     *
     * If you are currently using the Hub for multi-client use like so:
     *
     * ```
     * // OLD
     * const hub = new Hub();
     * hub.bindClient(client);
     * makeMain(hub)
     * ```
     *
     * instead initialize the client as follows:
     *
     * ```
     * // NEW
     * Sentry.withIsolationScope(() => {
     *    Sentry.setCurrentClient(client);
     *    client.init();
     * });
     * ```
     *
     * If you are using the Hub to capture events like so:
     *
     * ```
     * // OLD
     * const client = new Client();
     * const hub = new Hub(client);
     * hub.captureException()
     * ```
     *
     * instead capture isolated events as follows:
     *
     * ```
     * // NEW
     * const client = new Client();
     * const scope = new Scope();
     * scope.setClient(client);
     * scope.captureException();
     * ```
     */
    constructor(client?: Client, scope?: Scope, isolationScope?: Scope, _version?: number);
    /**
     * Checks if this hub's version is older than the given version.
     *
     * @param version A version number to compare to.
     * @return True if the given version is newer; otherwise false.
     *
     * @deprecated This will be removed in v8.
     */
    isOlderThan(version: number): boolean;
    /**
     * This binds the given client to the current scope.
     * @param client An SDK client (client) instance.
     *
     * @deprecated Use `initAndBind()` directly, or `setCurrentClient()` and/or `client.init()` instead.
     */
    bindClient(client?: Client): void;
    /**
     * @inheritDoc
     *
     * @deprecated Use `withScope` instead.
     */
    pushScope(): Scope;
    /**
     * @inheritDoc
     *
     * @deprecated Use `withScope` instead.
     */
    popScope(): boolean;
    /**
     * @inheritDoc
     *
     * @deprecated Use `Sentry.withScope()` instead.
     */
    withScope<T>(callback: (scope: Scope) => T): T;
    /**
     * @inheritDoc
     *
     * @deprecated Use `Sentry.getClient()` instead.
     */
    getClient<C extends Client>(): C | undefined;
    /**
     * Returns the scope of the top stack.
     *
     * @deprecated Use `Sentry.getCurrentScope()` instead.
     */
    getScope(): Scope;
    /**
     * @deprecated Use `Sentry.getIsolationScope()` instead.
     */
    getIsolationScope(): Scope;
    /**
     * Returns the scope stack for domains or the process.
     * @deprecated This will be removed in v8.
     */
    getStack(): Layer[];
    /**
     * Returns the topmost scope layer in the order domain > local > process.
     * @deprecated This will be removed in v8.
     */
    getStackTop(): Layer;
    /**
     * @inheritDoc
     *
     * @deprecated Use `Sentry.captureException()` instead.
     */
    captureException(exception: unknown, hint?: EventHint): string;
    /**
     * @inheritDoc
     *
     * @deprecated Use  `Sentry.captureMessage()` instead.
     */
    captureMessage(message: string, level?: Severity | SeverityLevel, hint?: EventHint): string;
    /**
     * @inheritDoc
     *
     * @deprecated Use `Sentry.captureEvent()` instead.
     */
    captureEvent(event: Event, hint?: EventHint): string;
    /**
     * @inheritDoc
     *
     * @deprecated This will be removed in v8.
     */
    lastEventId(): string | undefined;
    /**
     * @inheritDoc
     *
     * @deprecated Use `Sentry.addBreadcrumb()` instead.
     */
    addBreadcrumb(breadcrumb: Breadcrumb, hint?: BreadcrumbHint): void;
    /**
     * @inheritDoc
     * @deprecated Use `Sentry.setUser()` instead.
     */
    setUser(user: User | null): void;
    /**
     * @inheritDoc
     * @deprecated Use `Sentry.setTags()` instead.
     */
    setTags(tags: {
        [key: string]: Primitive;
    }): void;
    /**
     * @inheritDoc
     * @deprecated Use `Sentry.setExtras()` instead.
     */
    setExtras(extras: Extras): void;
    /**
     * @inheritDoc
     * @deprecated Use `Sentry.setTag()` instead.
     */
    setTag(key: string, value: Primitive): void;
    /**
     * @inheritDoc
     * @deprecated Use `Sentry.setExtra()` instead.
     */
    setExtra(key: string, extra: Extra): void;
    /**
     * @inheritDoc
     * @deprecated Use `Sentry.setContext()` instead.
     */
    setContext(name: string, context: {
        [key: string]: any;
    } | null): void;
    /**
     * @inheritDoc
     *
     * @deprecated Use `getScope()` directly.
     */
    configureScope(callback: (scope: Scope) => void): void;
    /**
     * @inheritDoc
     */
    run(callback: (hub: Hub) => void): void;
    /**
     * @inheritDoc
     * @deprecated Use `Sentry.getClient().getIntegrationByName()` instead.
     */
    getIntegration<T extends Integration>(integration: IntegrationClass<T>): T | null;
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
     * @param context Properties of the new `Transaction`.
     * @param customSamplingContext Information given to the transaction sampling function (along with context-dependent
     * default values). See {@link Options.tracesSampler}.
     *
     * @returns The transaction which was just started
     *
     * @deprecated Use `startSpan()`, `startSpanManual()` or `startInactiveSpan()` instead.
     */
    startTransaction(context: TransactionContext, customSamplingContext?: CustomSamplingContext): Transaction;
    /**
     * @inheritDoc
     * @deprecated Use `spanToTraceHeader()` instead.
     */
    traceHeaders(): {
        [key: string]: string;
    };
    /**
     * @inheritDoc
     *
     * @deprecated Use top level `captureSession` instead.
     */
    captureSession(endSession?: boolean): void;
    /**
     * @inheritDoc
     * @deprecated Use top level `endSession` instead.
     */
    endSession(): void;
    /**
     * @inheritDoc
     * @deprecated Use top level `startSession` instead.
     */
    startSession(context?: SessionContext): Session;
    /**
     * Returns if default PII should be sent to Sentry and propagated in ourgoing requests
     * when Tracing is used.
     *
     * @deprecated Use top-level `getClient().getOptions().sendDefaultPii` instead. This function
     * only unnecessarily increased API surface but only wrapped accessing the option.
     */
    shouldSendDefaultPii(): boolean;
    /**
     * Sends the current Session on the scope
     */
    private _sendSessionUpdate;
    /**
     * Calls global extension method and binding current instance to the function call
     */
    private _callExtensionMethod;
}
/**
 * Returns the global shim registry.
 *
 * FIXME: This function is problematic, because despite always returning a valid Carrier,
 * it has an optional `__SENTRY__` property, which then in turn requires us to always perform an unnecessary check
 * at the call-site. We always access the carrier through this function, so we can guarantee that `__SENTRY__` is there.
 **/
export declare function getMainCarrier(): Carrier;
/**
 * Replaces the current main hub with the passed one on the global object
 *
 * @returns The old replaced hub
 *
 * @deprecated Use `setCurrentClient()` instead.
 */
export declare function makeMain(hub: Hub): Hub;
/**
 * Returns the default hub instance.
 *
 * If a hub is already registered in the global carrier but this module
 * contains a more recent version, it replaces the registered version.
 * Otherwise, the currently registered hub will be returned.
 *
 * @deprecated Use the respective replacement method directly instead.
 */
export declare function getCurrentHub(): Hub;
/**
 * Get the currently active isolation scope.
 * The isolation scope is active for the current exection context,
 * meaning that it will remain stable for the same Hub.
 */
export declare function getIsolationScope(): Scope;
/**
 * @private Private API with no semver guarantees!
 *
 * If the carrier does not contain a hub, a new hub is created with the global hub client and scope.
 */
export declare function ensureHubOnCarrier(carrier: Carrier, parent?: Hub): void;
/**
 * @private Private API with no semver guarantees!
 *
 * Sets the global async context strategy
 */
export declare function setAsyncContextStrategy(strategy: AsyncContextStrategy | undefined): void;
/**
 * Runs the supplied callback in its own async context. Async Context strategies are defined per SDK.
 *
 * @param callback The callback to run in its own async context
 * @param options Options to pass to the async context strategy
 * @returns The result of the callback
 */
export declare function runWithAsyncContext<T>(callback: () => T, options?: RunWithAsyncContextOptions): T;
/**
 * This will create a new {@link Hub} and add to the passed object on
 * __SENTRY__.hub.
 * @param carrier object
 * @hidden
 */
export declare function getHubFromCarrier(carrier: Carrier): Hub;
/**
 * This will set passed {@link Hub} on the passed object's __SENTRY__.hub attribute
 * @param carrier object
 * @param hub Hub
 * @returns A boolean indicating success or failure
 */
export declare function setHubOnCarrier(carrier: Carrier, hub: Hub): boolean;
//# sourceMappingURL=hub.d.ts.map