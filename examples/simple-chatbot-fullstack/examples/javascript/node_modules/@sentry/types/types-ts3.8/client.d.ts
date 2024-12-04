import { Breadcrumb, BreadcrumbHint } from './breadcrumb';
import { CheckIn, MonitorConfig } from './checkin';
import { EventDropReason } from './clientreport';
import { DataCategory } from './datacategory';
import { DsnComponents } from './dsn';
import { DynamicSamplingContext, Envelope } from './envelope';
import { Event, EventHint } from './event';
import { EventProcessor } from './eventprocessor';
import { FeedbackEvent } from './feedback';
import { Integration, IntegrationClass } from './integration';
import { MetricBucketItem } from './metrics';
import { ClientOptions } from './options';
import { ParameterizedString } from './parameterize';
import { Scope } from './scope';
import { SdkMetadata } from './sdkmetadata';
import { Session, SessionAggregates } from './session';
import { Severity, SeverityLevel } from './severity';
import { StartSpanOptions } from './startSpanOptions';
import { Transaction } from './transaction';
import { Transport, TransportMakeRequestResponse } from './transport';
/**
 * User-Facing Sentry SDK Client.
 *
 * This interface contains all methods to interface with the SDK once it has
 * been installed. It allows to send events to Sentry, record breadcrumbs and
 * set a context included in every event. Since the SDK mutates its environment,
 * there will only be one instance during runtime.
 *
 */
export interface Client<O extends ClientOptions = ClientOptions> {
    /**
     * Captures an exception event and sends it to Sentry.
     *
     * @param exception An exception-like object.
     * @param hint May contain additional information about the original exception.
     * @param scope An optional scope containing event metadata.
     * @returns The event id
     */
    captureException(exception: any, hint?: EventHint, scope?: Scope): string | undefined;
    /**
     * Captures a message event and sends it to Sentry.
     *
     * @param message The message to send to Sentry.
     * @param level Define the level of the message.
     * @param hint May contain additional information about the original exception.
     * @param scope An optional scope containing event metadata.
     * @returns The event id
     */
    captureMessage(message: string, level?: Severity | SeverityLevel, hint?: EventHint, scope?: Scope): string | undefined;
    /**
     * Captures a manually created event and sends it to Sentry.
     *
     * @param event The event to send to Sentry.
     * @param hint May contain additional information about the original exception.
     * @param scope An optional scope containing event metadata.
     * @returns The event id
     */
    captureEvent(event: Event, hint?: EventHint, scope?: Scope): string | undefined;
    /**
     * Captures a session
     *
     * @param session Session to be delivered
     */
    captureSession?(session: Session): void;
    /**
     * Create a cron monitor check in and send it to Sentry. This method is not available on all clients.
     *
     * @param checkIn An object that describes a check in.
     * @param upsertMonitorConfig An optional object that describes a monitor config. Use this if you want
     * to create a monitor automatically when sending a check in.
     * @param scope An optional scope containing event metadata.
     * @returns A string representing the id of the check in.
     */
    captureCheckIn?(checkIn: CheckIn, monitorConfig?: MonitorConfig, scope?: Scope): string;
    /** Returns the current Dsn. */
    getDsn(): DsnComponents | undefined;
    /** Returns the current options. */
    getOptions(): O;
    /**
     * @inheritdoc
     *
     * TODO (v8): Make this a required method.
     */
    getSdkMetadata?(): SdkMetadata | undefined;
    /**
     * Returns the transport that is used by the client.
     * Please note that the transport gets lazy initialized so it will only be there once the first event has been sent.
     *
     * @returns The transport.
     */
    getTransport(): Transport | undefined;
    /**
     * Flush the event queue and set the client to `enabled = false`. See {@link Client.flush}.
     *
     * @param timeout Maximum time in ms the client should wait before shutting down. Omitting this parameter will cause
     *   the client to wait until all events are sent before disabling itself.
     * @returns A promise which resolves to `true` if the flush completes successfully before the timeout, or `false` if
     * it doesn't.
     */
    close(timeout?: number): PromiseLike<boolean>;
    /**
     * Wait for all events to be sent or the timeout to expire, whichever comes first.
     *
     * @param timeout Maximum time in ms the client should wait for events to be flushed. Omitting this parameter will
     *   cause the client to wait until all events are sent before resolving the promise.
     * @returns A promise that will resolve with `true` if all events are sent before the timeout, or `false` if there are
     * still events in the queue when the timeout is reached.
     */
    flush(timeout?: number): PromiseLike<boolean>;
    /**
     * Adds an event processor that applies to any event processed by this client.
     *
     * TODO (v8): Make this a required method.
     */
    addEventProcessor?(eventProcessor: EventProcessor): void;
    /**
     * Get all added event processors for this client.
     *
     * TODO (v8): Make this a required method.
     */
    getEventProcessors?(): EventProcessor[];
    /**
     * Returns the client's instance of the given integration class, it any.
     * @deprecated Use `getIntegrationByName()` instead.
     */
    getIntegration<T extends Integration>(integration: IntegrationClass<T>): T | null;
    /** Get the instance of the integration with the given name on the client, if it was added. */
    getIntegrationByName?<T extends Integration = Integration>(name: string): T | undefined;
    /**
     * Add an integration to the client.
     * This can be used to e.g. lazy load integrations.
     * In most cases, this should not be necessary, and you're better off just passing the integrations via `integrations: []` at initialization time.
     * However, if you find the need to conditionally load & add an integration, you can use `addIntegration` to do so.
     *
     * TODO (v8): Make this a required method.
     * */
    addIntegration?(integration: Integration): void;
    /**
     * This is an internal function to setup all integrations that should run on the client.
     * @deprecated Use `client.init()` instead.
     */
    setupIntegrations(forceInitialize?: boolean): void;
    /**
     * Initialize this client.
     * Call this after the client was set on a scope.
     */
    init?(): void;
    /** Creates an {@link Event} from all inputs to `captureException` and non-primitive inputs to `captureMessage`. */
    eventFromException(exception: any, hint?: EventHint): PromiseLike<Event>;
    /** Creates an {@link Event} from primitive inputs to `captureMessage`. */
    eventFromMessage(message: ParameterizedString, level?: Severity | SeverityLevel, hint?: EventHint): PromiseLike<Event>;
    /** Submits the event to Sentry */
    sendEvent(event: Event, hint?: EventHint): void;
    /** Submits the session to Sentry */
    sendSession(session: Session | SessionAggregates): void;
    /**
     * Record on the client that an event got dropped (ie, an event that will not be sent to sentry).
     *
     * @param reason The reason why the event got dropped.
     * @param category The data category of the dropped event.
     * @param event The dropped event.
     */
    recordDroppedEvent(reason: EventDropReason, dataCategory: DataCategory, event?: Event): void;
    /**
     * Captures serialized metrics and sends them to Sentry.
     *
     * @experimental This API is experimental and might experience breaking changes
     */
    captureAggregateMetrics?(metricBucketItems: Array<MetricBucketItem>): void;
    /**
     * Register a callback for transaction start.
     * Receives the transaction as argument.
     */
    on?(hook: 'startTransaction', callback: (transaction: Transaction) => void): void;
    /**
     * Register a callback for transaction finish.
     * Receives the transaction as argument.
     */
    on?(hook: 'finishTransaction', callback: (transaction: Transaction) => void): void;
    /**
     * Register a callback for transaction start and finish.
     */
    on?(hook: 'beforeEnvelope', callback: (envelope: Envelope) => void): void;
    /**
     * Register a callback for before sending an event.
     * This is called right before an event is sent and should not be used to mutate the event.
     * Receives an Event & EventHint as arguments.
     */
    on?(hook: 'beforeSendEvent', callback: (event: Event, hint?: EventHint | undefined) => void): void;
    /**
     * Register a callback for preprocessing an event,
     * before it is passed to (global) event processors.
     * Receives an Event & EventHint as arguments.
     */
    on?(hook: 'preprocessEvent', callback: (event: Event, hint?: EventHint | undefined) => void): void;
    /**
     * Register a callback for when an event has been sent.
     */
    on?(hook: 'afterSendEvent', callback: (event: Event, sendResponse: TransportMakeRequestResponse | void) => void): void;
    /**
     * Register a callback before a breadcrumb is added.
     */
    on?(hook: 'beforeAddBreadcrumb', callback: (breadcrumb: Breadcrumb, hint?: BreadcrumbHint) => void): void;
    /**
     * Register a callback when a DSC (Dynamic Sampling Context) is created.
     */
    on?(hook: 'createDsc', callback: (dsc: DynamicSamplingContext) => void): void;
    /**
     * Register a callback when an OpenTelemetry span is ended (in @sentry/opentelemetry-node).
     * The option argument may be mutated to drop the span.
     */
    on?(hook: 'otelSpanEnd', callback: (otelSpan: unknown, mutableOptions: {
        drop: boolean;
    }) => void): void;
    /**
     * Register a callback when a Feedback event has been prepared.
     * This should be used to mutate the event. The options argument can hint
     * about what kind of mutation it expects.
     */
    on?(hook: 'beforeSendFeedback', callback: (feedback: FeedbackEvent, options?: {
        includeReplay?: boolean;
    }) => void): void;
    /**
     * A hook for BrowserTracing to trigger a span start for a page load.
     */
    on?(hook: 'startPageLoadSpan', callback: (options: StartSpanOptions) => void): void;
    /**
     * A hook for BrowserTracing to trigger a span for a navigation.
     */
    on?(hook: 'startNavigationSpan', callback: (options: StartSpanOptions) => void): void;
    /**
     * Fire a hook event for transaction start.
     * Expects to be given a transaction as the second argument.
     */
    emit?(hook: 'startTransaction', transaction: Transaction): void;
    /**
     * Fire a hook event for transaction finish.
     * Expects to be given a transaction as the second argument.
     */
    emit?(hook: 'finishTransaction', transaction: Transaction): void;
    emit?(hook: 'beforeEnvelope', envelope: Envelope): void;
    /**
     * Fire a hook event before sending an event.
     * This is called right before an event is sent and should not be used to mutate the event.
     * Expects to be given an Event & EventHint as the second/third argument.
     */
    emit?(hook: 'beforeSendEvent', event: Event, hint?: EventHint): void;
    /**
     * Fire a hook event to process events before they are passed to (global) event processors.
     * Expects to be given an Event & EventHint as the second/third argument.
     */
    emit?(hook: 'preprocessEvent', event: Event, hint?: EventHint): void;
    emit?(hook: 'afterSendEvent', event: Event, sendResponse: TransportMakeRequestResponse | void): void;
    /**
     * Fire a hook for when a breadcrumb is added. Expects the breadcrumb as second argument.
     */
    emit?(hook: 'beforeAddBreadcrumb', breadcrumb: Breadcrumb, hint?: BreadcrumbHint): void;
    /**
     * Fire a hook for when a DSC (Dynamic Sampling Context) is created. Expects the DSC as second argument.
     */
    emit?(hook: 'createDsc', dsc: DynamicSamplingContext): void;
    /**
     * Fire a hook for when an OpenTelemetry span is ended (in @sentry/opentelemetry-node).
     * Expects the OTEL span & as second argument, and an option object as third argument.
     * The option argument may be mutated to drop the span.
     */
    emit?(hook: 'otelSpanEnd', otelSpan: unknown, mutableOptions: {
        drop: boolean;
    }): void;
    /**
     * Fire a hook event for after preparing a feedback event. Events to be given
     * a feedback event as the second argument, and an optional options object as
     * third argument.
     */
    emit?(hook: 'beforeSendFeedback', feedback: FeedbackEvent, options?: {
        includeReplay?: boolean;
    }): void;
    /**
     * Emit a hook event for BrowserTracing to trigger a span start for a page load.
     */
    emit?(hook: 'startPageLoadSpan', options: StartSpanOptions): void;
    /**
     * Emit a hook event for BrowserTracing to trigger a span for a navigation.
     */
    emit?(hook: 'startNavigationSpan', options: StartSpanOptions): void;
}
//# sourceMappingURL=client.d.ts.map
