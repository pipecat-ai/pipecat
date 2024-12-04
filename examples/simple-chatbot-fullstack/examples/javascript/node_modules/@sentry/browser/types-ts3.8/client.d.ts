import { Scope } from '@sentry/core';
import { BaseClient } from '@sentry/core';
import { BrowserClientProfilingOptions, BrowserClientReplayOptions, ClientOptions, Event, EventHint, Options, ParameterizedString, Severity, SeverityLevel, UserFeedback } from '@sentry/types';
import { BrowserTransportOptions } from './transports/types';
/**
 * Configuration options for the Sentry Browser SDK.
 * @see @sentry/types Options for more information.
 */
export type BrowserOptions = Options<BrowserTransportOptions> & BrowserClientReplayOptions & BrowserClientProfilingOptions;
/**
 * Configuration options for the Sentry Browser SDK Client class
 * @see BrowserClient for more information.
 */
export type BrowserClientOptions = ClientOptions<BrowserTransportOptions> & BrowserClientReplayOptions & BrowserClientProfilingOptions;
/**
 * The Sentry Browser SDK Client.
 *
 * @see BrowserOptions for documentation on configuration options.
 * @see SentryClient for usage documentation.
 */
export declare class BrowserClient extends BaseClient<BrowserClientOptions> {
    /**
     * Creates a new Browser SDK instance.
     *
     * @param options Configuration options for this SDK.
     */
    constructor(options: BrowserClientOptions);
    /**
     * @inheritDoc
     */
    eventFromException(exception: unknown, hint?: EventHint): PromiseLike<Event>;
    /**
     * @inheritDoc
     */
    eventFromMessage(message: ParameterizedString, level?: Severity | SeverityLevel, hint?: EventHint): PromiseLike<Event>;
    /**
     * Sends user feedback to Sentry.
     */
    captureUserFeedback(feedback: UserFeedback): void;
    /**
     * @inheritDoc
     */
    protected _prepareEvent(event: Event, hint: EventHint, scope?: Scope): PromiseLike<Event | null>;
    /**
     * Sends client reports as an envelope.
     */
    private _flushOutcomes;
}
//# sourceMappingURL=client.d.ts.map
