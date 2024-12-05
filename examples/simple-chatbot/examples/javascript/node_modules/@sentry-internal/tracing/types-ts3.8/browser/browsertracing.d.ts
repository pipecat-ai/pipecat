import { Hub } from '@sentry/core';
import { EventProcessor, Integration, Transaction, TransactionContext } from '@sentry/types';
import { RequestInstrumentationOptions } from './request';
export declare const BROWSER_TRACING_INTEGRATION_ID = "BrowserTracing";
/** Options for Browser Tracing integration */
export interface BrowserTracingOptions extends RequestInstrumentationOptions {
    /**
     * The time to wait in ms until the transaction will be finished during an idle state. An idle state is defined
     * by a moment where there are no in-progress spans.
     *
     * The transaction will use the end timestamp of the last finished span as the endtime for the transaction.
     * If there are still active spans when this the `idleTimeout` is set, the `idleTimeout` will get reset.
     * Time is in ms.
     *
     * Default: 1000
     */
    idleTimeout: number;
    /**
     * The max duration for a transaction. If a transaction duration hits the `finalTimeout` value, it
     * will be finished.
     * Time is in ms.
     *
     * Default: 30000
     */
    finalTimeout: number;
    /**
     * The heartbeat interval. If no new spans are started or open spans are finished within 3 heartbeats,
     * the transaction will be finished.
     * Time is in ms.
     *
     * Default: 5000
     */
    heartbeatInterval: number;
    /**
     * Flag to enable/disable creation of `navigation` transaction on history changes.
     *
     * Default: true
     */
    startTransactionOnLocationChange: boolean;
    /**
     * Flag to enable/disable creation of `pageload` transaction on first pageload.
     *
     * Default: true
     */
    startTransactionOnPageLoad: boolean;
    /**
     * Flag Transactions where tabs moved to background with "cancelled". Browser background tab timing is
     * not suited towards doing precise measurements of operations. By default, we recommend that this option
     * be enabled as background transactions can mess up your statistics in nondeterministic ways.
     *
     * Default: true
     */
    markBackgroundTransactions: boolean;
    /**
     * If true, Sentry will capture long tasks and add them to the corresponding transaction.
     *
     * Default: true
     */
    enableLongTask: boolean;
    /**
     * If true, Sentry will capture INP web vitals as standalone spans .
     *
     * Default: false
     */
    enableInp: boolean;
    /**
     * Sample rate to determine interaction span sampling.
     * interactionsSampleRate is applied on top of the global tracesSampleRate.
     * ie a tracesSampleRate of 0.1 and interactionsSampleRate of 0.5 will result in a 0.05 sample rate for interactions.
     *
     * Default: 1
     */
    interactionsSampleRate: number;
    /**
     * _metricOptions allows the user to send options to change how metrics are collected.
     *
     * _metricOptions is currently experimental.
     *
     * Default: undefined
     */
    _metricOptions?: Partial<{
        /**
         * @deprecated This property no longer has any effect and will be removed in v8.
         */
        _reportAllChanges: boolean;
    }>;
    /**
     * _experiments allows the user to send options to define how this integration works.
     * Note that the `enableLongTask` options is deprecated in favor of the option at the top level, and will be removed in v8.
     *
     * TODO (v8): Remove enableLongTask
     *
     * Default: undefined
     */
    _experiments: Partial<{
        enableLongTask: boolean;
        enableInteractions: boolean;
        onStartRouteTransaction: (t: Transaction | undefined, ctx: TransactionContext, getCurrentHub: () => Hub) => void;
    }>;
    /**
     * beforeNavigate is called before a pageload/navigation transaction is created and allows users to modify transaction
     * context data, or drop the transaction entirely (by setting `sampled = false` in the context).
     *
     * Note: For legacy reasons, transactions can also be dropped by returning `undefined`.
     *
     * @param context: The context data which will be passed to `startTransaction` by default
     *
     * @returns A (potentially) modified context object, with `sampled = false` if the transaction should be dropped.
     */
    beforeNavigate?(this: void, context: TransactionContext): TransactionContext | undefined;
    /**
     * Instrumentation that creates routing change transactions. By default creates
     * pageload and navigation transactions.
     */
    routingInstrumentation<T extends Transaction>(this: void, customStartTransaction: (context: TransactionContext) => T | undefined, startTransactionOnPageLoad?: boolean, startTransactionOnLocationChange?: boolean): void;
}
/**
 * The Browser Tracing integration automatically instruments browser pageload/navigation
 * actions as transactions, and captures requests, metrics and errors as spans.
 *
 * The integration can be configured with a variety of options, and can be extended to use
 * any routing library. This integration uses {@see IdleTransaction} to create transactions.
 *
 * @deprecated Use `browserTracingIntegration()` instead.
 */
export declare class BrowserTracing implements Integration {
    /** Browser Tracing integration options */
    options: BrowserTracingOptions;
    /**
     * @inheritDoc
     */
    name: string;
    private _getCurrentHub?;
    private _collectWebVitals;
    private _hasSetTracePropagationTargets;
    private _interactionIdToRouteNameMapping;
    private _latestRoute;
    constructor(_options?: Partial<BrowserTracingOptions>);
    /**
     * @inheritDoc
     */
    setupOnce(_: (callback: EventProcessor) => void, getCurrentHub: () => Hub): void;
    /** Create routing idle transaction. */
    private _createRouteTransaction;
    /** Start listener for interaction transactions */
    private _registerInteractionListener;
    /** Creates a listener on interaction entries, and maps interactionIds to the origin path of the interaction */
    private _registerInpInteractionListener;
}
/** Returns the value of a meta tag */
export declare function getMetaContent(metaName: string): string | undefined;
//# sourceMappingURL=browsertracing.d.ts.map
