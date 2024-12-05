import type { SpanTimeInput, TransactionContext } from '@sentry/types';
import type { Hub } from '../hub';
import type { Span } from './span';
import { SpanRecorder } from './span';
import { Transaction } from './transaction';
export declare const TRACING_DEFAULTS: {
    idleTimeout: number;
    finalTimeout: number;
    heartbeatInterval: number;
};
/**
 * @inheritDoc
 */
export declare class IdleTransactionSpanRecorder extends SpanRecorder {
    private readonly _pushActivity;
    private readonly _popActivity;
    transactionSpanId: string;
    constructor(_pushActivity: (id: string) => void, _popActivity: (id: string) => void, transactionSpanId: string, maxlen?: number);
    /**
     * @inheritDoc
     */
    add(span: Span): void;
}
export type BeforeFinishCallback = (transactionSpan: IdleTransaction, endTimestamp: number) => void;
/**
 * An IdleTransaction is a transaction that automatically finishes. It does this by tracking child spans as activities.
 * You can have multiple IdleTransactions active, but if the `onScope` option is specified, the idle transaction will
 * put itself on the scope on creation.
 */
export declare class IdleTransaction extends Transaction {
    private readonly _idleHub;
    /**
     * The time to wait in ms until the idle transaction will be finished. This timer is started each time
     * there are no active spans on this transaction.
     */
    private readonly _idleTimeout;
    /**
     * The final value in ms that a transaction cannot exceed
     */
    private readonly _finalTimeout;
    private readonly _heartbeatInterval;
    private readonly _onScope;
    activities: Record<string, boolean>;
    private _prevHeartbeatString;
    private _heartbeatCounter;
    private _finished;
    private _idleTimeoutCanceledPermanently;
    private readonly _beforeFinishCallbacks;
    /**
     * Timer that tracks Transaction idleTimeout
     */
    private _idleTimeoutID;
    private _finishReason;
    private _autoFinishAllowed;
    /**
     * @deprecated Transactions will be removed in v8. Use spans instead.
     */
    constructor(transactionContext: TransactionContext, _idleHub: Hub, 
    /**
     * The time to wait in ms until the idle transaction will be finished. This timer is started each time
     * there are no active spans on this transaction.
     */
    _idleTimeout?: number, 
    /**
     * The final value in ms that a transaction cannot exceed
     */
    _finalTimeout?: number, _heartbeatInterval?: number, _onScope?: boolean, 
    /**
     * When set to `true`, will disable the idle timeout (`_idleTimeout` option) and heartbeat mechanisms (`_heartbeatInterval`
     * option) until the `sendAutoFinishSignal()` method is called. The final timeout mechanism (`_finalTimeout` option)
     * will not be affected by this option, meaning the transaction will definitely be finished when the final timeout is
     * reached, no matter what this option is configured to.
     *
     * Defaults to `false`.
     */
    delayAutoFinishUntilSignal?: boolean);
    /** {@inheritDoc} */
    end(endTimestamp?: SpanTimeInput): string | undefined;
    /**
     * Register a callback function that gets executed before the transaction finishes.
     * Useful for cleanup or if you want to add any additional spans based on current context.
     *
     * This is exposed because users have no other way of running something before an idle transaction
     * finishes.
     */
    registerBeforeFinishCallback(callback: BeforeFinishCallback): void;
    /**
     * @inheritDoc
     */
    initSpanRecorder(maxlen?: number): void;
    /**
     * Cancels the existing idle timeout, if there is one.
     * @param restartOnChildSpanChange Default is `true`.
     *                                 If set to false the transaction will end
     *                                 with the last child span.
     */
    cancelIdleTimeout(endTimestamp?: Parameters<IdleTransaction['end']>[0], { restartOnChildSpanChange, }?: {
        restartOnChildSpanChange?: boolean;
    }): void;
    /**
     * Temporary method used to externally set the transaction's `finishReason`
     *
     * ** WARNING**
     * This is for the purpose of experimentation only and will be removed in the near future, do not use!
     *
     * @internal
     *
     */
    setFinishReason(reason: string): void;
    /**
     * Permits the IdleTransaction to automatically end itself via the idle timeout and heartbeat mechanisms when the `delayAutoFinishUntilSignal` option was set to `true`.
     */
    sendAutoFinishSignal(): void;
    /**
     * Restarts idle timeout, if there is no running idle timeout it will start one.
     */
    private _restartIdleTimeout;
    /**
     * Start tracking a specific activity.
     * @param spanId The span id that represents the activity
     */
    private _pushActivity;
    /**
     * Remove an activity from usage
     * @param spanId The span id that represents the activity
     */
    private _popActivity;
    /**
     * Checks when entries of this.activities are not changing for 3 beats.
     * If this occurs we finish the transaction.
     */
    private _beat;
    /**
     * Pings the heartbeat
     */
    private _pingHeartbeat;
}
//# sourceMappingURL=idletransaction.d.ts.map