import type { CustomSamplingContext, TransactionContext } from '@sentry/types';
import type { Hub } from '../hub';
import { IdleTransaction } from './idletransaction';
/**
 * Create new idle transaction.
 */
export declare function startIdleTransaction(hub: Hub, transactionContext: TransactionContext, idleTimeout: number, finalTimeout: number, onScope?: boolean, customSamplingContext?: CustomSamplingContext, heartbeatInterval?: number, delayAutoFinishUntilSignal?: boolean): IdleTransaction;
/**
 * Adds tracing extensions to the global hub.
 */
export declare function addTracingExtensions(): void;
//# sourceMappingURL=hubextensions.d.ts.map