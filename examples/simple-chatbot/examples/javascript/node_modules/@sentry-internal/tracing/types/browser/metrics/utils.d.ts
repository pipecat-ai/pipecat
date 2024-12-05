import type { Transaction } from '@sentry/core';
import type { Span, SpanContext } from '@sentry/types';
/**
 * Checks if a given value is a valid measurement value.
 */
export declare function isMeasurementValue(value: unknown): value is number;
/**
 * Helper function to start child on transactions. This function will make sure that the transaction will
 * use the start timestamp of the created child span if it is earlier than the transactions actual
 * start timestamp.
 *
 * Note: this will not be possible anymore in v8,
 * unless we do some special handling for browser here...
 */
export declare function _startChild(transaction: Transaction, { startTimestamp, ...ctx }: SpanContext): Span;
//# sourceMappingURL=utils.d.ts.map