import type { Transaction } from '@sentry/types';
import { extractTraceparentData as _extractTraceparentData } from '@sentry/utils';
import type { Hub } from '../hub';
/**
 * Grabs active transaction off scope.
 *
 * @deprecated You should not rely on the transaction, but just use `startSpan()` APIs instead.
 */
export declare function getActiveTransaction<T extends Transaction>(maybeHub?: Hub): T | undefined;
export { stripUrlQueryAndFragment } from '@sentry/utils';
/**
 * The `extractTraceparentData` function and `TRACEPARENT_REGEXP` constant used
 * to be declared in this file. It was later moved into `@sentry/utils` as part of a
 * move to remove `@sentry/tracing` dependencies from `@sentry/node` (`extractTraceparentData`
 * is the only tracing function used by `@sentry/node`).
 *
 * These exports are kept here for backwards compatability's sake.
 *
 * See https://github.com/getsentry/sentry-javascript/issues/4642 for more details.
 *
 * @deprecated Import this function from `@sentry/utils` instead
 */
export declare const extractTraceparentData: typeof _extractTraceparentData;
//# sourceMappingURL=utils.d.ts.map