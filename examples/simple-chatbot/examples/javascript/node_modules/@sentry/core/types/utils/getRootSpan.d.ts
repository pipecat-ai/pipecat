import type { Span } from '@sentry/types';
/**
 * Returns the root span of a given span.
 *
 * As long as we use `Transaction`s internally, the returned root span
 * will be a `Transaction` but be aware that this might change in the future.
 *
 * If the given span has no root span or transaction, `undefined` is returned.
 */
export declare function getRootSpan(span: Span): Span | undefined;
//# sourceMappingURL=getRootSpan.d.ts.map