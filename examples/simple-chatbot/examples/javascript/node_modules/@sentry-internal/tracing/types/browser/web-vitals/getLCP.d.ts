import type { ReportCallback, StopListening } from './types';
/**
 * Calculates the [LCP](https://web.dev/lcp/) value for the current page and
 * calls the `callback` function once the value is ready (along with the
 * relevant `largest-contentful-paint` performance entry used to determine the
 * value). The reported value is a `DOMHighResTimeStamp`.
 */
export declare const onLCP: (onReport: ReportCallback) => StopListening | undefined;
//# sourceMappingURL=getLCP.d.ts.map