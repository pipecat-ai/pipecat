import { browserTracingIntegration } from '@sentry-internal/tracing';
import { DsnLike, Integration, Mechanism, WrappedFunction } from '@sentry/types';
export declare const WINDOW: import("@sentry/utils").InternalGlobal & Window;
/**
 * @hidden
 */
export declare function shouldIgnoreOnError(): boolean;
/**
 * @hidden
 */
export declare function ignoreNextOnError(): void;
/**
 * Instruments the given function and sends an event to Sentry every time the
 * function throws an exception.
 *
 * @param fn A function to wrap. It is generally safe to pass an unbound function, because the returned wrapper always
 * has a correct `this` context.
 * @returns The wrapped function.
 * @hidden
 */
export declare function wrap(fn: WrappedFunction, options?: {
    mechanism?: Mechanism;
}, before?: WrappedFunction): any;
/**
 * All properties the report dialog supports
 *
 * @deprecated This type will be removed in the next major version of the Sentry SDK. `showReportDialog` will still be around, however the `eventId` option will now be required.
 */
export interface ReportDialogOptions {
    [key: string]: any;
    eventId?: string;
    dsn?: DsnLike;
    user?: {
        email?: string;
        name?: string;
    };
    lang?: string;
    title?: string;
    subtitle?: string;
    subtitle2?: string;
    labelName?: string;
    labelEmail?: string;
    labelComments?: string;
    labelClose?: string;
    labelSubmit?: string;
    errorGeneric?: string;
    errorFormEntry?: string;
    successMessage?: string;
    /** Callback after reportDialog showed up */
    onLoad?(this: void): void;
    /** Callback after reportDialog closed */
    onClose?(this: void): void;
}
/**
 * This is a slim shim of `browserTracingIntegration` for the CDN bundles.
 * Since the actual functional integration uses a different code from `BrowserTracing`,
 * we want to avoid shipping both of them in the CDN bundles, as that would blow up the size.
 * Instead, we provide a functional integration with the same API, but the old implementation.
 * This means that it's not possible to register custom routing instrumentation, but that's OK for now.
 * We also don't expose the utilities for this anyhow in the CDN bundles.
 * For users that need custom routing in CDN bundles, they have to continue using `new BrowserTracing()` until v8.
 */
export declare function bundleBrowserTracingIntegration(options?: Parameters<typeof browserTracingIntegration>[0]): Integration;
//# sourceMappingURL=helpers.d.ts.map
