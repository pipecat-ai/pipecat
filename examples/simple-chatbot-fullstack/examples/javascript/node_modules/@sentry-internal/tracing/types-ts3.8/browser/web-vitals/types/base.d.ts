import { FirstInputPolyfillEntry, NavigationTimingPolyfillEntry } from './polyfills';
export interface Metric {
    /**
     * The name of the metric (in acronym form).
     */
    name: 'CLS' | 'FCP' | 'FID' | 'INP' | 'LCP' | 'TTFB';
    /**
     * The current value of the metric.
     */
    value: number;
    /**
     * The rating as to whether the metric value is within the "good",
     * "needs improvement", or "poor" thresholds of the metric.
     */
    rating: 'good' | 'needs-improvement' | 'poor';
    /**
     * The delta between the current value and the last-reported value.
     * On the first report, `delta` and `value` will always be the same.
     */
    delta: number;
    /**
     * A unique ID representing this particular metric instance. This ID can
     * be used by an analytics tool to dedupe multiple values sent for the same
     * metric instance, or to group multiple deltas together and calculate a
     * total. It can also be used to differentiate multiple different metric
     * instances sent from the same page, which can happen if the page is
     * restored from the back/forward cache (in that case new metrics object
     * get created).
     */
    id: string;
    /**
     * Any performance entries relevant to the metric value calculation.
     * The array may also be empty if the metric value was not based on any
     * entries (e.g. a CLS value of 0 given no layout shifts).
     */
    entries: (PerformanceEntry | LayoutShift | FirstInputPolyfillEntry | NavigationTimingPolyfillEntry)[];
    /**
     * The type of navigation
     *
     * Navigation Timing API (or `undefined` if the browser doesn't
     * support that API). For pages that are restored from the bfcache, this
     * value will be 'back-forward-cache'.
     */
    navigationType: 'navigate' | 'reload' | 'back-forward' | 'back-forward-cache' | 'prerender';
}
/**
 * A version of the `Metric` that is used with the attribution build.
 */
export interface MetricWithAttribution extends Metric {
    /**
     * An object containing potentially-helpful debugging information that
     * can be sent along with the metric value for the current page visit in
     * order to help identify issues happening to real-users in the field.
     */
    attribution: {
        [key: string]: unknown;
    };
}
export interface ReportCallback {
    (metric: Metric): void;
}
export interface ReportOpts {
    reportAllChanges?: boolean;
    durationThreshold?: number;
}
/**
 * The loading state of the document. Note: this value is similar to
 * `document.readyState` but it subdivides the "interactive" state into the
 * time before and after the DOMContentLoaded event fires.
 *
 * State descriptions:
 * - `loading`: the initial document response has not yet been fully downloaded
 *   and parsed. This is equivalent to the corresponding `readyState` value.
 * - `dom-interactive`: the document has been fully loaded and parsed, but
 *   scripts may not have yet finished loading and executing.
 * - `dom-content-loaded`: the document is fully loaded and parsed, and all
 *   scripts (except `async` scripts) have loaded and finished executing.
 * - `complete`: the document and all of its sub-resources have finished
 *   loading. This is equivalent to the corresponding `readyState` value.
 */
export type LoadState = 'loading' | 'dom-interactive' | 'dom-content-loaded' | 'complete';
export type StopListening = () => void;
//# sourceMappingURL=base.d.ts.map
