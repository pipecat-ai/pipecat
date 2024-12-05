import { AllPerformanceEntry, AllPerformanceEntryData, LargestContentfulPaintData, ReplayPerformanceEntry } from '../types';
/**
 * Create replay performance entries from the browser performance entries.
 */
export declare function createPerformanceEntries(entries: AllPerformanceEntry[]): ReplayPerformanceEntry<AllPerformanceEntryData>[];
/**
 * Add a LCP event to the replay based on an LCP metric.
 */
export declare function getLargestContentfulPaint(metric: {
    value: number;
    entries: PerformanceEntry[];
}): ReplayPerformanceEntry<LargestContentfulPaintData>;
//# sourceMappingURL=createPerformanceEntries.d.ts.map
