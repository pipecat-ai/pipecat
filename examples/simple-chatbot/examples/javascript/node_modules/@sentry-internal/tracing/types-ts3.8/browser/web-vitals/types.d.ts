import { Transaction, TransactionContext, User } from '@sentry/types';
import { FirstInputPolyfillCallback } from './types/polyfills';
export * from './types/base';
export * from './types/polyfills';
export * from './types/cls';
export * from './types/fid';
export * from './types/lcp';
export interface WebVitalsGlobal {
    firstInputPolyfill: (onFirstInput: FirstInputPolyfillCallback) => void;
    resetFirstInputPolyfill: () => void;
    firstHiddenTime: number;
}
declare global {
    interface Window {
        webVitals: WebVitalsGlobal;
        __WEB_VITALS_POLYFILL__: boolean;
    }
}
export type PerformancePaintTiming = PerformanceEntry;
export interface PerformanceEventTiming extends PerformanceEntry {
    processingStart: DOMHighResTimeStamp;
    processingEnd: DOMHighResTimeStamp;
    duration: DOMHighResTimeStamp;
    cancelable?: boolean;
    target?: Element;
}
interface PerformanceEntryMap {
    navigation: PerformanceNavigationTiming;
    resource: PerformanceResourceTiming;
    paint: PerformancePaintTiming;
}
export interface NavigatorNetworkInformation {
    readonly connection?: NetworkInformation;
}
type ConnectionType = 'bluetooth' | 'cellular' | 'ethernet' | 'mixed' | 'none' | 'other' | 'unknown' | 'wifi' | 'wimax';
type EffectiveConnectionType = '2g' | '3g' | '4g' | 'slow-2g';
type Megabit = number;
type Millisecond = number;
interface NetworkInformation extends EventTarget {
    readonly type?: ConnectionType;
    readonly effectiveType?: EffectiveConnectionType;
    readonly downlinkMax?: Megabit;
    readonly downlink?: Megabit;
    readonly rtt?: Millisecond;
    readonly saveData?: boolean;
    onchange?: EventListener;
}
export interface NavigatorDeviceMemory {
    readonly deviceMemory?: number;
}
export type NavigationTimingPolyfillEntry = Pick<PerformanceNavigationTiming, Exclude<keyof PerformanceNavigationTiming, 'initiatorType' | 'nextHopProtocol' | 'redirectCount' | 'transferSize' | 'encodedBodySize' | 'decodedBodySize' | 'toJSON'>>;
declare global {
    interface Document {
        prerendering?: boolean;
    }
    interface Performance {
        getEntriesByType<K extends keyof PerformanceEntryMap>(type: K): PerformanceEntryMap[K][];
    }
    interface PerformanceObserverInit {
        durationThreshold?: number;
    }
    interface PerformanceNavigationTiming {
        activationStart?: number;
    }
    interface PerformanceEventTiming extends PerformanceEntry {
        duration: DOMHighResTimeStamp;
        interactionId?: number;
        readonly target: Node | null;
    }
    interface LayoutShiftAttribution {
        node?: Node;
        previousRect: DOMRectReadOnly;
        currentRect: DOMRectReadOnly;
    }
    interface LayoutShift extends PerformanceEntry {
        value: number;
        sources: LayoutShiftAttribution[];
        hadRecentInput: boolean;
    }
    interface LargestContentfulPaint extends PerformanceEntry {
        renderTime: DOMHighResTimeStamp;
        loadTime: DOMHighResTimeStamp;
        size: number;
        id: string;
        url: string;
        element?: Element;
    }
}
export type InteractionRouteNameMapping = {
    [key: string]: {
        routeName: string;
        duration: number;
        parentContext: TransactionContext;
        user?: User;
        activeTransaction?: Transaction;
        replayId?: string;
        startTime: number;
    };
};
//# sourceMappingURL=types.d.ts.map
