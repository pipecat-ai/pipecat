import { Integration } from '@sentry/types';
export interface LazyLoadedIntegration<T = object> extends Integration {
    /**
     * Loads the integration's dependency and caches it so it doesn't have to be loaded again.
     *
     * If this returns undefined, the dependency could not be loaded.
     */
    loadDependency(): T | undefined;
}
export declare const lazyLoadedNodePerformanceMonitoringIntegrations: (() => LazyLoadedIntegration)[];
//# sourceMappingURL=lazy.d.ts.map
