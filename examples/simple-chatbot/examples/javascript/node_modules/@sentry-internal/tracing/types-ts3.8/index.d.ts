export * from './exports';
export { Apollo, Express, GraphQL, Mongo, Mysql, Postgres, Prisma, lazyLoadedNodePerformanceMonitoringIntegrations, } from './node';
export { LazyLoadedIntegration } from './node';
export { BrowserTracing, browserTracingIntegration, startBrowserTracingNavigationSpan, startBrowserTracingPageLoadSpan, BROWSER_TRACING_INTEGRATION_ID, instrumentOutgoingRequests, defaultRequestInstrumentationOptions, addPerformanceInstrumentationHandler, addClsInstrumentationHandler, addFidInstrumentationHandler, addLcpInstrumentationHandler, } from './browser';
export { addTracingHeadersToFetchRequest, instrumentFetchRequest } from './common/fetch';
export { RequestInstrumentationOptions } from './browser';
export { addExtensionMethods } from './extensions';
//# sourceMappingURL=index.d.ts.map
