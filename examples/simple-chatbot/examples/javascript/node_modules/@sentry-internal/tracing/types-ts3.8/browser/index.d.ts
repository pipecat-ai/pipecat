export * from '../exports';
export { RequestInstrumentationOptions } from './request';
export { BrowserTracing, BROWSER_TRACING_INTEGRATION_ID, } from './browsertracing';
export { browserTracingIntegration, startBrowserTracingNavigationSpan, startBrowserTracingPageLoadSpan, } from './browserTracingIntegration';
export { instrumentOutgoingRequests, defaultRequestInstrumentationOptions } from './request';
export { addPerformanceInstrumentationHandler, addClsInstrumentationHandler, addFidInstrumentationHandler, addLcpInstrumentationHandler, addTtfbInstrumentationHandler, addInpInstrumentationHandler, } from './instrument';
//# sourceMappingURL=index.d.ts.map
