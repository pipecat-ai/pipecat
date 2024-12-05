export { IdleTransaction, Span, SpanStatus, Transaction, extractTraceparentData, getActiveTransaction, hasTracingEnabled, spanStatusfromHttpCode, startIdleTransaction } from '@sentry/core';
export { TRACEPARENT_REGEXP, stripUrlQueryAndFragment } from '@sentry/utils';
export { Express } from './node/integrations/express.js';
export { Postgres } from './node/integrations/postgres.js';
export { Mysql } from './node/integrations/mysql.js';
export { Mongo } from './node/integrations/mongo.js';
export { Prisma } from './node/integrations/prisma.js';
export { GraphQL } from './node/integrations/graphql.js';
export { Apollo } from './node/integrations/apollo.js';
export { lazyLoadedNodePerformanceMonitoringIntegrations } from './node/integrations/lazy.js';
export { BROWSER_TRACING_INTEGRATION_ID, BrowserTracing } from './browser/browsertracing.js';
export { browserTracingIntegration, startBrowserTracingNavigationSpan, startBrowserTracingPageLoadSpan } from './browser/browserTracingIntegration.js';
export { defaultRequestInstrumentationOptions, instrumentOutgoingRequests } from './browser/request.js';
export { addClsInstrumentationHandler, addFidInstrumentationHandler, addLcpInstrumentationHandler, addPerformanceInstrumentationHandler } from './browser/instrument.js';
export { addTracingHeadersToFetchRequest, instrumentFetchRequest } from './common/fetch.js';
export { addExtensionMethods } from './extensions.js';
//# sourceMappingURL=index.js.map
