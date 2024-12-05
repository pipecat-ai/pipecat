Object.defineProperty(exports, '__esModule', { value: true });

const core = require('@sentry/core');
const utils = require('@sentry/utils');
const express = require('./node/integrations/express.js');
const postgres = require('./node/integrations/postgres.js');
const mysql = require('./node/integrations/mysql.js');
const mongo = require('./node/integrations/mongo.js');
const prisma = require('./node/integrations/prisma.js');
const graphql = require('./node/integrations/graphql.js');
const apollo = require('./node/integrations/apollo.js');
const lazy = require('./node/integrations/lazy.js');
const browsertracing = require('./browser/browsertracing.js');
const browserTracingIntegration = require('./browser/browserTracingIntegration.js');
const request = require('./browser/request.js');
const instrument = require('./browser/instrument.js');
const fetch = require('./common/fetch.js');
const extensions = require('./extensions.js');



exports.IdleTransaction = core.IdleTransaction;
exports.Span = core.Span;
exports.SpanStatus = core.SpanStatus;
exports.Transaction = core.Transaction;
exports.extractTraceparentData = core.extractTraceparentData;
exports.getActiveTransaction = core.getActiveTransaction;
exports.hasTracingEnabled = core.hasTracingEnabled;
exports.spanStatusfromHttpCode = core.spanStatusfromHttpCode;
exports.startIdleTransaction = core.startIdleTransaction;
exports.TRACEPARENT_REGEXP = utils.TRACEPARENT_REGEXP;
exports.stripUrlQueryAndFragment = utils.stripUrlQueryAndFragment;
exports.Express = express.Express;
exports.Postgres = postgres.Postgres;
exports.Mysql = mysql.Mysql;
exports.Mongo = mongo.Mongo;
exports.Prisma = prisma.Prisma;
exports.GraphQL = graphql.GraphQL;
exports.Apollo = apollo.Apollo;
exports.lazyLoadedNodePerformanceMonitoringIntegrations = lazy.lazyLoadedNodePerformanceMonitoringIntegrations;
exports.BROWSER_TRACING_INTEGRATION_ID = browsertracing.BROWSER_TRACING_INTEGRATION_ID;
exports.BrowserTracing = browsertracing.BrowserTracing;
exports.browserTracingIntegration = browserTracingIntegration.browserTracingIntegration;
exports.startBrowserTracingNavigationSpan = browserTracingIntegration.startBrowserTracingNavigationSpan;
exports.startBrowserTracingPageLoadSpan = browserTracingIntegration.startBrowserTracingPageLoadSpan;
exports.defaultRequestInstrumentationOptions = request.defaultRequestInstrumentationOptions;
exports.instrumentOutgoingRequests = request.instrumentOutgoingRequests;
exports.addClsInstrumentationHandler = instrument.addClsInstrumentationHandler;
exports.addFidInstrumentationHandler = instrument.addFidInstrumentationHandler;
exports.addLcpInstrumentationHandler = instrument.addLcpInstrumentationHandler;
exports.addPerformanceInstrumentationHandler = instrument.addPerformanceInstrumentationHandler;
exports.addTracingHeadersToFetchRequest = fetch.addTracingHeadersToFetchRequest;
exports.instrumentFetchRequest = fetch.instrumentFetchRequest;
exports.addExtensionMethods = extensions.addExtensionMethods;
//# sourceMappingURL=index.js.map
