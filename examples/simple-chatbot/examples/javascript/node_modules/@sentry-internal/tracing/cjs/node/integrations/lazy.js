Object.defineProperty(exports, '__esModule', { value: true });

const utils = require('@sentry/utils');

const lazyLoadedNodePerformanceMonitoringIntegrations = [
  () => {
    const integration = utils.dynamicRequire(module, './apollo')

;
    return new integration.Apollo();
  },
  () => {
    const integration = utils.dynamicRequire(module, './apollo')

;
    return new integration.Apollo({ useNestjs: true });
  },
  () => {
    const integration = utils.dynamicRequire(module, './graphql')

;
    return new integration.GraphQL();
  },
  () => {
    const integration = utils.dynamicRequire(module, './mongo')

;
    return new integration.Mongo();
  },
  () => {
    const integration = utils.dynamicRequire(module, './mongo')

;
    return new integration.Mongo({ mongoose: true });
  },
  () => {
    const integration = utils.dynamicRequire(module, './mysql')

;
    return new integration.Mysql();
  },
  () => {
    const integration = utils.dynamicRequire(module, './postgres')

;
    return new integration.Postgres();
  },
];

exports.lazyLoadedNodePerformanceMonitoringIntegrations = lazyLoadedNodePerformanceMonitoringIntegrations;
//# sourceMappingURL=lazy.js.map
