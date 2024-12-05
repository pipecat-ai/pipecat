import { dynamicRequire } from '@sentry/utils';

const lazyLoadedNodePerformanceMonitoringIntegrations = [
  () => {
    const integration = dynamicRequire(module, './apollo')

;
    return new integration.Apollo();
  },
  () => {
    const integration = dynamicRequire(module, './apollo')

;
    return new integration.Apollo({ useNestjs: true });
  },
  () => {
    const integration = dynamicRequire(module, './graphql')

;
    return new integration.GraphQL();
  },
  () => {
    const integration = dynamicRequire(module, './mongo')

;
    return new integration.Mongo();
  },
  () => {
    const integration = dynamicRequire(module, './mongo')

;
    return new integration.Mongo({ mongoose: true });
  },
  () => {
    const integration = dynamicRequire(module, './mysql')

;
    return new integration.Mysql();
  },
  () => {
    const integration = dynamicRequire(module, './postgres')

;
    return new integration.Postgres();
  },
];

export { lazyLoadedNodePerformanceMonitoringIntegrations };
//# sourceMappingURL=lazy.js.map
