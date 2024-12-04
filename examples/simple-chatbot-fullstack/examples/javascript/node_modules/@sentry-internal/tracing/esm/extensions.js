import { addTracingExtensions, getMainCarrier } from '@sentry/core';
import { isNodeEnv, loadModule, dynamicRequire } from '@sentry/utils';

/**
 * @private
 */
function _autoloadDatabaseIntegrations() {
  const carrier = getMainCarrier();
  if (!carrier.__SENTRY__) {
    return;
  }

  const packageToIntegrationMapping = {
    mongodb() {
      const integration = dynamicRequire(module, './node/integrations/mongo')

;
      return new integration.Mongo();
    },
    mongoose() {
      const integration = dynamicRequire(module, './node/integrations/mongo')

;
      return new integration.Mongo();
    },
    mysql() {
      const integration = dynamicRequire(module, './node/integrations/mysql')

;
      return new integration.Mysql();
    },
    pg() {
      const integration = dynamicRequire(module, './node/integrations/postgres')

;
      return new integration.Postgres();
    },
  };

  const mappedPackages = Object.keys(packageToIntegrationMapping)
    .filter(moduleName => !!loadModule(moduleName))
    .map(pkg => {
      try {
        return packageToIntegrationMapping[pkg]();
      } catch (e) {
        return undefined;
      }
    })
    .filter(p => p) ;

  if (mappedPackages.length > 0) {
    carrier.__SENTRY__.integrations = [...(carrier.__SENTRY__.integrations || []), ...mappedPackages];
  }
}

/**
 * This patches the global object and injects the Tracing extensions methods
 */
function addExtensionMethods() {
  addTracingExtensions();

  // Detect and automatically load specified integrations.
  if (isNodeEnv()) {
    _autoloadDatabaseIntegrations();
  }
}

export { addExtensionMethods };
//# sourceMappingURL=extensions.js.map
