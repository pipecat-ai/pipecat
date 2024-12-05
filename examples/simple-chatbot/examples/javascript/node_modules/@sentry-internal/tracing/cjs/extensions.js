Object.defineProperty(exports, '__esModule', { value: true });

const core = require('@sentry/core');
const utils = require('@sentry/utils');

/**
 * @private
 */
function _autoloadDatabaseIntegrations() {
  const carrier = core.getMainCarrier();
  if (!carrier.__SENTRY__) {
    return;
  }

  const packageToIntegrationMapping = {
    mongodb() {
      const integration = utils.dynamicRequire(module, './node/integrations/mongo')

;
      return new integration.Mongo();
    },
    mongoose() {
      const integration = utils.dynamicRequire(module, './node/integrations/mongo')

;
      return new integration.Mongo();
    },
    mysql() {
      const integration = utils.dynamicRequire(module, './node/integrations/mysql')

;
      return new integration.Mysql();
    },
    pg() {
      const integration = utils.dynamicRequire(module, './node/integrations/postgres')

;
      return new integration.Postgres();
    },
  };

  const mappedPackages = Object.keys(packageToIntegrationMapping)
    .filter(moduleName => !!utils.loadModule(moduleName))
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
  core.addTracingExtensions();

  // Detect and automatically load specified integrations.
  if (utils.isNodeEnv()) {
    _autoloadDatabaseIntegrations();
  }
}

exports.addExtensionMethods = addExtensionMethods;
//# sourceMappingURL=extensions.js.map
