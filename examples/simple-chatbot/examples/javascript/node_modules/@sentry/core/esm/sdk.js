import { logger, consoleSandbox } from '@sentry/utils';
import { DEBUG_BUILD } from './debug-build.js';
import { getCurrentScope } from './exports.js';
import { getCurrentHub } from './hub.js';

/** A class object that can instantiate Client objects. */

/**
 * Internal function to create a new SDK client instance. The client is
 * installed and then bound to the current scope.
 *
 * @param clientClass The client class to instantiate.
 * @param options Options to pass to the client.
 */
function initAndBind(
  clientClass,
  options,
) {
  if (options.debug === true) {
    if (DEBUG_BUILD) {
      logger.enable();
    } else {
      // use `console.warn` rather than `logger.warn` since by non-debug bundles have all `logger.x` statements stripped
      consoleSandbox(() => {
        // eslint-disable-next-line no-console
        console.warn('[Sentry] Cannot initialize SDK with `debug` option using a non-debug bundle.');
      });
    }
  }
  const scope = getCurrentScope();
  scope.update(options.initialScope);

  const client = new clientClass(options);
  setCurrentClient(client);
  initializeClient(client);
}

/**
 * Make the given client the current client.
 */
function setCurrentClient(client) {
  // eslint-disable-next-line deprecation/deprecation
  const hub = getCurrentHub();
  // eslint-disable-next-line deprecation/deprecation
  const top = hub.getStackTop();
  top.client = client;
  top.scope.setClient(client);
}

/**
 * Initialize the client for the current scope.
 * Make sure to call this after `setCurrentClient()`.
 */
function initializeClient(client) {
  if (client.init) {
    client.init();
    // TODO v8: Remove this fallback
    // eslint-disable-next-line deprecation/deprecation
  } else if (client.setupIntegrations) {
    // eslint-disable-next-line deprecation/deprecation
    client.setupIntegrations();
  }
}

export { initAndBind, setCurrentClient };
//# sourceMappingURL=sdk.js.map
