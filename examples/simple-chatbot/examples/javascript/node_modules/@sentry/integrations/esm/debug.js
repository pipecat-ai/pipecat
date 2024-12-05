import { defineIntegration, convertIntegrationFnToClass } from '@sentry/core';
import { consoleSandbox } from '@sentry/utils';

const INTEGRATION_NAME = 'Debug';

const _debugIntegration = ((options = {}) => {
  const _options = {
    debugger: false,
    stringify: false,
    ...options,
  };

  return {
    name: INTEGRATION_NAME,
    // TODO v8: Remove this
    setupOnce() {}, // eslint-disable-line @typescript-eslint/no-empty-function
    setup(client) {
      if (!client.on) {
        return;
      }

      client.on('beforeSendEvent', (event, hint) => {
        if (_options.debugger) {
          // eslint-disable-next-line no-debugger
          debugger;
        }

        /* eslint-disable no-console */
        consoleSandbox(() => {
          if (_options.stringify) {
            console.log(JSON.stringify(event, null, 2));
            if (hint && Object.keys(hint).length) {
              console.log(JSON.stringify(hint, null, 2));
            }
          } else {
            console.log(event);
            if (hint && Object.keys(hint).length) {
              console.log(hint);
            }
          }
        });
        /* eslint-enable no-console */
      });
    },
  };
}) ;

const debugIntegration = defineIntegration(_debugIntegration);

/**
 * Integration to debug sent Sentry events.
 * This integration should not be used in production.
 *
 * @deprecated Use `debugIntegration()` instead.
 */
// eslint-disable-next-line deprecation/deprecation
const Debug = convertIntegrationFnToClass(INTEGRATION_NAME, debugIntegration)

;

export { Debug, debugIntegration };
//# sourceMappingURL=debug.js.map
