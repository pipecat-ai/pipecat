import { defineIntegration, convertIntegrationFnToClass, getClient, withScope, captureMessage, captureException } from '@sentry/core';
import { CONSOLE_LEVELS, GLOBAL_OBJ, addConsoleInstrumentationHandler, severityLevelFromString, addExceptionMechanism, safeJoin } from '@sentry/utils';

const INTEGRATION_NAME = 'CaptureConsole';

const _captureConsoleIntegration = ((options = {}) => {
  const levels = options.levels || CONSOLE_LEVELS;

  return {
    name: INTEGRATION_NAME,
    // TODO v8: Remove this
    setupOnce() {}, // eslint-disable-line @typescript-eslint/no-empty-function
    setup(client) {
      if (!('console' in GLOBAL_OBJ)) {
        return;
      }

      addConsoleInstrumentationHandler(({ args, level }) => {
        if (getClient() !== client || !levels.includes(level)) {
          return;
        }

        consoleHandler(args, level);
      });
    },
  };
}) ;

const captureConsoleIntegration = defineIntegration(_captureConsoleIntegration);

/**
 * Send Console API calls as Sentry Events.
 * @deprecated Use `captureConsoleIntegration()` instead.
 */
// eslint-disable-next-line deprecation/deprecation
const CaptureConsole = convertIntegrationFnToClass(
  INTEGRATION_NAME,
  captureConsoleIntegration,
)

;

function consoleHandler(args, level) {
  const captureContext = {
    level: severityLevelFromString(level),
    extra: {
      arguments: args,
    },
  };

  withScope(scope => {
    scope.addEventProcessor(event => {
      event.logger = 'console';

      addExceptionMechanism(event, {
        handled: false,
        type: 'console',
      });

      return event;
    });

    if (level === 'assert' && args[0] === false) {
      const message = `Assertion failed: ${safeJoin(args.slice(1), ' ') || 'console.assert'}`;
      scope.setExtra('arguments', args.slice(1));
      captureMessage(message, captureContext);
      return;
    }

    const error = args.find(arg => arg instanceof Error);
    if (level === 'error' && error) {
      captureException(error, captureContext);
      return;
    }

    const message = safeJoin(args, ' ');
    captureMessage(message, captureContext);
  });
}

export { CaptureConsole, captureConsoleIntegration };
//# sourceMappingURL=captureconsole.js.map
