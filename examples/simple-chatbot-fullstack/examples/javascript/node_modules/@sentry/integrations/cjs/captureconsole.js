Object.defineProperty(exports, '__esModule', { value: true });

const core = require('@sentry/core');
const utils = require('@sentry/utils');

const INTEGRATION_NAME = 'CaptureConsole';

const _captureConsoleIntegration = ((options = {}) => {
  const levels = options.levels || utils.CONSOLE_LEVELS;

  return {
    name: INTEGRATION_NAME,
    // TODO v8: Remove this
    setupOnce() {}, // eslint-disable-line @typescript-eslint/no-empty-function
    setup(client) {
      if (!('console' in utils.GLOBAL_OBJ)) {
        return;
      }

      utils.addConsoleInstrumentationHandler(({ args, level }) => {
        if (core.getClient() !== client || !levels.includes(level)) {
          return;
        }

        consoleHandler(args, level);
      });
    },
  };
}) ;

const captureConsoleIntegration = core.defineIntegration(_captureConsoleIntegration);

/**
 * Send Console API calls as Sentry Events.
 * @deprecated Use `captureConsoleIntegration()` instead.
 */
// eslint-disable-next-line deprecation/deprecation
const CaptureConsole = core.convertIntegrationFnToClass(
  INTEGRATION_NAME,
  captureConsoleIntegration,
)

;

function consoleHandler(args, level) {
  const captureContext = {
    level: utils.severityLevelFromString(level),
    extra: {
      arguments: args,
    },
  };

  core.withScope(scope => {
    scope.addEventProcessor(event => {
      event.logger = 'console';

      utils.addExceptionMechanism(event, {
        handled: false,
        type: 'console',
      });

      return event;
    });

    if (level === 'assert' && args[0] === false) {
      const message = `Assertion failed: ${utils.safeJoin(args.slice(1), ' ') || 'console.assert'}`;
      scope.setExtra('arguments', args.slice(1));
      core.captureMessage(message, captureContext);
      return;
    }

    const error = args.find(arg => arg instanceof Error);
    if (level === 'error' && error) {
      core.captureException(error, captureContext);
      return;
    }

    const message = utils.safeJoin(args, ' ');
    core.captureMessage(message, captureContext);
  });
}

exports.CaptureConsole = CaptureConsole;
exports.captureConsoleIntegration = captureConsoleIntegration;
//# sourceMappingURL=captureconsole.js.map
