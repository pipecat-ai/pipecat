import { DEBUG_BUILD } from './debug-build.js';
import { GLOBAL_OBJ } from './worldwide.js';

/** Prefix for logging strings */
const PREFIX = 'Sentry Logger ';

const CONSOLE_LEVELS = [
  'debug',
  'info',
  'warn',
  'error',
  'log',
  'assert',
  'trace',
] ;

/** This may be mutated by the console instrumentation. */
const originalConsoleMethods

 = {};

/** JSDoc */

/**
 * Temporarily disable sentry console instrumentations.
 *
 * @param callback The function to run against the original `console` messages
 * @returns The results of the callback
 */
function consoleSandbox(callback) {
  if (!('console' in GLOBAL_OBJ)) {
    return callback();
  }

  const console = GLOBAL_OBJ.console ;
  const wrappedFuncs = {};

  const wrappedLevels = Object.keys(originalConsoleMethods) ;

  // Restore all wrapped console methods
  wrappedLevels.forEach(level => {
    const originalConsoleMethod = originalConsoleMethods[level] ;
    wrappedFuncs[level] = console[level] ;
    console[level] = originalConsoleMethod;
  });

  try {
    return callback();
  } finally {
    // Revert restoration to wrapped state
    wrappedLevels.forEach(level => {
      console[level] = wrappedFuncs[level] ;
    });
  }
}

function makeLogger() {
  let enabled = false;
  const logger = {
    enable: () => {
      enabled = true;
    },
    disable: () => {
      enabled = false;
    },
    isEnabled: () => enabled,
  };

  if (DEBUG_BUILD) {
    CONSOLE_LEVELS.forEach(name => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      logger[name] = (...args) => {
        if (enabled) {
          consoleSandbox(() => {
            GLOBAL_OBJ.console[name](`${PREFIX}[${name}]:`, ...args);
          });
        }
      };
    });
  } else {
    CONSOLE_LEVELS.forEach(name => {
      logger[name] = () => undefined;
    });
  }

  return logger ;
}

const logger = makeLogger();

export { CONSOLE_LEVELS, consoleSandbox, logger, originalConsoleMethods };
//# sourceMappingURL=logger.js.map
