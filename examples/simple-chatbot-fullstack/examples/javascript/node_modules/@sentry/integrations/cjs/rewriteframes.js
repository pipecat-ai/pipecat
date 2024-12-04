Object.defineProperty(exports, '__esModule', { value: true });

const core = require('@sentry/core');
const utils = require('@sentry/utils');

const INTEGRATION_NAME = 'RewriteFrames';

const _rewriteFramesIntegration = ((options = {}) => {
  const root = options.root;
  const prefix = options.prefix || 'app:///';

  const iteratee =
    options.iteratee ||
    ((frame) => {
      if (!frame.filename) {
        return frame;
      }
      // Determine if this is a Windows frame by checking for a Windows-style prefix such as `C:\`
      const isWindowsFrame =
        /^[a-zA-Z]:\\/.test(frame.filename) ||
        // or the presence of a backslash without a forward slash (which are not allowed on Windows)
        (frame.filename.includes('\\') && !frame.filename.includes('/'));
      // Check if the frame filename begins with `/`
      const startsWithSlash = /^\//.test(frame.filename);
      if (isWindowsFrame || startsWithSlash) {
        const filename = isWindowsFrame
          ? frame.filename
              .replace(/^[a-zA-Z]:/, '') // remove Windows-style prefix
              .replace(/\\/g, '/') // replace all `\\` instances with `/`
          : frame.filename;
        const base = root ? utils.relative(root, filename) : utils.basename(filename);
        frame.filename = `${prefix}${base}`;
      }
      return frame;
    });

  /** Process an exception event. */
  function _processExceptionsEvent(event) {
    try {
      return {
        ...event,
        exception: {
          ...event.exception,
          // The check for this is performed inside `process` call itself, safe to skip here
          // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
          values: event.exception.values.map(value => ({
            ...value,
            ...(value.stacktrace && { stacktrace: _processStacktrace(value.stacktrace) }),
          })),
        },
      };
    } catch (_oO) {
      return event;
    }
  }

  /** Process a stack trace. */
  function _processStacktrace(stacktrace) {
    return {
      ...stacktrace,
      frames: stacktrace && stacktrace.frames && stacktrace.frames.map(f => iteratee(f)),
    };
  }

  return {
    name: INTEGRATION_NAME,
    // TODO v8: Remove this
    setupOnce() {}, // eslint-disable-line @typescript-eslint/no-empty-function
    processEvent(originalEvent) {
      let processedEvent = originalEvent;

      if (originalEvent.exception && Array.isArray(originalEvent.exception.values)) {
        processedEvent = _processExceptionsEvent(processedEvent);
      }

      return processedEvent;
    },
  };
}) ;

const rewriteFramesIntegration = core.defineIntegration(_rewriteFramesIntegration);

/**
 * Rewrite event frames paths.
 * @deprecated Use `rewriteFramesIntegration()` instead.
 */
// eslint-disable-next-line deprecation/deprecation
const RewriteFrames = core.convertIntegrationFnToClass(
  INTEGRATION_NAME,
  rewriteFramesIntegration,
)

;

exports.RewriteFrames = RewriteFrames;
exports.rewriteFramesIntegration = rewriteFramesIntegration;
//# sourceMappingURL=rewriteframes.js.map
