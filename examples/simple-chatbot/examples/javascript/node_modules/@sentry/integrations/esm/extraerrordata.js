import { defineIntegration, convertIntegrationFnToClass } from '@sentry/core';
import { isError, normalize, isPlainObject, addNonEnumerableProperty, logger } from '@sentry/utils';
import { DEBUG_BUILD } from './debug-build.js';

const INTEGRATION_NAME = 'ExtraErrorData';

const _extraErrorDataIntegration = ((options = {}) => {
  const depth = options.depth || 3;

  // TODO(v8): Flip the default for this option to true
  const captureErrorCause = options.captureErrorCause || false;

  return {
    name: INTEGRATION_NAME,
    // TODO v8: Remove this
    setupOnce() {}, // eslint-disable-line @typescript-eslint/no-empty-function
    processEvent(event, hint) {
      return _enhanceEventWithErrorData(event, hint, depth, captureErrorCause);
    },
  };
}) ;

const extraErrorDataIntegration = defineIntegration(_extraErrorDataIntegration);

/**
 * Extract additional data for from original exceptions.
 * @deprecated Use `extraErrorDataIntegration()` instead.
 */
// eslint-disable-next-line deprecation/deprecation
const ExtraErrorData = convertIntegrationFnToClass(
  INTEGRATION_NAME,
  extraErrorDataIntegration,
)

;

function _enhanceEventWithErrorData(
  event,
  hint = {},
  depth,
  captureErrorCause,
) {
  if (!hint.originalException || !isError(hint.originalException)) {
    return event;
  }
  const exceptionName = (hint.originalException ).name || hint.originalException.constructor.name;

  const errorData = _extractErrorData(hint.originalException , captureErrorCause);

  if (errorData) {
    const contexts = {
      ...event.contexts,
    };

    const normalizedErrorData = normalize(errorData, depth);

    if (isPlainObject(normalizedErrorData)) {
      // We mark the error data as "already normalized" here, because we don't want other normalization procedures to
      // potentially truncate the data we just already normalized, with a certain depth setting.
      addNonEnumerableProperty(normalizedErrorData, '__sentry_skip_normalization__', true);
      contexts[exceptionName] = normalizedErrorData;
    }

    return {
      ...event,
      contexts,
    };
  }

  return event;
}

/**
 * Extract extra information from the Error object
 */
function _extractErrorData(error, captureErrorCause) {
  // We are trying to enhance already existing event, so no harm done if it won't succeed
  try {
    const nativeKeys = [
      'name',
      'message',
      'stack',
      'line',
      'column',
      'fileName',
      'lineNumber',
      'columnNumber',
      'toJSON',
    ];

    const extraErrorInfo = {};

    // We want only enumerable properties, thus `getOwnPropertyNames` is redundant here, as we filter keys anyway.
    for (const key of Object.keys(error)) {
      if (nativeKeys.indexOf(key) !== -1) {
        continue;
      }
      const value = error[key];
      extraErrorInfo[key] = isError(value) ? value.toString() : value;
    }

    // Error.cause is a standard property that is non enumerable, we therefore need to access it separately.
    // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Error/cause
    if (captureErrorCause && error.cause !== undefined) {
      extraErrorInfo.cause = isError(error.cause) ? error.cause.toString() : error.cause;
    }

    // Check if someone attached `toJSON` method to grab even more properties (eg. axios is doing that)
    if (typeof error.toJSON === 'function') {
      const serializedError = error.toJSON() ;

      for (const key of Object.keys(serializedError)) {
        const value = serializedError[key];
        extraErrorInfo[key] = isError(value) ? value.toString() : value;
      }
    }

    return extraErrorInfo;
  } catch (oO) {
    DEBUG_BUILD && logger.error('Unable to extract extra data from the Error object:', oO);
  }

  return null;
}

export { ExtraErrorData, extraErrorDataIntegration };
//# sourceMappingURL=extraerrordata.js.map
