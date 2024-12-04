import { isError, isPlainObject, isParameterizedString } from './is.js';
import { addExceptionTypeValue, addExceptionMechanism } from './misc.js';
import { normalizeToSize } from './normalize.js';
import { extractExceptionKeysForMessage } from './object.js';

/**
 * Extracts stack frames from the error.stack string
 */
function parseStackFrames(stackParser, error) {
  return stackParser(error.stack || '', 1);
}

/**
 * Extracts stack frames from the error and builds a Sentry Exception
 */
function exceptionFromError(stackParser, error) {
  const exception = {
    type: error.name || error.constructor.name,
    value: error.message,
  };

  const frames = parseStackFrames(stackParser, error);
  if (frames.length) {
    exception.stacktrace = { frames };
  }

  return exception;
}

function getMessageForObject(exception) {
  if ('name' in exception && typeof exception.name === 'string') {
    let message = `'${exception.name}' captured as exception`;

    if ('message' in exception && typeof exception.message === 'string') {
      message += ` with message '${exception.message}'`;
    }

    return message;
  } else if ('message' in exception && typeof exception.message === 'string') {
    return exception.message;
  } else {
    // This will allow us to group events based on top-level keys
    // which is much better than creating new group when any key/value change
    return `Object captured as exception with keys: ${extractExceptionKeysForMessage(
      exception ,
    )}`;
  }
}

/**
 * Builds and Event from a Exception
 *
 * TODO(v8): Remove getHub fallback
 * @hidden
 */
function eventFromUnknownInput(
  // eslint-disable-next-line deprecation/deprecation
  getHubOrClient,
  stackParser,
  exception,
  hint,
) {
  const client =
    typeof getHubOrClient === 'function'
      ? // eslint-disable-next-line deprecation/deprecation
        getHubOrClient().getClient()
      : getHubOrClient;

  let ex = exception;
  const providedMechanism =
    hint && hint.data && (hint.data ).mechanism;
  const mechanism = providedMechanism || {
    handled: true,
    type: 'generic',
  };

  let extras;

  if (!isError(exception)) {
    if (isPlainObject(exception)) {
      const normalizeDepth = client && client.getOptions().normalizeDepth;
      extras = { ['__serialized__']: normalizeToSize(exception , normalizeDepth) };

      const message = getMessageForObject(exception);
      ex = (hint && hint.syntheticException) || new Error(message);
      (ex ).message = message;
    } else {
      // This handles when someone does: `throw "something awesome";`
      // We use synthesized Error here so we can extract a (rough) stack trace.
      ex = (hint && hint.syntheticException) || new Error(exception );
      (ex ).message = exception ;
    }
    mechanism.synthetic = true;
  }

  const event = {
    exception: {
      values: [exceptionFromError(stackParser, ex )],
    },
  };

  if (extras) {
    event.extra = extras;
  }

  addExceptionTypeValue(event, undefined, undefined);
  addExceptionMechanism(event, mechanism);

  return {
    ...event,
    event_id: hint && hint.event_id,
  };
}

/**
 * Builds and Event from a Message
 * @hidden
 */
function eventFromMessage(
  stackParser,
  message,
  // eslint-disable-next-line deprecation/deprecation
  level = 'info',
  hint,
  attachStacktrace,
) {
  const event = {
    event_id: hint && hint.event_id,
    level,
  };

  if (attachStacktrace && hint && hint.syntheticException) {
    const frames = parseStackFrames(stackParser, hint.syntheticException);
    if (frames.length) {
      event.exception = {
        values: [
          {
            value: message,
            stacktrace: { frames },
          },
        ],
      };
    }
  }

  if (isParameterizedString(message)) {
    const { __sentry_template_string__, __sentry_template_values__ } = message;

    event.logentry = {
      message: __sentry_template_string__,
      params: __sentry_template_values__,
    };
    return event;
  }

  event.message = message;
  return event;
}

export { eventFromMessage, eventFromUnknownInput, exceptionFromError, parseStackFrames };
//# sourceMappingURL=eventbuilder.js.map
