import { extractPathForTransaction, addRequestDataToEvent } from '@sentry/utils';
import { convertIntegrationFnToClass, defineIntegration } from '../integration.js';
import { spanToJSON } from '../utils/spanUtils.js';

const DEFAULT_OPTIONS = {
  include: {
    cookies: true,
    data: true,
    headers: true,
    ip: false,
    query_string: true,
    url: true,
    user: {
      id: true,
      username: true,
      email: true,
    },
  },
  transactionNamingScheme: 'methodPath',
};

const INTEGRATION_NAME = 'RequestData';

const _requestDataIntegration = ((options = {}) => {
  const _addRequestData = addRequestDataToEvent;
  const _options = {
    ...DEFAULT_OPTIONS,
    ...options,
    include: {
      // @ts-expect-error It's mad because `method` isn't a known `include` key. (It's only here and not set by default in
      // `addRequestDataToEvent` for legacy reasons. TODO (v8): Change that.)
      method: true,
      ...DEFAULT_OPTIONS.include,
      ...options.include,
      user:
        options.include && typeof options.include.user === 'boolean'
          ? options.include.user
          : {
              ...DEFAULT_OPTIONS.include.user,
              // Unclear why TS still thinks `options.include.user` could be a boolean at this point
              ...((options.include || {}).user ),
            },
    },
  };

  return {
    name: INTEGRATION_NAME,
    // TODO v8: Remove this
    setupOnce() {}, // eslint-disable-line @typescript-eslint/no-empty-function
    processEvent(event, _hint, client) {
      // Note: In the long run, most of the logic here should probably move into the request data utility functions. For
      // the moment it lives here, though, until https://github.com/getsentry/sentry-javascript/issues/5718 is addressed.
      // (TL;DR: Those functions touch many parts of the repo in many different ways, and need to be clened up. Once
      // that's happened, it will be easier to add this logic in without worrying about unexpected side effects.)
      const { transactionNamingScheme } = _options;

      const { sdkProcessingMetadata = {} } = event;
      const req = sdkProcessingMetadata.request;

      if (!req) {
        return event;
      }

      // The Express request handler takes a similar `include` option to that which can be passed to this integration.
      // If passed there, we store it in `sdkProcessingMetadata`. TODO(v8): Force express and GCP people to use this
      // integration, so that all of this passing and conversion isn't necessary
      const addRequestDataOptions =
        sdkProcessingMetadata.requestDataOptionsFromExpressHandler ||
        sdkProcessingMetadata.requestDataOptionsFromGCPWrapper ||
        convertReqDataIntegrationOptsToAddReqDataOpts(_options);

      const processedEvent = _addRequestData(event, req, addRequestDataOptions);

      // Transaction events already have the right `transaction` value
      if (event.type === 'transaction' || transactionNamingScheme === 'handler') {
        return processedEvent;
      }

      // In all other cases, use the request's associated transaction (if any) to overwrite the event's `transaction`
      // value with a high-quality one
      const reqWithTransaction = req ;
      const transaction = reqWithTransaction._sentryTransaction;
      if (transaction) {
        const name = spanToJSON(transaction).description || '';

        // TODO (v8): Remove the nextjs check and just base it on `transactionNamingScheme` for all SDKs. (We have to
        // keep it the way it is for the moment, because changing the names of transactions in Sentry has the potential
        // to break things like alert rules.)
        const shouldIncludeMethodInTransactionName =
          getSDKName(client) === 'sentry.javascript.nextjs'
            ? name.startsWith('/api')
            : transactionNamingScheme !== 'path';

        const [transactionValue] = extractPathForTransaction(req, {
          path: true,
          method: shouldIncludeMethodInTransactionName,
          customRoute: name,
        });

        processedEvent.transaction = transactionValue;
      }

      return processedEvent;
    },
  };
}) ;

const requestDataIntegration = defineIntegration(_requestDataIntegration);

/**
 * Add data about a request to an event. Primarily for use in Node-based SDKs, but included in `@sentry/integrations`
 * so it can be used in cross-platform SDKs like `@sentry/nextjs`.
 * @deprecated Use `requestDataIntegration()` instead.
 */
// eslint-disable-next-line deprecation/deprecation
const RequestData = convertIntegrationFnToClass(INTEGRATION_NAME, requestDataIntegration)

;

/** Convert this integration's options to match what `addRequestDataToEvent` expects */
/** TODO: Can possibly be deleted once https://github.com/getsentry/sentry-javascript/issues/5718 is fixed */
function convertReqDataIntegrationOptsToAddReqDataOpts(
  integrationOptions,
) {
  const {
    transactionNamingScheme,
    include: { ip, user, ...requestOptions },
  } = integrationOptions;

  const requestIncludeKeys = [];
  for (const [key, value] of Object.entries(requestOptions)) {
    if (value) {
      requestIncludeKeys.push(key);
    }
  }

  let addReqDataUserOpt;
  if (user === undefined) {
    addReqDataUserOpt = true;
  } else if (typeof user === 'boolean') {
    addReqDataUserOpt = user;
  } else {
    const userIncludeKeys = [];
    for (const [key, value] of Object.entries(user)) {
      if (value) {
        userIncludeKeys.push(key);
      }
    }
    addReqDataUserOpt = userIncludeKeys;
  }

  return {
    include: {
      ip,
      user: addReqDataUserOpt,
      request: requestIncludeKeys.length !== 0 ? requestIncludeKeys : undefined,
      transaction: transactionNamingScheme,
    },
  };
}

function getSDKName(client) {
  try {
    // For a long chain like this, it's fewer bytes to combine a try-catch with assuming everything is there than to
    // write out a long chain of `a && a.b && a.b.c && ...`
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    return client.getOptions()._metadata.sdk.name;
  } catch (err) {
    // In theory we should never get here
    return undefined;
  }
}

export { RequestData, requestDataIntegration };
//# sourceMappingURL=requestdata.js.map
