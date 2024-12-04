Object.defineProperty(exports, '__esModule', { value: true });

const utils = require('@sentry/utils');
const integration = require('../integration.js');
const metadata = require('../metadata.js');

const INTEGRATION_NAME = 'ModuleMetadata';

const _moduleMetadataIntegration = (() => {
  return {
    name: INTEGRATION_NAME,
    // TODO v8: Remove this
    setupOnce() {}, // eslint-disable-line @typescript-eslint/no-empty-function
    setup(client) {
      if (typeof client.on !== 'function') {
        return;
      }

      // We need to strip metadata from stack frames before sending them to Sentry since these are client side only.
      client.on('beforeEnvelope', envelope => {
        utils.forEachEnvelopeItem(envelope, (item, type) => {
          if (type === 'event') {
            const event = Array.isArray(item) ? (item )[1] : undefined;

            if (event) {
              metadata.stripMetadataFromStackFrames(event);
              item[1] = event;
            }
          }
        });
      });
    },

    processEvent(event, _hint, client) {
      const stackParser = client.getOptions().stackParser;
      metadata.addMetadataToStackFrames(stackParser, event);
      return event;
    },
  };
}) ;

const moduleMetadataIntegration = integration.defineIntegration(_moduleMetadataIntegration);

/**
 * Adds module metadata to stack frames.
 *
 * Metadata can be injected by the Sentry bundler plugins using the `_experiments.moduleMetadata` config option.
 *
 * When this integration is added, the metadata passed to the bundler plugin is added to the stack frames of all events
 * under the `module_metadata` property. This can be used to help in tagging or routing of events from different teams
 * our sources
 *
 * @deprecated Use `moduleMetadataIntegration()` instead.
 */
// eslint-disable-next-line deprecation/deprecation
const ModuleMetadata = integration.convertIntegrationFnToClass(
  INTEGRATION_NAME,
  moduleMetadataIntegration,
)

;

exports.ModuleMetadata = ModuleMetadata;
exports.moduleMetadataIntegration = moduleMetadataIntegration;
//# sourceMappingURL=metadata.js.map
