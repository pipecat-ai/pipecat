Object.defineProperty(exports, '__esModule', { value: true });

const utils = require('@sentry/utils');
const localForage = require('localforage');
const debugBuild = require('./debug-build.js');

const WINDOW = utils.GLOBAL_OBJ ;

/**
 * cache offline errors and send when connected
 * @deprecated The offline integration has been deprecated in favor of the offline transport wrapper.
 *
 * http://docs.sentry.io/platforms/javascript/configuration/transports/#offline-caching
 */
class Offline  {
  /**
   * @inheritDoc
   */
   static __initStatic() {this.id = 'Offline';}

  /**
   * @inheritDoc
   */

  /**
   * the current hub instance
   */

  /**
   * maximum number of events to store while offline
   */

  /**
   * event cache
   */

  /**
   * @inheritDoc
   */
   constructor(options = {}) {
    this.name = Offline.id;

    this.maxStoredEvents = options.maxStoredEvents || 30; // set a reasonable default
    // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
    this.offlineEventStore = localForage.createInstance({
      name: 'sentry/offlineEventStore',
    });
  }

  /**
   * @inheritDoc
   */
   setupOnce(addGlobalEventProcessor, getCurrentHub) {
    this.hub = getCurrentHub();

    if ('addEventListener' in WINDOW) {
      WINDOW.addEventListener('online', () => {
        void this._sendEvents().catch(() => {
          debugBuild.DEBUG_BUILD && utils.logger.warn('could not send cached events');
        });
      });
    }

    const eventProcessor = event => {
      // eslint-disable-next-line deprecation/deprecation
      if (this.hub && this.hub.getIntegration(Offline)) {
        // cache if we are positively offline
        if ('navigator' in WINDOW && 'onLine' in WINDOW.navigator && !WINDOW.navigator.onLine) {
          debugBuild.DEBUG_BUILD && utils.logger.log('Event dropped due to being a offline - caching instead');

          void this._cacheEvent(event)
            .then((_event) => this._enforceMaxEvents())
            .catch((_error) => {
              debugBuild.DEBUG_BUILD && utils.logger.warn('could not cache event while offline');
            });

          // return null on success or failure, because being offline will still result in an error
          return null;
        }
      }

      return event;
    };

    eventProcessor.id = this.name;
    addGlobalEventProcessor(eventProcessor);

    // if online now, send any events stored in a previous offline session
    if ('navigator' in WINDOW && 'onLine' in WINDOW.navigator && WINDOW.navigator.onLine) {
      void this._sendEvents().catch(() => {
        debugBuild.DEBUG_BUILD && utils.logger.warn('could not send cached events');
      });
    }
  }

  /**
   * cache an event to send later
   * @param event an event
   */
   async _cacheEvent(event) {
    return this.offlineEventStore.setItem(utils.uuid4(), utils.normalize(event));
  }

  /**
   * purge excess events if necessary
   */
   async _enforceMaxEvents() {
    const events = [];

    return this.offlineEventStore
      .iterate((event, cacheKey, _index) => {
        // aggregate events
        events.push({ cacheKey, event });
      })
      .then(
        () =>
          // this promise resolves when the iteration is finished
          this._purgeEvents(
            // purge all events past maxStoredEvents in reverse chronological order
            events
              .sort((a, b) => (b.event.timestamp || 0) - (a.event.timestamp || 0))
              .slice(this.maxStoredEvents < events.length ? this.maxStoredEvents : events.length)
              .map(event => event.cacheKey),
          ),
      )
      .catch((_error) => {
        debugBuild.DEBUG_BUILD && utils.logger.warn('could not enforce max events');
      });
  }

  /**
   * purge event from cache
   */
   async _purgeEvent(cacheKey) {
    return this.offlineEventStore.removeItem(cacheKey);
  }

  /**
   * purge events from cache
   */
   async _purgeEvents(cacheKeys) {
    // trail with .then to ensure the return type as void and not void|void[]
    return Promise.all(cacheKeys.map(cacheKey => this._purgeEvent(cacheKey))).then();
  }

  /**
   * send all events
   */
   async _sendEvents() {
    return this.offlineEventStore.iterate((event, cacheKey, _index) => {
      if (this.hub) {
        this.hub.captureEvent(event);

        void this._purgeEvent(cacheKey).catch((_error) => {
          debugBuild.DEBUG_BUILD && utils.logger.warn('could not purge event from cache');
        });
      } else {
        debugBuild.DEBUG_BUILD && utils.logger.warn('no hub found - could not send cached event');
      }
    });
  }
} Offline.__initStatic();

exports.Offline = Offline;
//# sourceMappingURL=offline.js.map
