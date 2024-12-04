Object.defineProperty(exports, '__esModule', { value: true });

const core = require('@sentry/core');
const utils = require('@sentry/utils');
const debugBuild = require('../../common/debug-build.js');
const nodeUtils = require('./utils/node-utils.js');

function isValidPrismaClient(possibleClient) {
  return !!possibleClient && !!(possibleClient )['$use'];
}

/** Tracing integration for @prisma/client package */
class Prisma  {
  /**
   * @inheritDoc
   */
   static __initStatic() {this.id = 'Prisma';}

  /**
   * @inheritDoc
   */

  /**
   * @inheritDoc
   */
   constructor(options = {}) {
    this.name = Prisma.id;

    // We instrument the PrismaClient inside the constructor and not inside `setupOnce` because in some cases of server-side
    // bundling (Next.js) multiple Prisma clients can be instantiated, even though users don't intend to. When instrumenting
    // in setupOnce we can only ever instrument one client.
    // https://github.com/getsentry/sentry-javascript/issues/7216#issuecomment-1602375012
    // In the future we might explore providing a dedicated PrismaClient middleware instead of this hack.
    if (isValidPrismaClient(options.client) && !options.client._sentryInstrumented) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      utils.addNonEnumerableProperty(options.client , '_sentryInstrumented', true);

      const clientData = {};
      try {
        const engineConfig = (options.client )._engineConfig;
        if (engineConfig) {
          const { activeProvider, clientVersion } = engineConfig;
          if (activeProvider) {
            clientData['db.system'] = activeProvider;
          }
          if (clientVersion) {
            clientData['db.prisma.version'] = clientVersion;
          }
        }
      } catch (e) {
        // ignore
      }

      options.client.$use((params, next) => {
        // eslint-disable-next-line deprecation/deprecation
        if (nodeUtils.shouldDisableAutoInstrumentation(core.getCurrentHub)) {
          return next(params);
        }

        const action = params.action;
        const model = params.model;

        return core.startSpan(
          {
            name: model ? `${model} ${action}` : action,
            onlyIfParent: true,
            op: 'db.prisma',
            attributes: {
              [core.SEMANTIC_ATTRIBUTE_SENTRY_ORIGIN]: 'auto.db.prisma',
            },
            data: { ...clientData, 'db.operation': action },
          },
          () => next(params),
        );
      });
    } else {
      debugBuild.DEBUG_BUILD &&
        utils.logger.warn('Unsupported Prisma client provided to PrismaIntegration. Provided client:', options.client);
    }
  }

  /**
   * @inheritDoc
   */
   setupOnce() {
    // Noop - here for backwards compatibility
  }
} Prisma.__initStatic();

exports.Prisma = Prisma;
//# sourceMappingURL=prisma.js.map
