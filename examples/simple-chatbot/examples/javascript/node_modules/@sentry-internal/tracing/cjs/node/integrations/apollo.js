var {
  _optionalChain
} = require('@sentry/utils');

Object.defineProperty(exports, '__esModule', { value: true });

const utils = require('@sentry/utils');
const debugBuild = require('../../common/debug-build.js');
const nodeUtils = require('./utils/node-utils.js');

/** Tracing integration for Apollo */
class Apollo  {
  /**
   * @inheritDoc
   */
   static __initStatic() {this.id = 'Apollo';}

  /**
   * @inheritDoc
   */

  /**
   * @inheritDoc
   */
   constructor(
    options = {
      useNestjs: false,
    },
  ) {
    this.name = Apollo.id;
    this._useNest = !!options.useNestjs;
  }

  /** @inheritdoc */
   loadDependency() {
    if (this._useNest) {
      this._module = this._module || utils.loadModule('@nestjs/graphql');
    } else {
      this._module = this._module || utils.loadModule('apollo-server-core');
    }

    return this._module;
  }

  /**
   * @inheritDoc
   */
  // eslint-disable-next-line deprecation/deprecation
   setupOnce(_, getCurrentHub) {
    if (nodeUtils.shouldDisableAutoInstrumentation(getCurrentHub)) {
      debugBuild.DEBUG_BUILD && utils.logger.log('Apollo Integration is skipped because of instrumenter configuration.');
      return;
    }

    if (this._useNest) {
      const pkg = this.loadDependency();

      if (!pkg) {
        debugBuild.DEBUG_BUILD && utils.logger.error('Apollo-NestJS Integration was unable to require @nestjs/graphql package.');
        return;
      }

      /**
       * Iterate over resolvers of NestJS ResolversExplorerService before schemas are constructed.
       */
      utils.fill(
        pkg.GraphQLFactory.prototype,
        'mergeWithSchema',
        function (orig) {
          return function (

            ...args
          ) {
            utils.fill(this.resolversExplorerService, 'explore', function (orig) {
              return function () {
                const resolvers = utils.arrayify(orig.call(this));

                const instrumentedResolvers = instrumentResolvers(resolvers, getCurrentHub);

                return instrumentedResolvers;
              };
            });

            return orig.call(this, ...args);
          };
        },
      );
    } else {
      const pkg = this.loadDependency();

      if (!pkg) {
        debugBuild.DEBUG_BUILD && utils.logger.error('Apollo Integration was unable to require apollo-server-core package.');
        return;
      }

      /**
       * Iterate over resolvers of the ApolloServer instance before schemas are constructed.
       */
      utils.fill(pkg.ApolloServerBase.prototype, 'constructSchema', function (orig) {
        return function (

) {
          if (!this.config.resolvers) {
            if (debugBuild.DEBUG_BUILD) {
              if (this.config.schema) {
                utils.logger.warn(
                  'Apollo integration is not able to trace `ApolloServer` instances constructed via `schema` property.' +
                    'If you are using NestJS with Apollo, please use `Sentry.Integrations.Apollo({ useNestjs: true })` instead.',
                );
                utils.logger.warn();
              } else if (this.config.modules) {
                utils.logger.warn(
                  'Apollo integration is not able to trace `ApolloServer` instances constructed via `modules` property.',
                );
              }

              utils.logger.error('Skipping tracing as no resolvers found on the `ApolloServer` instance.');
            }

            return orig.call(this);
          }

          const resolvers = utils.arrayify(this.config.resolvers);

          this.config.resolvers = instrumentResolvers(resolvers, getCurrentHub);

          return orig.call(this);
        };
      });
    }
  }
}Apollo.__initStatic();

// eslint-disable-next-line deprecation/deprecation
function instrumentResolvers(resolvers, getCurrentHub) {
  return resolvers.map(model => {
    Object.keys(model).forEach(resolverGroupName => {
      Object.keys(model[resolverGroupName]).forEach(resolverName => {
        if (typeof model[resolverGroupName][resolverName] !== 'function') {
          return;
        }

        wrapResolver(model, resolverGroupName, resolverName, getCurrentHub);
      });
    });

    return model;
  });
}

/**
 * Wrap a single resolver which can be a parent of other resolvers and/or db operations.
 */
function wrapResolver(
  model,
  resolverGroupName,
  resolverName,
  // eslint-disable-next-line deprecation/deprecation
  getCurrentHub,
) {
  utils.fill(model[resolverGroupName], resolverName, function (orig) {
    return function ( ...args) {
      // eslint-disable-next-line deprecation/deprecation
      const scope = getCurrentHub().getScope();
      // eslint-disable-next-line deprecation/deprecation
      const parentSpan = scope.getSpan();
      // eslint-disable-next-line deprecation/deprecation
      const span = _optionalChain([parentSpan, 'optionalAccess', _2 => _2.startChild, 'call', _3 => _3({
        description: `${resolverGroupName}.${resolverName}`,
        op: 'graphql.resolve',
        origin: 'auto.graphql.apollo',
      })]);

      const rv = orig.call(this, ...args);

      if (utils.isThenable(rv)) {
        return rv.then((res) => {
          _optionalChain([span, 'optionalAccess', _4 => _4.end, 'call', _5 => _5()]);
          return res;
        });
      }

      _optionalChain([span, 'optionalAccess', _6 => _6.end, 'call', _7 => _7()]);

      return rv;
    };
  });
}

exports.Apollo = Apollo;
//# sourceMappingURL=apollo.js.map
