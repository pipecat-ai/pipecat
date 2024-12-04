var {
  _optionalChain
} = require('@sentry/utils');

Object.defineProperty(exports, '__esModule', { value: true });

const utils = require('@sentry/utils');
const debugBuild = require('../../common/debug-build.js');
const nodeUtils = require('./utils/node-utils.js');

/** Tracing integration for graphql package */
class GraphQL  {
  /**
   * @inheritDoc
   */
   static __initStatic() {this.id = 'GraphQL';}

  /**
   * @inheritDoc
   */

   constructor() {
    this.name = GraphQL.id;
  }

  /** @inheritdoc */
   loadDependency() {
    return (this._module = this._module || utils.loadModule('graphql/execution/execute.js'));
  }

  /**
   * @inheritDoc
   */
  // eslint-disable-next-line deprecation/deprecation
   setupOnce(_, getCurrentHub) {
    if (nodeUtils.shouldDisableAutoInstrumentation(getCurrentHub)) {
      debugBuild.DEBUG_BUILD && utils.logger.log('GraphQL Integration is skipped because of instrumenter configuration.');
      return;
    }

    const pkg = this.loadDependency();

    if (!pkg) {
      debugBuild.DEBUG_BUILD && utils.logger.error('GraphQL Integration was unable to require graphql/execution package.');
      return;
    }

    utils.fill(pkg, 'execute', function (orig) {
      return function ( ...args) {
        // eslint-disable-next-line deprecation/deprecation
        const scope = getCurrentHub().getScope();
        // eslint-disable-next-line deprecation/deprecation
        const parentSpan = scope.getSpan();

        // eslint-disable-next-line deprecation/deprecation
        const span = _optionalChain([parentSpan, 'optionalAccess', _2 => _2.startChild, 'call', _3 => _3({
          description: 'execute',
          op: 'graphql.execute',
          origin: 'auto.graphql.graphql',
        })]);

        // eslint-disable-next-line deprecation/deprecation
        _optionalChain([scope, 'optionalAccess', _4 => _4.setSpan, 'call', _5 => _5(span)]);

        const rv = orig.call(this, ...args);

        if (utils.isThenable(rv)) {
          return rv.then((res) => {
            _optionalChain([span, 'optionalAccess', _6 => _6.end, 'call', _7 => _7()]);
            // eslint-disable-next-line deprecation/deprecation
            _optionalChain([scope, 'optionalAccess', _8 => _8.setSpan, 'call', _9 => _9(parentSpan)]);

            return res;
          });
        }

        _optionalChain([span, 'optionalAccess', _10 => _10.end, 'call', _11 => _11()]);
        // eslint-disable-next-line deprecation/deprecation
        _optionalChain([scope, 'optionalAccess', _12 => _12.setSpan, 'call', _13 => _13(parentSpan)]);
        return rv;
      };
    });
  }
}GraphQL.__initStatic();

exports.GraphQL = GraphQL;
//# sourceMappingURL=graphql.js.map
