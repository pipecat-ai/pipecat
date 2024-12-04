import { _optionalChain } from '@sentry/utils';
import { loadModule, logger, fill, isThenable } from '@sentry/utils';
import { DEBUG_BUILD } from '../../common/debug-build.js';
import { shouldDisableAutoInstrumentation } from './utils/node-utils.js';

/** Tracing integration for node-postgres package */
class Postgres  {
  /**
   * @inheritDoc
   */
   static __initStatic() {this.id = 'Postgres';}

  /**
   * @inheritDoc
   */

   constructor(options = {}) {
    this.name = Postgres.id;
    this._usePgNative = !!options.usePgNative;
    this._module = options.module;
  }

  /** @inheritdoc */
   loadDependency() {
    return (this._module = this._module || loadModule('pg'));
  }

  /**
   * @inheritDoc
   */
  // eslint-disable-next-line deprecation/deprecation
   setupOnce(_, getCurrentHub) {
    if (shouldDisableAutoInstrumentation(getCurrentHub)) {
      DEBUG_BUILD && logger.log('Postgres Integration is skipped because of instrumenter configuration.');
      return;
    }

    const pkg = this.loadDependency();

    if (!pkg) {
      DEBUG_BUILD && logger.error('Postgres Integration was unable to require `pg` package.');
      return;
    }

    const Client = this._usePgNative ? _optionalChain([pkg, 'access', _2 => _2.native, 'optionalAccess', _3 => _3.Client]) : pkg.Client;

    if (!Client) {
      DEBUG_BUILD && logger.error("Postgres Integration was unable to access 'pg-native' bindings.");
      return;
    }

    /**
     * function (query, callback) => void
     * function (query, params, callback) => void
     * function (query) => Promise
     * function (query, params) => Promise
     * function (pg.Cursor) => pg.Cursor
     */
    fill(Client.prototype, 'query', function (orig) {
      return function ( config, values, callback) {
        // eslint-disable-next-line deprecation/deprecation
        const scope = getCurrentHub().getScope();
        // eslint-disable-next-line deprecation/deprecation
        const parentSpan = scope.getSpan();

        const data = {
          'db.system': 'postgresql',
        };

        try {
          if (this.database) {
            data['db.name'] = this.database;
          }
          if (this.host) {
            data['server.address'] = this.host;
          }
          if (this.port) {
            data['server.port'] = this.port;
          }
          if (this.user) {
            data['db.user'] = this.user;
          }
        } catch (e) {
          // ignore
        }

        // eslint-disable-next-line deprecation/deprecation
        const span = _optionalChain([parentSpan, 'optionalAccess', _4 => _4.startChild, 'call', _5 => _5({
          description: typeof config === 'string' ? config : (config ).text,
          op: 'db',
          origin: 'auto.db.postgres',
          data,
        })]);

        if (typeof callback === 'function') {
          return orig.call(this, config, values, function (err, result) {
            _optionalChain([span, 'optionalAccess', _6 => _6.end, 'call', _7 => _7()]);
            callback(err, result);
          });
        }

        if (typeof values === 'function') {
          return orig.call(this, config, function (err, result) {
            _optionalChain([span, 'optionalAccess', _8 => _8.end, 'call', _9 => _9()]);
            values(err, result);
          });
        }

        const rv = typeof values !== 'undefined' ? orig.call(this, config, values) : orig.call(this, config);

        if (isThenable(rv)) {
          return rv.then((res) => {
            _optionalChain([span, 'optionalAccess', _10 => _10.end, 'call', _11 => _11()]);
            return res;
          });
        }

        _optionalChain([span, 'optionalAccess', _12 => _12.end, 'call', _13 => _13()]);
        return rv;
      };
    });
  }
}Postgres.__initStatic();

export { Postgres };
//# sourceMappingURL=postgres.js.map
