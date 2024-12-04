import { _optionalChain } from '@sentry/utils';
import { loadModule, logger, fill } from '@sentry/utils';
import { DEBUG_BUILD } from '../../common/debug-build.js';
import { shouldDisableAutoInstrumentation } from './utils/node-utils.js';

/** Tracing integration for node-mysql package */
class Mysql  {
  /**
   * @inheritDoc
   */
   static __initStatic() {this.id = 'Mysql';}

  /**
   * @inheritDoc
   */

   constructor() {
    this.name = Mysql.id;
  }

  /** @inheritdoc */
   loadDependency() {
    return (this._module = this._module || loadModule('mysql/lib/Connection.js'));
  }

  /**
   * @inheritDoc
   */
  // eslint-disable-next-line deprecation/deprecation
   setupOnce(_, getCurrentHub) {
    if (shouldDisableAutoInstrumentation(getCurrentHub)) {
      DEBUG_BUILD && logger.log('Mysql Integration is skipped because of instrumenter configuration.');
      return;
    }

    const pkg = this.loadDependency();

    if (!pkg) {
      DEBUG_BUILD && logger.error('Mysql Integration was unable to require `mysql` package.');
      return;
    }

    let mySqlConfig = undefined;

    try {
      pkg.prototype.connect = new Proxy(pkg.prototype.connect, {
        apply(wrappingTarget, thisArg, args) {
          if (!mySqlConfig) {
            mySqlConfig = thisArg.config;
          }
          return wrappingTarget.apply(thisArg, args);
        },
      });
    } catch (e) {
      DEBUG_BUILD && logger.error('Mysql Integration was unable to instrument `mysql` config.');
    }

    function spanDataFromConfig() {
      if (!mySqlConfig) {
        return {};
      }
      return {
        'server.address': mySqlConfig.host,
        'server.port': mySqlConfig.port,
        'db.user': mySqlConfig.user,
      };
    }

    function finishSpan(span) {
      if (!span) {
        return;
      }

      const data = spanDataFromConfig();
      Object.keys(data).forEach(key => {
        span.setAttribute(key, data[key]);
      });

      span.end();
    }

    // The original function will have one of these signatures:
    //    function (callback) => void
    //    function (options, callback) => void
    //    function (options, values, callback) => void
    fill(pkg, 'createQuery', function (orig) {
      return function ( options, values, callback) {
        // eslint-disable-next-line deprecation/deprecation
        const scope = getCurrentHub().getScope();
        // eslint-disable-next-line deprecation/deprecation
        const parentSpan = scope.getSpan();

        // eslint-disable-next-line deprecation/deprecation
        const span = _optionalChain([parentSpan, 'optionalAccess', _2 => _2.startChild, 'call', _3 => _3({
          description: typeof options === 'string' ? options : (options ).sql,
          op: 'db',
          origin: 'auto.db.mysql',
          data: {
            'db.system': 'mysql',
          },
        })]);

        if (typeof callback === 'function') {
          return orig.call(this, options, values, function (err, result, fields) {
            finishSpan(span);
            callback(err, result, fields);
          });
        }

        if (typeof values === 'function') {
          return orig.call(this, options, function (err, result, fields) {
            finishSpan(span);
            values(err, result, fields);
          });
        }

        // streaming, no callback!
        const query = orig.call(this, options, values) ;

        query.on('end', () => {
          finishSpan(span);
        });

        return query;
      };
    });
  }
}Mysql.__initStatic();

export { Mysql };
//# sourceMappingURL=mysql.js.map
