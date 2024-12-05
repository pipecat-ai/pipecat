Object.defineProperty(exports, '__esModule', { value: true });

const utils = require('./utils.js');

/**
 * Adds a measurement to the current active transaction.
 */
function setMeasurement(name, value, unit) {
  // eslint-disable-next-line deprecation/deprecation
  const transaction = utils.getActiveTransaction();
  if (transaction) {
    // eslint-disable-next-line deprecation/deprecation
    transaction.setMeasurement(name, value, unit);
  }
}

exports.setMeasurement = setMeasurement;
//# sourceMappingURL=measurement.js.map
