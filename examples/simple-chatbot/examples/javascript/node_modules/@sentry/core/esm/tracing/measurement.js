import { getActiveTransaction } from './utils.js';

/**
 * Adds a measurement to the current active transaction.
 */
function setMeasurement(name, value, unit) {
  // eslint-disable-next-line deprecation/deprecation
  const transaction = getActiveTransaction();
  if (transaction) {
    // eslint-disable-next-line deprecation/deprecation
    transaction.setMeasurement(name, value, unit);
  }
}

export { setMeasurement };
//# sourceMappingURL=measurement.js.map
