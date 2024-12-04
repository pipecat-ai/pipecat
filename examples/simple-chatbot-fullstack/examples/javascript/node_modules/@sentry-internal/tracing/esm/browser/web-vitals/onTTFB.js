import { WINDOW } from '../types.js';
import { bindReporter } from './lib/bindReporter.js';
import { getActivationStart } from './lib/getActivationStart.js';
import { getNavigationEntry } from './lib/getNavigationEntry.js';
import { initMetric } from './lib/initMetric.js';

/*
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Runs in the next task after the page is done loading and/or prerendering.
 * @param callback
 */
const whenReady = (callback) => {
  if (!WINDOW.document) {
    return;
  }

  if (WINDOW.document.prerendering) {
    addEventListener('prerenderingchange', () => whenReady(callback), true);
  } else if (WINDOW.document.readyState !== 'complete') {
    addEventListener('load', () => whenReady(callback), true);
  } else {
    // Queue a task so the callback runs after `loadEventEnd`.
    setTimeout(callback, 0);
  }
};

/**
 * Calculates the [TTFB](https://web.dev/time-to-first-byte/) value for the
 * current page and calls the `callback` function once the page has loaded,
 * along with the relevant `navigation` performance entry used to determine the
 * value. The reported value is a `DOMHighResTimeStamp`.
 *
 * Note, this function waits until after the page is loaded to call `callback`
 * in order to ensure all properties of the `navigation` entry are populated.
 * This is useful if you want to report on other metrics exposed by the
 * [Navigation Timing API](https://w3c.github.io/navigation-timing/). For
 * example, the TTFB metric starts from the page's [time
 * origin](https://www.w3.org/TR/hr-time-2/#sec-time-origin), which means it
 * includes time spent on DNS lookup, connection negotiation, network latency,
 * and server processing time.
 */
const onTTFB = (onReport, opts) => {
  // Set defaults
  // eslint-disable-next-line no-param-reassign
  opts = opts || {};

  // https://web.dev/ttfb/#what-is-a-good-ttfb-score
  // const thresholds = [800, 1800];

  const metric = initMetric('TTFB');
  const report = bindReporter(onReport, metric, opts.reportAllChanges);

  whenReady(() => {
    const navEntry = getNavigationEntry() ;

    if (navEntry) {
      // The activationStart reference is used because TTFB should be
      // relative to page activation rather than navigation start if the
      // page was prerendered. But in cases where `activationStart` occurs
      // after the first byte is received, this time should be clamped at 0.
      metric.value = Math.max(navEntry.responseStart - getActivationStart(), 0);

      // In some cases the value reported is negative or is larger
      // than the current page time. Ignore these cases:
      // https://github.com/GoogleChrome/web-vitals/issues/137
      // https://github.com/GoogleChrome/web-vitals/issues/162
      if (metric.value < 0 || metric.value > performance.now()) return;

      metric.entries = [navEntry];

      report(true);
    }
  });
};

export { onTTFB };
//# sourceMappingURL=onTTFB.js.map
