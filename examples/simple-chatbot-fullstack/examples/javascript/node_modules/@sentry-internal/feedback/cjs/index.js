Object.defineProperty(exports, '__esModule', { value: true });

const utils = require('@sentry/utils');
const core = require('@sentry/core');

// exporting a separate copy of `WINDOW` rather than exporting the one from `@sentry/browser`
// prevents the browser package from being bundled in the CDN bundle, and avoids a
// circular dependency between the browser and feedback packages
const WINDOW = utils.GLOBAL_OBJ ;

const LIGHT_BACKGROUND = '#ffffff';
const INHERIT = 'inherit';
const SUBMIT_COLOR = 'rgba(108, 95, 199, 1)';
const LIGHT_THEME = {
  fontFamily: "system-ui, 'Helvetica Neue', Arial, sans-serif",
  fontSize: '14px',

  background: LIGHT_BACKGROUND,
  backgroundHover: '#f6f6f7',
  foreground: '#2b2233',
  border: '1.5px solid rgba(41, 35, 47, 0.13)',
  borderRadius: '25px',
  boxShadow: '0px 4px 24px 0px rgba(43, 34, 51, 0.12)',

  success: '#268d75',
  error: '#df3338',

  submitBackground: 'rgba(88, 74, 192, 1)',
  submitBackgroundHover: SUBMIT_COLOR,
  submitBorder: SUBMIT_COLOR,
  submitOutlineFocus: '#29232f',
  submitForeground: LIGHT_BACKGROUND,
  submitForegroundHover: LIGHT_BACKGROUND,

  cancelBackground: 'transparent',
  cancelBackgroundHover: 'var(--background-hover)',
  cancelBorder: 'var(--border)',
  cancelOutlineFocus: 'var(--input-outline-focus)',
  cancelForeground: 'var(--foreground)',
  cancelForegroundHover: 'var(--foreground)',

  inputBackground: INHERIT,
  inputForeground: INHERIT,
  inputBorder: 'var(--border)',
  inputOutlineFocus: SUBMIT_COLOR,

  formBorderRadius: '20px',
  formContentBorderRadius: '6px',
};

const DEFAULT_THEME = {
  light: LIGHT_THEME,
  dark: {
    ...LIGHT_THEME,

    background: '#29232f',
    backgroundHover: '#352f3b',
    foreground: '#ebe6ef',
    border: '1.5px solid rgba(235, 230, 239, 0.15)',

    success: '#2da98c',
    error: '#f55459',
  },
};

const ACTOR_LABEL = 'Report a Bug';
const CANCEL_BUTTON_LABEL = 'Cancel';
const SUBMIT_BUTTON_LABEL = 'Send Bug Report';
const FORM_TITLE = 'Report a Bug';
const EMAIL_PLACEHOLDER = 'your.email@example.org';
const EMAIL_LABEL = 'Email';
const MESSAGE_PLACEHOLDER = "What's the bug? What did you expect?";
const MESSAGE_LABEL = 'Description';
const NAME_PLACEHOLDER = 'Your Name';
const NAME_LABEL = 'Name';
const IS_REQUIRED_LABEL = '(required)';
const SUCCESS_MESSAGE_TEXT = 'Thank you for your report!';

const FEEDBACK_WIDGET_SOURCE = 'widget';
const FEEDBACK_API_SOURCE = 'api';

/**
 * Prepare a feedback event & enrich it with the SDK metadata.
 */
async function prepareFeedbackEvent({
  client,
  scope,
  event,
}) {
  const eventHint = {};
  if (client.emit) {
    client.emit('preprocessEvent', event, eventHint);
  }

  const preparedEvent = (await core.prepareEvent(
    client.getOptions(),
    event,
    eventHint,
    scope,
    client,
    core.getIsolationScope(),
  )) ;

  if (preparedEvent === null) {
    // Taken from baseclient's `_processEvent` method, where this is handled for errors/transactions
    client.recordDroppedEvent('event_processor', 'feedback', event);
    return null;
  }

  // This normally happens in browser client "_prepareEvent"
  // but since we do not use this private method from the client, but rather the plain import
  // we need to do this manually.
  preparedEvent.platform = preparedEvent.platform || 'javascript';

  return preparedEvent;
}

/**
 * Send feedback using transport
 */
async function sendFeedbackRequest(
  { feedback: { message, email, name, source, url } },
  { includeReplay = true } = {},
) {
  const client = core.getClient();
  const transport = client && client.getTransport();
  const dsn = client && client.getDsn();

  if (!client || !transport || !dsn) {
    return;
  }

  const baseEvent = {
    contexts: {
      feedback: {
        contact_email: email,
        name,
        message,
        url,
        source,
      },
    },
    type: 'feedback',
  };

  return core.withScope(async scope => {
    // No use for breadcrumbs in feedback
    scope.clearBreadcrumbs();

    if ([FEEDBACK_API_SOURCE, FEEDBACK_WIDGET_SOURCE].includes(String(source))) {
      scope.setLevel('info');
    }

    const feedbackEvent = await prepareFeedbackEvent({
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      scope: scope ,
      client,
      event: baseEvent,
    });

    if (!feedbackEvent) {
      return;
    }

    if (client.emit) {
      client.emit('beforeSendFeedback', feedbackEvent, { includeReplay: Boolean(includeReplay) });
    }

    const envelope = core.createEventEnvelope(feedbackEvent, dsn, client.getOptions()._metadata, client.getOptions().tunnel);

    let response;

    try {
      response = await transport.send(envelope);
    } catch (err) {
      const error = new Error('Unable to send Feedback');

      try {
        // In case browsers don't allow this property to be writable
        // @ts-expect-error This needs lib es2022 and newer
        error.cause = err;
      } catch (e) {
        // nothing to do
      }
      throw error;
    }

    // TODO (v8): we can remove this guard once transport.send's type signature doesn't include void anymore
    if (!response) {
      return;
    }

    // Require valid status codes, otherwise can assume feedback was not sent successfully
    if (typeof response.statusCode === 'number' && (response.statusCode < 200 || response.statusCode >= 300)) {
      throw new Error('Unable to send Feedback');
    }

    return response;
  });
}

/*
 * For reference, the fully built event looks something like this:
 * {
 *     "type": "feedback",
 *     "event_id": "d2132d31b39445f1938d7e21b6bf0ec4",
 *     "timestamp": 1597977777.6189718,
 *     "dist": "1.12",
 *     "platform": "javascript",
 *     "environment": "production",
 *     "release": 42,
 *     "tags": {"transaction": "/organizations/:orgId/performance/:eventSlug/"},
 *     "sdk": {"name": "name", "version": "version"},
 *     "user": {
 *         "id": "123",
 *         "username": "user",
 *         "email": "user@site.com",
 *         "ip_address": "192.168.11.12",
 *     },
 *     "request": {
 *         "url": None,
 *         "headers": {
 *             "user-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.5 Safari/605.1.15"
 *         },
 *     },
 *     "contexts": {
 *         "feedback": {
 *             "message": "test message",
 *             "contact_email": "test@example.com",
 *             "type": "feedback",
 *         },
 *         "trace": {
 *             "trace_id": "4C79F60C11214EB38604F4AE0781BFB2",
 *             "span_id": "FA90FDEAD5F74052",
 *             "type": "trace",
 *         },
 *         "replay": {
 *             "replay_id": "e2d42047b1c5431c8cba85ee2a8ab25d",
 *         },
 *     },
 *   }
 */

/**
 * Public API to send a Feedback item to Sentry
 */
function sendFeedback(
  { name, email, message, source = FEEDBACK_API_SOURCE, url = utils.getLocationHref() },
  options = {},
) {
  if (!message) {
    throw new Error('Unable to submit feedback with empty message');
  }

  return sendFeedbackRequest(
    {
      feedback: {
        name,
        email,
        message,
        url,
        source,
      },
    },
    options,
  );
}

/**
 * This serves as a build time flag that will be true by default, but false in non-debug builds or if users replace `__SENTRY_DEBUG__` in their generated code.
 *
 * ATTENTION: This constant must never cross package boundaries (i.e. be exported) to guarantee that it can be used for tree shaking.
 */
const DEBUG_BUILD = (typeof __SENTRY_DEBUG__ === 'undefined' || __SENTRY_DEBUG__);

/**
 * Quick and dirty deep merge for the Feedback integration options
 */
function mergeOptions(
  defaultOptions,
  optionOverrides,
) {
  return {
    ...defaultOptions,
    ...optionOverrides,
    themeDark: {
      ...defaultOptions.themeDark,
      ...optionOverrides.themeDark,
    },
    themeLight: {
      ...defaultOptions.themeLight,
      ...optionOverrides.themeLight,
    },
  };
}

/**
 * Creates <style> element for widget actor (button that opens the dialog)
 */
function createActorStyles(d) {
  const style = d.createElement('style');
  style.textContent = `
.widget__actor {
  position: fixed;
  left: var(--left);
  right: var(--right);
  bottom: var(--bottom);
  top: var(--top);
  z-index: var(--z-index);

  line-height: 16px;

  display: flex;
  align-items: center;
  gap: 8px;

  border-radius: var(--border-radius);
  cursor: pointer;
  font-family: inherit;
  font-size: var(--font-size);
  font-weight: 600;
  padding: 16px;
  text-decoration: none;
  z-index: 9000;

  color: var(--foreground);
  background-color: var(--background);
  border: var(--border);
  box-shadow: var(--box-shadow);
  opacity: 1;
  transition: opacity 0.1s ease-in-out;
}

.widget__actor:hover {
  background-color: var(--background-hover);
}

.widget__actor svg {
  width: 16px;
  height: 16px;
}

.widget__actor--hidden {
  opacity: 0;
  pointer-events: none;
  visibility: hidden;
}

.widget__actor__text {
}

@media (max-width: 600px) {
  .widget__actor__text {
    display: none;
  }
}

.feedback-icon path {
  fill: var(--foreground);
}
`;

  return style;
}

/**
 * Creates <style> element for widget dialog
 */
function createDialogStyles(d) {
  const style = d.createElement('style');

  style.textContent = `
.dialog {
  line-height: 25px;
  background-color: rgba(0, 0, 0, 0.05);
  border: none;
  position: fixed;
  inset: 0;
  z-index: 10000;
  width: 100vw;
  height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 1;
  transition: opacity 0.2s ease-in-out;
}

.dialog:not([open]) {
  opacity: 0;
  pointer-events: none;
  visibility: hidden;
}
.dialog:not([open]) .dialog__content {
  transform: translate(0, -16px) scale(0.98);
}

.dialog__content {
  position: fixed;
  left: var(--left);
  right: var(--right);
  bottom: var(--bottom);
  top: var(--top);

  border: var(--border);
  border-radius: var(--form-border-radius);
  background-color: var(--background);
  color: var(--foreground);

  width: 320px;
  max-width: 100%;
  max-height: calc(100% - 2rem);
  display: flex;
  flex-direction: column;
  box-shadow: var(--box-shadow);
  transition: transform 0.2s ease-in-out;
  transform: translate(0, 0) scale(1);
}

.dialog__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 20px;
  font-weight: 600;
  padding: 24px 24px 0 24px;
  margin: 0;
  margin-bottom: 16px;
}

.brand-link {
  display: inline-flex;
}

.error {
  color: var(--error);
  margin-bottom: 16px;
}

.form {
  display: grid;
  overflow: auto;
  flex-direction: column;
  gap: 16px;
  padding: 0 24px 24px;
}

.form__error-container {
  color: var(--error);
}

.form__error-container--hidden {
  display: none;
}

.form__label {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin: 0px;
}

.form__label__text {
  display: grid;
  gap: 4px;
  align-items: center;
  grid-auto-flow: column;
  grid-auto-columns: max-content;
}

.form__label__text--required {
  font-size: 0.85em;
}

.form__input {
  line-height: inherit;
  background-color: var(--input-background);
  box-sizing: border-box;
  border: var(--input-border);
  border-radius: var(--form-content-border-radius);
  color: var(--input-foreground);
  font-family: inherit;
  font-size: var(--font-size);
  font-weight: 500;
  padding: 6px 12px;
}

.form__input::placeholder {
  color: var(--input-foreground);
  opacity: 0.65;
}

.form__input:focus-visible {
  outline: 1px auto var(--input-outline-focus);
}

.form__input--textarea {
  font-family: inherit;
  resize: vertical;
}

.btn-group {
  display: grid;
  gap: 8px;
  margin-top: 8px;
}

.btn {
  line-height: inherit;
  border: var(--cancel-border);
  border-radius: var(--form-content-border-radius);
  cursor: pointer;
  font-family: inherit;
  font-size: var(--font-size);
  font-weight: 600;
  padding: 6px 16px;
}
.btn[disabled] {
  opacity: 0.6;
  pointer-events: none;
}

.btn--primary {
  background-color: var(--submit-background);
  border-color: var(--submit-border);
  color: var(--submit-foreground);
}
.btn--primary:hover {
  background-color: var(--submit-background-hover);
  color: var(--submit-foreground-hover);
}
.btn--primary:focus-visible {
  outline: 1px auto var(--submit-outline-focus);
}

.btn--default {
  background-color: var(--cancel-background);
  color: var(--cancel-foreground);
  font-weight: 500;
}
.btn--default:hover {
  background-color: var(--cancel-background-hover);
  color: var(--cancel-foreground-hover);
}
.btn--default:focus-visible {
  outline: 1px auto var(--cancel-outline-focus);
}

.success-message {
  background-color: var(--background);
  border: var(--border);
  border-radius: var(--border-radius);
  box-shadow: var(--box-shadow);
  font-weight: 600;
  color: var(--success);
  padding: 12px 24px;
  line-height: 25px;
  display: grid;
  align-items: center;
  grid-auto-flow: column;
  gap: 6px;
  cursor: default;
}

.success-icon path {
  fill: var(--success);
}
`;

  return style;
}

function getThemedCssVariables(theme) {
  return `
  --background: ${theme.background};
  --background-hover: ${theme.backgroundHover};
  --foreground: ${theme.foreground};
  --error: ${theme.error};
  --success: ${theme.success};
  --border: ${theme.border};
  --border-radius: ${theme.borderRadius};
  --box-shadow: ${theme.boxShadow};

  --submit-background: ${theme.submitBackground};
  --submit-background-hover: ${theme.submitBackgroundHover};
  --submit-border: ${theme.submitBorder};
  --submit-outline-focus: ${theme.submitOutlineFocus};
  --submit-foreground: ${theme.submitForeground};
  --submit-foreground-hover: ${theme.submitForegroundHover};

  --cancel-background: ${theme.cancelBackground};
  --cancel-background-hover: ${theme.cancelBackgroundHover};
  --cancel-border: ${theme.cancelBorder};
  --cancel-outline-focus: ${theme.cancelOutlineFocus};
  --cancel-foreground: ${theme.cancelForeground};
  --cancel-foreground-hover: ${theme.cancelForegroundHover};

  --input-background: ${theme.inputBackground};
  --input-foreground: ${theme.inputForeground};
  --input-border: ${theme.inputBorder};
  --input-outline-focus: ${theme.inputOutlineFocus};

  --form-border-radius: ${theme.formBorderRadius};
  --form-content-border-radius: ${theme.formContentBorderRadius};
  `;
}

/**
 * Creates <style> element for widget actor (button that opens the dialog)
 */
function createMainStyles(
  d,
  colorScheme,
  themes,
) {
  const style = d.createElement('style');
  style.textContent = `
:host {
  --bottom: 1rem;
  --right: 1rem;
  --top: auto;
  --left: auto;
  --z-index: 100000;
  --font-family: ${themes.light.fontFamily};
  --font-size: ${themes.light.fontSize};

  position: fixed;
  left: var(--left);
  right: var(--right);
  bottom: var(--bottom);
  top: var(--top);
  z-index: var(--z-index);

  font-family: var(--font-family);
  font-size: var(--font-size);

  ${getThemedCssVariables(colorScheme === 'dark' ? themes.dark : themes.light)}
}

${
  colorScheme === 'system'
    ? `
@media (prefers-color-scheme: dark) {
  :host {
    ${getThemedCssVariables(themes.dark)}
  }
}`
    : ''
}
}`;

  return style;
}

/**
 * Creates shadow host
 */
function createShadowHost({ id, colorScheme, themeDark, themeLight })

 {
  try {
    const doc = WINDOW.document;

    // Create the host
    const host = doc.createElement('div');
    host.id = id;

    // Create the shadow root
    const shadow = host.attachShadow({ mode: 'open' });

    shadow.appendChild(createMainStyles(doc, colorScheme, { dark: themeDark, light: themeLight }));
    shadow.appendChild(createDialogStyles(doc));

    return { shadow, host };
  } catch (e) {
    // Shadow DOM probably not supported
    utils.logger.warn('[Feedback] Browser does not support shadow DOM API');
    throw new Error('Browser does not support shadow DOM API.');
  }
}

/**
 * Handles UI behavior of dialog when feedback is submitted, calls
 * `sendFeedback` to send feedback.
 */
async function handleFeedbackSubmit(
  dialog,
  feedback,
  options,
) {
  if (!dialog) {
    // Not sure when this would happen
    return;
  }

  const showFetchError = () => {
    if (!dialog) {
      return;
    }
    dialog.showError('There was a problem submitting feedback, please wait and try again.');
  };

  dialog.hideError();

  try {
    const resp = await sendFeedback({ ...feedback, source: FEEDBACK_WIDGET_SOURCE }, options);

    // Success!
    return resp;
  } catch (err) {
    DEBUG_BUILD && utils.logger.error(err);
    showFetchError();
  }
}

/**
 * Helper function to set a dict of attributes on element (w/ specified namespace)
 */
function setAttributesNS(el, attributes) {
  Object.entries(attributes).forEach(([key, val]) => {
    el.setAttributeNS(null, key, val);
  });
  return el;
}

const SIZE = 20;
const XMLNS$2 = 'http://www.w3.org/2000/svg';

/**
 * Feedback Icon
 */
function Icon() {
  const createElementNS = (tagName) =>
    WINDOW.document.createElementNS(XMLNS$2, tagName);
  const svg = setAttributesNS(createElementNS('svg'), {
    class: 'feedback-icon',
    width: `${SIZE}`,
    height: `${SIZE}`,
    viewBox: `0 0 ${SIZE} ${SIZE}`,
    fill: 'none',
  });

  const g = setAttributesNS(createElementNS('g'), {
    clipPath: 'url(#clip0_57_80)',
  });

  const path = setAttributesNS(createElementNS('path'), {
    ['fill-rule']: 'evenodd',
    ['clip-rule']: 'evenodd',
    d: 'M15.6622 15H12.3997C12.2129 14.9959 12.031 14.9396 11.8747 14.8375L8.04965 12.2H7.49956V19.1C7.4875 19.3348 7.3888 19.5568 7.22256 19.723C7.05632 19.8892 6.83435 19.9879 6.59956 20H2.04956C1.80193 19.9968 1.56535 19.8969 1.39023 19.7218C1.21511 19.5467 1.1153 19.3101 1.11206 19.0625V12.2H0.949652C0.824431 12.2017 0.700142 12.1783 0.584123 12.1311C0.468104 12.084 0.362708 12.014 0.274155 11.9255C0.185602 11.8369 0.115689 11.7315 0.0685419 11.6155C0.0213952 11.4995 -0.00202913 11.3752 -0.00034808 11.25V3.75C-0.00900498 3.62067 0.0092504 3.49095 0.0532651 3.36904C0.0972798 3.24712 0.166097 3.13566 0.255372 3.04168C0.344646 2.94771 0.452437 2.87327 0.571937 2.82307C0.691437 2.77286 0.82005 2.74798 0.949652 2.75H8.04965L11.8747 0.1625C12.031 0.0603649 12.2129 0.00407221 12.3997 0H15.6622C15.9098 0.00323746 16.1464 0.103049 16.3215 0.278167C16.4966 0.453286 16.5964 0.689866 16.5997 0.9375V3.25269C17.3969 3.42959 18.1345 3.83026 18.7211 4.41679C19.5322 5.22788 19.9878 6.32796 19.9878 7.47502C19.9878 8.62209 19.5322 9.72217 18.7211 10.5333C18.1345 11.1198 17.3969 11.5205 16.5997 11.6974V14.0125C16.6047 14.1393 16.5842 14.2659 16.5395 14.3847C16.4948 14.5035 16.4268 14.6121 16.3394 14.7042C16.252 14.7962 16.147 14.8698 16.0307 14.9206C15.9144 14.9714 15.7891 14.9984 15.6622 15ZM1.89695 10.325H1.88715V4.625H8.33715C8.52423 4.62301 8.70666 4.56654 8.86215 4.4625L12.6872 1.875H14.7247V13.125H12.6872L8.86215 10.4875C8.70666 10.3835 8.52423 10.327 8.33715 10.325H2.20217C2.15205 10.3167 2.10102 10.3125 2.04956 10.3125C1.9981 10.3125 1.94708 10.3167 1.89695 10.325ZM2.98706 12.2V18.1625H5.66206V12.2H2.98706ZM16.5997 9.93612V5.01393C16.6536 5.02355 16.7072 5.03495 16.7605 5.04814C17.1202 5.13709 17.4556 5.30487 17.7425 5.53934C18.0293 5.77381 18.2605 6.06912 18.4192 6.40389C18.578 6.73866 18.6603 7.10452 18.6603 7.47502C18.6603 7.84552 18.578 8.21139 18.4192 8.54616C18.2605 8.88093 18.0293 9.17624 17.7425 9.41071C17.4556 9.64518 17.1202 9.81296 16.7605 9.90191C16.7072 9.91509 16.6536 9.9265 16.5997 9.93612Z',
  });
  svg.appendChild(g).appendChild(path);

  const speakerDefs = createElementNS('defs');
  const speakerClipPathDef = setAttributesNS(createElementNS('clipPath'), {
    id: 'clip0_57_80',
  });

  const speakerRect = setAttributesNS(createElementNS('rect'), {
    width: `${SIZE}`,
    height: `${SIZE}`,
    fill: 'white',
  });

  speakerClipPathDef.appendChild(speakerRect);
  speakerDefs.appendChild(speakerClipPathDef);

  svg.appendChild(speakerDefs).appendChild(speakerClipPathDef).appendChild(speakerRect);

  return {
    get el() {
      return svg;
    },
  };
}

/**
 * Helper function to create an element. Could be used as a JSX factory
 * (i.e. React-like syntax).
 */
function createElement(
  tagName,
  attributes,
  ...children
) {
  const doc = WINDOW.document;
  const element = doc.createElement(tagName);

  if (attributes) {
    Object.entries(attributes).forEach(([attribute, attributeValue]) => {
      if (attribute === 'className' && typeof attributeValue === 'string') {
        // JSX does not allow class as a valid name
        element.setAttribute('class', attributeValue);
      } else if (typeof attributeValue === 'boolean' && attributeValue) {
        element.setAttribute(attribute, '');
      } else if (typeof attributeValue === 'string') {
        element.setAttribute(attribute, attributeValue);
      } else if (attribute.startsWith('on') && typeof attributeValue === 'function') {
        element.addEventListener(attribute.substring(2).toLowerCase(), attributeValue);
      }
    });
  }
  for (const child of children) {
    appendChild(element, child);
  }

  return element;
}

function appendChild(parent, child) {
  const doc = WINDOW.document;
  if (typeof child === 'undefined' || child === null) {
    return;
  }

  if (Array.isArray(child)) {
    for (const value of child) {
      appendChild(parent, value);
    }
  } else if (child === false) ; else if (typeof child === 'string') {
    parent.appendChild(doc.createTextNode(child));
  } else if (child instanceof Node) {
    parent.appendChild(child);
  } else {
    parent.appendChild(doc.createTextNode(String(child)));
  }
}

/**
 *
 */
function Actor({ buttonLabel, onClick }) {
  function _handleClick(e) {
    onClick && onClick(e);
  }

  const el = createElement(
    'button',
    {
      type: 'button',
      className: 'widget__actor',
      ['aria-label']: buttonLabel,
      ['aria-hidden']: 'false',
    },
    Icon().el,
    buttonLabel
      ? createElement(
          'span',
          {
            className: 'widget__actor__text',
          },
          buttonLabel,
        )
      : null,
  );

  el.addEventListener('click', _handleClick);

  return {
    get el() {
      return el;
    },
    show: () => {
      el.classList.remove('widget__actor--hidden');
      el.setAttribute('aria-hidden', 'false');
    },
    hide: () => {
      el.classList.add('widget__actor--hidden');
      el.setAttribute('aria-hidden', 'true');
    },
  };
}

/**
 *
 */
function SubmitButton({ label }) {
  const el = createElement(
    'button',
    {
      type: 'submit',
      className: 'btn btn--primary',
      ['aria-label']: label,
    },
    label,
  );

  return {
    el,
  };
}

function retrieveStringValue(formData, key) {
  const value = formData.get(key);
  if (typeof value === 'string') {
    return value.trim();
  }
  return '';
}

/**
 * Creates the form element
 */
function Form({
  nameLabel,
  namePlaceholder,
  emailLabel,
  emailPlaceholder,
  messageLabel,
  messagePlaceholder,
  isRequiredLabel,
  cancelButtonLabel,
  submitButtonLabel,

  showName,
  showEmail,
  isNameRequired,
  isEmailRequired,

  defaultName,
  defaultEmail,
  onCancel,
  onSubmit,
}) {
  const { el: submitEl } = SubmitButton({
    label: submitButtonLabel,
  });

  function handleSubmit(e) {
    e.preventDefault();

    if (!(e.target instanceof HTMLFormElement)) {
      return;
    }

    try {
      if (onSubmit) {
        const formData = new FormData(e.target );
        const feedback = {
          name: retrieveStringValue(formData, 'name'),
          email: retrieveStringValue(formData, 'email'),
          message: retrieveStringValue(formData, 'message'),
        };

        onSubmit(feedback);
      }
    } catch (e2) {
      // pass
    }
  }

  const errorEl = createElement('div', {
    className: 'form__error-container form__error-container--hidden',
    ['aria-hidden']: 'true',
  });

  function showError(message) {
    errorEl.textContent = message;
    errorEl.classList.remove('form__error-container--hidden');
    errorEl.setAttribute('aria-hidden', 'false');
  }

  function hideError() {
    errorEl.textContent = '';
    errorEl.classList.add('form__error-container--hidden');
    errorEl.setAttribute('aria-hidden', 'true');
  }

  const nameEl = createElement('input', {
    id: 'name',
    type: showName ? 'text' : 'hidden',
    ['aria-hidden']: showName ? 'false' : 'true',
    name: 'name',
    required: isNameRequired,
    className: 'form__input',
    placeholder: namePlaceholder,
    value: defaultName,
  });

  const emailEl = createElement('input', {
    id: 'email',
    type: showEmail ? 'text' : 'hidden',
    ['aria-hidden']: showEmail ? 'false' : 'true',
    name: 'email',
    required: isEmailRequired,
    className: 'form__input',
    placeholder: emailPlaceholder,
    value: defaultEmail,
  });

  const messageEl = createElement('textarea', {
    id: 'message',
    autoFocus: 'true',
    rows: '5',
    name: 'message',
    required: true,
    className: 'form__input form__input--textarea',
    placeholder: messagePlaceholder,
  });

  const cancelEl = createElement(
    'button',
    {
      type: 'button',
      className: 'btn btn--default',
      ['aria-label']: cancelButtonLabel,
      onClick: (e) => {
        onCancel && onCancel(e);
      },
    },
    cancelButtonLabel,
  );

  const formEl = createElement(
    'form',
    {
      className: 'form',
      onSubmit: handleSubmit,
    },
    [
      errorEl,

      showName &&
        createElement(
          'label',
          {
            htmlFor: 'name',
            className: 'form__label',
          },
          [
            createElement(
              'span',
              { className: 'form__label__text' },
              nameLabel,
              isNameRequired &&
                createElement('span', { className: 'form__label__text--required' }, ` ${isRequiredLabel}`),
            ),
            nameEl,
          ],
        ),
      !showName && nameEl,

      showEmail &&
        createElement(
          'label',
          {
            htmlFor: 'email',
            className: 'form__label',
          },
          [
            createElement(
              'span',
              { className: 'form__label__text' },
              emailLabel,
              isEmailRequired &&
                createElement('span', { className: 'form__label__text--required' }, ` ${isRequiredLabel}`),
            ),
            emailEl,
          ],
        ),
      !showEmail && emailEl,

      createElement(
        'label',
        {
          htmlFor: 'message',
          className: 'form__label',
        },
        [
          createElement(
            'span',
            { className: 'form__label__text' },
            messageLabel,
            createElement('span', { className: 'form__label__text--required' }, ` ${isRequiredLabel}`),
          ),
          messageEl,
        ],
      ),

      createElement(
        'div',
        {
          className: 'btn-group',
        },
        [submitEl, cancelEl],
      ),
    ],
  );

  return {
    get el() {
      return formEl;
    },
    showError,
    hideError,
  };
}

const XMLNS$1 = 'http://www.w3.org/2000/svg';

/**
 * Sentry Logo
 */
function Logo({ colorScheme }) {
  const createElementNS = (tagName) =>
    WINDOW.document.createElementNS(XMLNS$1, tagName);
  const svg = setAttributesNS(createElementNS('svg'), {
    class: 'sentry-logo',
    width: '32',
    height: '30',
    viewBox: '0 0 72 66',
    fill: 'none',
  });

  const path = setAttributesNS(createElementNS('path'), {
    transform: 'translate(11, 11)',
    d: 'M29,2.26a4.67,4.67,0,0,0-8,0L14.42,13.53A32.21,32.21,0,0,1,32.17,40.19H27.55A27.68,27.68,0,0,0,12.09,17.47L6,28a15.92,15.92,0,0,1,9.23,12.17H4.62A.76.76,0,0,1,4,39.06l2.94-5a10.74,10.74,0,0,0-3.36-1.9l-2.91,5a4.54,4.54,0,0,0,1.69,6.24A4.66,4.66,0,0,0,4.62,44H19.15a19.4,19.4,0,0,0-8-17.31l2.31-4A23.87,23.87,0,0,1,23.76,44H36.07a35.88,35.88,0,0,0-16.41-31.8l4.67-8a.77.77,0,0,1,1.05-.27c.53.29,20.29,34.77,20.66,35.17a.76.76,0,0,1-.68,1.13H40.6q.09,1.91,0,3.81h4.78A4.59,4.59,0,0,0,50,39.43a4.49,4.49,0,0,0-.62-2.28Z',
  });
  svg.append(path);

  const defs = createElementNS('defs');
  const style = createElementNS('style');

  style.textContent = `
    path {
      fill: ${colorScheme === 'dark' ? '#fff' : '#362d59'};
    }`;

  if (colorScheme === 'system') {
    style.textContent += `
    @media (prefers-color-scheme: dark) {
      path: {
        fill: '#fff';
      }
    }
    `;
  }

  defs.append(style);
  svg.append(defs);

  return {
    get el() {
      return svg;
    },
  };
}

/**
 * Feedback dialog component that has the form
 */
function Dialog({
  formTitle,
  showBranding,
  showName,
  showEmail,
  isNameRequired,
  isEmailRequired,
  colorScheme,
  defaultName,
  defaultEmail,
  onClosed,
  onCancel,
  onSubmit,
  ...textLabels
}) {
  let el = null;

  /**
   * Handles when the dialog is clicked. In our case, the dialog is the
   * semi-transparent bg behind the form. We want clicks outside of the form to
   * hide the form.
   */
  function handleDialogClick() {
    close();

    // Only this should trigger `onClose`, we don't want the `close()` method to
    // trigger it, otherwise it can cause cycles.
    onClosed && onClosed();
  }

  /**
   * Close the dialog
   */
  function close() {
    if (el) {
      el.open = false;
    }
  }

  /**
   * Opens the dialog
   */
  function open() {
    if (el) {
      el.open = true;
    }
  }

  /**
   * Check if dialog is currently opened
   */
  function checkIsOpen() {
    return (el && el.open === true) || false;
  }

  const {
    el: formEl,
    showError,
    hideError,
  } = Form({
    showEmail,
    showName,
    isEmailRequired,
    isNameRequired,

    defaultName,
    defaultEmail,
    onSubmit,
    onCancel,
    ...textLabels,
  });

  el = createElement(
    'dialog',
    {
      className: 'dialog',
      open: true,
      onClick: handleDialogClick,
    },
    createElement(
      'div',
      {
        className: 'dialog__content',
        onClick: e => {
          // Stop event propagation so clicks on content modal do not propagate to dialog (which will close dialog)
          e.stopPropagation();
        },
      },
      createElement(
        'h2',
        { className: 'dialog__header' },
        formTitle,
        showBranding &&
          createElement(
            'a',
            {
              className: 'brand-link',
              target: '_blank',
              href: 'https://sentry.io/welcome/',
              title: 'Powered by Sentry',
              rel: 'noopener noreferrer',
            },
            Logo({ colorScheme }).el,
          ),
      ),
      formEl,
    ),
  );

  return {
    get el() {
      return el;
    },
    showError,
    hideError,
    open,
    close,
    checkIsOpen,
  };
}

const WIDTH = 16;
const HEIGHT = 17;
const XMLNS = 'http://www.w3.org/2000/svg';

/**
 * Success Icon (checkmark)
 */
function SuccessIcon() {
  const createElementNS = (tagName) =>
    WINDOW.document.createElementNS(XMLNS, tagName);
  const svg = setAttributesNS(createElementNS('svg'), {
    class: 'success-icon',
    width: `${WIDTH}`,
    height: `${HEIGHT}`,
    viewBox: `0 0 ${WIDTH} ${HEIGHT}`,
    fill: 'none',
  });

  const g = setAttributesNS(createElementNS('g'), {
    clipPath: 'url(#clip0_57_156)',
  });

  const path2 = setAttributesNS(createElementNS('path'), {
    ['fill-rule']: 'evenodd',
    ['clip-rule']: 'evenodd',
    d: 'M3.55544 15.1518C4.87103 16.0308 6.41775 16.5 8 16.5C10.1217 16.5 12.1566 15.6571 13.6569 14.1569C15.1571 12.6566 16 10.6217 16 8.5C16 6.91775 15.5308 5.37103 14.6518 4.05544C13.7727 2.73985 12.5233 1.71447 11.0615 1.10897C9.59966 0.503466 7.99113 0.34504 6.43928 0.653721C4.88743 0.962403 3.46197 1.72433 2.34315 2.84315C1.22433 3.96197 0.462403 5.38743 0.153721 6.93928C-0.15496 8.49113 0.00346625 10.0997 0.608967 11.5615C1.21447 13.0233 2.23985 14.2727 3.55544 15.1518ZM4.40546 3.1204C5.46945 2.40946 6.72036 2.03 8 2.03C9.71595 2.03 11.3616 2.71166 12.575 3.92502C13.7883 5.13838 14.47 6.78405 14.47 8.5C14.47 9.77965 14.0905 11.0306 13.3796 12.0945C12.6687 13.1585 11.6582 13.9878 10.476 14.4775C9.29373 14.9672 7.99283 15.0953 6.73777 14.8457C5.48271 14.596 4.32987 13.9798 3.42502 13.075C2.52018 12.1701 1.90397 11.0173 1.65432 9.76224C1.40468 8.50718 1.5328 7.20628 2.0225 6.02404C2.5122 4.8418 3.34148 3.83133 4.40546 3.1204Z',
  });
  const path = setAttributesNS(createElementNS('path'), {
    d: 'M6.68775 12.4297C6.78586 12.4745 6.89218 12.4984 7 12.5C7.11275 12.4955 7.22315 12.4664 7.32337 12.4145C7.4236 12.3627 7.51121 12.2894 7.58 12.2L12 5.63999C12.0848 5.47724 12.1071 5.28902 12.0625 5.11098C12.0178 4.93294 11.9095 4.77744 11.7579 4.67392C11.6064 4.57041 11.4221 4.52608 11.24 4.54931C11.0579 4.57254 10.8907 4.66173 10.77 4.79999L6.88 10.57L5.13 8.56999C5.06508 8.49566 4.98613 8.43488 4.89768 8.39111C4.80922 8.34735 4.713 8.32148 4.61453 8.31498C4.51605 8.30847 4.41727 8.32147 4.32382 8.35322C4.23038 8.38497 4.14413 8.43484 4.07 8.49999C3.92511 8.63217 3.83692 8.81523 3.82387 9.01092C3.81083 9.2066 3.87393 9.39976 4 9.54999L6.43 12.24C6.50187 12.3204 6.58964 12.385 6.68775 12.4297Z',
  });

  svg.appendChild(g).append(path, path2);

  const speakerDefs = createElementNS('defs');
  const speakerClipPathDef = setAttributesNS(createElementNS('clipPath'), {
    id: 'clip0_57_156',
  });

  const speakerRect = setAttributesNS(createElementNS('rect'), {
    width: `${WIDTH}`,
    height: `${WIDTH}`,
    fill: 'white',
    transform: 'translate(0 0.5)',
  });

  speakerClipPathDef.appendChild(speakerRect);
  speakerDefs.appendChild(speakerClipPathDef);

  svg.appendChild(speakerDefs).appendChild(speakerClipPathDef).appendChild(speakerRect);

  return {
    get el() {
      return svg;
    },
  };
}

/**
 * Feedback dialog component that has the form
 */
function SuccessMessage({ message, onRemove }) {
  function remove() {
    if (!el) {
      return;
    }

    el.remove();
    onRemove && onRemove();
  }

  const el = createElement(
    'div',
    {
      className: 'success-message',
      onClick: remove,
    },
    SuccessIcon().el,
    message,
  );

  return {
    el,
    remove,
  };
}

/**
 * Creates a new widget. Returns public methods that control widget behavior.
 */
function createWidget({
  shadow,
  options: { shouldCreateActor = true, ...options },
  attachTo,
}) {
  let actor;
  let dialog;
  let isDialogOpen = false;

  /**
   * Show the success message for 5 seconds
   */
  function showSuccessMessage() {
    if (!shadow) {
      return;
    }

    try {
      const success = SuccessMessage({
        message: options.successMessageText,
        onRemove: () => {
          if (timeoutId) {
            clearTimeout(timeoutId);
          }
          showActor();
        },
      });

      if (!success.el) {
        throw new Error('Unable to show success message');
      }

      shadow.appendChild(success.el);

      const timeoutId = setTimeout(() => {
        if (success) {
          success.remove();
        }
      }, 5000);
    } catch (err) {
      // TODO: error handling
      utils.logger.error(err);
    }
  }

  /**
   * Handler for when the feedback form is completed by the user. This will
   * create and send the feedback message as an event.
   */
  async function _handleFeedbackSubmit(feedback) {
    if (!dialog) {
      return;
    }

    // Simple validation for now, just check for non-empty required fields
    const emptyField = [];
    if (options.isNameRequired && !feedback.name) {
      emptyField.push(options.nameLabel);
    }
    if (options.isEmailRequired && !feedback.email) {
      emptyField.push(options.emailLabel);
    }
    if (!feedback.message) {
      emptyField.push(options.messageLabel);
    }
    if (emptyField.length > 0) {
      dialog.showError(`Please enter in the following required fields: ${emptyField.join(', ')}`);
      return;
    }

    const result = await handleFeedbackSubmit(dialog, feedback);

    // Error submitting feedback
    if (!result) {
      if (options.onSubmitError) {
        options.onSubmitError();
      }

      return;
    }

    // Success
    removeDialog();
    showSuccessMessage();

    if (options.onSubmitSuccess) {
      options.onSubmitSuccess();
    }
  }

  /**
   * Internal handler when dialog is opened
   */
  function handleOpenDialog() {
    // Flush replay if integration exists
    const client = core.getClient();
    const replay =
      client &&
      client.getIntegrationByName &&
      client.getIntegrationByName('Replay');
    if (!replay) {
      return;
    }
    replay.flush().catch(err => {
      DEBUG_BUILD && utils.logger.error(err);
    });
  }

  /**
   * Displays the default actor
   */
  function showActor() {
    actor && actor.show();
  }

  /**
   * Hides the default actor
   */
  function hideActor() {
    actor && actor.hide();
  }

  /**
   * Removes the default actor element
   */
  function removeActor() {
    actor && actor.el && actor.el.remove();
  }

  /**
   *
   */
  function openDialog() {
    try {
      if (dialog) {
        dialog.open();
        isDialogOpen = true;
        if (options.onFormOpen) {
          options.onFormOpen();
        }
        handleOpenDialog();
        return;
      }

      const userKey = options.useSentryUser;
      const scope = core.getCurrentScope();
      const user = scope && scope.getUser();

      dialog = Dialog({
        colorScheme: options.colorScheme,
        showBranding: options.showBranding,
        showName: options.showName || options.isNameRequired,
        showEmail: options.showEmail || options.isEmailRequired,
        isNameRequired: options.isNameRequired,
        isEmailRequired: options.isEmailRequired,
        formTitle: options.formTitle,
        cancelButtonLabel: options.cancelButtonLabel,
        submitButtonLabel: options.submitButtonLabel,
        emailLabel: options.emailLabel,
        emailPlaceholder: options.emailPlaceholder,
        messageLabel: options.messageLabel,
        messagePlaceholder: options.messagePlaceholder,
        nameLabel: options.nameLabel,
        namePlaceholder: options.namePlaceholder,
        isRequiredLabel: options.isRequiredLabel,
        defaultName: (userKey && user && user[userKey.name]) || '',
        defaultEmail: (userKey && user && user[userKey.email]) || '',
        onClosed: () => {
          showActor();
          isDialogOpen = false;

          if (options.onFormClose) {
            options.onFormClose();
          }
        },
        onCancel: () => {
          closeDialog();
          showActor();
        },
        onSubmit: _handleFeedbackSubmit,
      });

      if (!dialog.el) {
        throw new Error('Unable to open Feedback dialog');
      }

      shadow.appendChild(dialog.el);

      // Hides the default actor whenever dialog is opened
      hideActor();

      if (options.onFormOpen) {
        options.onFormOpen();
      }
      handleOpenDialog();
    } catch (err) {
      // TODO: Error handling?
      utils.logger.error(err);
    }
  }

  /**
   * Closes the dialog
   */
  function closeDialog() {
    if (dialog) {
      dialog.close();
      isDialogOpen = false;

      if (options.onFormClose) {
        options.onFormClose();
      }
    }
  }

  /**
   * Removes the dialog element from DOM
   */
  function removeDialog() {
    if (dialog) {
      closeDialog();
      const dialogEl = dialog.el;
      dialogEl && dialogEl.remove();
      dialog = undefined;
    }
  }

  /**
   *
   */
  function handleActorClick() {
    // Open dialog
    if (!isDialogOpen) {
      openDialog();
    }

    // Hide actor button
    hideActor();
  }

  if (attachTo) {
    attachTo.addEventListener('click', handleActorClick);
  } else if (shouldCreateActor) {
    actor = Actor({ buttonLabel: options.buttonLabel, onClick: handleActorClick });
    actor.el && shadow.appendChild(actor.el);
  }

  return {
    get actor() {
      return actor;
    },
    get dialog() {
      return dialog;
    },

    showActor,
    hideActor,
    removeActor,

    openDialog,
    closeDialog,
    removeDialog,
  };
}

const doc = WINDOW.document;

const feedbackIntegration = ((options) => {
  // eslint-disable-next-line deprecation/deprecation
  return new Feedback(options);
}) ;

/**
 * Feedback integration. When added as an integration to the SDK, it will
 * inject a button in the bottom-right corner of the window that opens a
 * feedback modal when clicked.
 *
 * @deprecated Use `feedbackIntegration()` instead.
 */
class Feedback  {
  /**
   * @inheritDoc
   */
   static __initStatic() {this.id = 'Feedback';}

  /**
   * @inheritDoc
   */

  /**
   * Feedback configuration options
   */

  /**
   * Reference to widget element that is created when autoInject is true
   */

  /**
   * List of all widgets that are created from the integration
   */

  /**
   * Reference to the host element where widget is inserted
   */

  /**
   * Refernce to Shadow DOM root
   */

  /**
   * Tracks if actor styles have ever been inserted into shadow DOM
   */

   constructor({
    autoInject = true,
    id = 'sentry-feedback',
    isEmailRequired = false,
    isNameRequired = false,
    showBranding = true,
    showEmail = true,
    showName = true,
    useSentryUser = {
      email: 'email',
      name: 'username',
    },

    themeDark,
    themeLight,
    colorScheme = 'system',

    buttonLabel = ACTOR_LABEL,
    cancelButtonLabel = CANCEL_BUTTON_LABEL,
    submitButtonLabel = SUBMIT_BUTTON_LABEL,
    formTitle = FORM_TITLE,
    emailPlaceholder = EMAIL_PLACEHOLDER,
    emailLabel = EMAIL_LABEL,
    messagePlaceholder = MESSAGE_PLACEHOLDER,
    messageLabel = MESSAGE_LABEL,
    namePlaceholder = NAME_PLACEHOLDER,
    nameLabel = NAME_LABEL,
    isRequiredLabel = IS_REQUIRED_LABEL,
    successMessageText = SUCCESS_MESSAGE_TEXT,

    onFormClose,
    onFormOpen,
    onSubmitError,
    onSubmitSuccess,
  } = {}) {
    // eslint-disable-next-line deprecation/deprecation
    this.name = Feedback.id;

    // tsc fails if these are not initialized explicitly constructor, e.g. can't call `_initialize()`
    this._host = null;
    this._shadow = null;
    this._widget = null;
    this._widgets = new Set();
    this._hasInsertedActorStyles = false;

    this.options = {
      autoInject,
      showBranding,
      id,
      isEmailRequired,
      isNameRequired,
      showEmail,
      showName,
      useSentryUser,

      colorScheme,
      themeDark: {
        ...DEFAULT_THEME.dark,
        ...themeDark,
      },
      themeLight: {
        ...DEFAULT_THEME.light,
        ...themeLight,
      },

      buttonLabel,
      cancelButtonLabel,
      submitButtonLabel,
      formTitle,
      emailLabel,
      emailPlaceholder,
      messageLabel,
      messagePlaceholder,
      nameLabel,
      namePlaceholder,
      isRequiredLabel,
      successMessageText,

      onFormClose,
      onFormOpen,
      onSubmitError,
      onSubmitSuccess,
    };
  }

  /**
   * Setup and initialize feedback container
   */
   setupOnce() {
    if (!utils.isBrowser()) {
      return;
    }

    try {
      this._cleanupWidgetIfExists();

      const { autoInject } = this.options;

      if (!autoInject) {
        // Nothing to do here
        return;
      }

      this._createWidget(this.options);
    } catch (err) {
      DEBUG_BUILD && utils.logger.error(err);
    }
  }

  /**
   * Allows user to open the dialog box. Creates a new widget if
   * `autoInject` was false, otherwise re-uses the default widget that was
   * created during initialization of the integration.
   */
   openDialog() {
    if (!this._widget) {
      this._createWidget({ ...this.options, shouldCreateActor: false });
    }

    if (!this._widget) {
      return;
    }

    this._widget.openDialog();
  }

  /**
   * Closes the dialog for the default widget, if it exists
   */
   closeDialog() {
    if (!this._widget) {
      // Nothing to do if widget does not exist
      return;
    }

    this._widget.closeDialog();
  }

  /**
   * Adds click listener to attached element to open a feedback dialog
   */
   attachTo(el, optionOverrides) {
    try {
      const options = mergeOptions(this.options, optionOverrides || {});

      return this._ensureShadowHost(options, ({ shadow }) => {
        const targetEl =
          typeof el === 'string' ? doc.querySelector(el) : typeof el.addEventListener === 'function' ? el : null;

        if (!targetEl) {
          DEBUG_BUILD && utils.logger.error('[Feedback] Unable to attach to target element');
          return null;
        }

        const widget = createWidget({ shadow, options, attachTo: targetEl });
        this._widgets.add(widget);

        if (!this._widget) {
          this._widget = widget;
        }

        return widget;
      });
    } catch (err) {
      DEBUG_BUILD && utils.logger.error(err);
      return null;
    }
  }

  /**
   * Creates a new widget. Accepts partial options to override any options passed to constructor.
   */
   createWidget(
    optionOverrides,
  ) {
    try {
      return this._createWidget(mergeOptions(this.options, optionOverrides || {}));
    } catch (err) {
      DEBUG_BUILD && utils.logger.error(err);
      return null;
    }
  }

  /**
   * Removes a single widget
   */
   removeWidget(widget) {
    if (!widget) {
      return false;
    }

    try {
      if (this._widgets.has(widget)) {
        widget.removeActor();
        widget.removeDialog();
        this._widgets.delete(widget);

        if (this._widget === widget) {
          // TODO: is more clean-up needed? e.g. call remove()
          this._widget = null;
        }

        return true;
      }
    } catch (err) {
      DEBUG_BUILD && utils.logger.error(err);
    }

    return false;
  }

  /**
   * Returns the default (first-created) widget
   */
   getWidget() {
    return this._widget;
  }

  /**
   * Removes the Feedback integration (including host, shadow DOM, and all widgets)
   */
   remove() {
    if (this._host) {
      this._host.remove();
    }
    this._initialize();
  }

  /**
   * Initializes values of protected properties
   */
   _initialize() {
    this._host = null;
    this._shadow = null;
    this._widget = null;
    this._widgets = new Set();
    this._hasInsertedActorStyles = false;
  }

  /**
   * Clean-up the widget if it already exists in the DOM. This shouldn't happen
   * in prod, but can happen in development with hot module reloading.
   */
   _cleanupWidgetIfExists() {
    if (this._host) {
      this.remove();
    }
    const existingFeedback = doc.querySelector(`#${this.options.id}`);
    if (existingFeedback) {
      existingFeedback.remove();
    }
  }

  /**
   * Creates a new widget, after ensuring shadow DOM exists
   */
   _createWidget(options) {
    return this._ensureShadowHost(options, ({ shadow }) => {
      const widget = createWidget({ shadow, options });

      if (!this._hasInsertedActorStyles && widget.actor) {
        shadow.appendChild(createActorStyles(doc));
        this._hasInsertedActorStyles = true;
      }

      this._widgets.add(widget);

      if (!this._widget) {
        this._widget = widget;
      }

      return widget;
    });
  }

  /**
   * Ensures that shadow DOM exists and is added to the DOM
   */
   _ensureShadowHost(
    options,
    cb,
  ) {
    let needsAppendHost = false;

    // Don't create if it already exists
    if (!this._shadow || !this._host) {
      const { id, colorScheme, themeLight, themeDark } = options;
      const { shadow, host } = createShadowHost({
        id,
        colorScheme,
        themeLight,
        themeDark,
      });
      this._shadow = shadow;
      this._host = host;
      needsAppendHost = true;
    }

    // set data attribute on host for different themes
    this._host.dataset.sentryFeedbackColorscheme = options.colorScheme;

    const result = cb({ shadow: this._shadow, host: this._host });

    if (needsAppendHost) {
      doc.body.appendChild(this._host);
    }

    return result;
  }
} Feedback.__initStatic();

exports.Feedback = Feedback;
exports.feedbackIntegration = feedbackIntegration;
exports.sendFeedback = sendFeedback;
//# sourceMappingURL=index.js.map
