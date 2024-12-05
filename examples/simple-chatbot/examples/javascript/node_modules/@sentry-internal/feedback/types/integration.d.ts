import type { Integration } from '@sentry/types';
import type { FeedbackInternalOptions, FeedbackWidget, OptionalFeedbackConfiguration } from './types';
import { createShadowHost } from './widget/createShadowHost';
export declare const feedbackIntegration: (options?: OptionalFeedbackConfiguration) => Feedback;
/**
 * Feedback integration. When added as an integration to the SDK, it will
 * inject a button in the bottom-right corner of the window that opens a
 * feedback modal when clicked.
 *
 * @deprecated Use `feedbackIntegration()` instead.
 */
export declare class Feedback implements Integration {
    /**
     * @inheritDoc
     */
    static id: string;
    /**
     * @inheritDoc
     */
    name: string;
    /**
     * Feedback configuration options
     */
    options: FeedbackInternalOptions;
    /**
     * Reference to widget element that is created when autoInject is true
     */
    private _widget;
    /**
     * List of all widgets that are created from the integration
     */
    private _widgets;
    /**
     * Reference to the host element where widget is inserted
     */
    private _host;
    /**
     * Refernce to Shadow DOM root
     */
    private _shadow;
    /**
     * Tracks if actor styles have ever been inserted into shadow DOM
     */
    private _hasInsertedActorStyles;
    constructor({ autoInject, id, isEmailRequired, isNameRequired, showBranding, showEmail, showName, useSentryUser, themeDark, themeLight, colorScheme, buttonLabel, cancelButtonLabel, submitButtonLabel, formTitle, emailPlaceholder, emailLabel, messagePlaceholder, messageLabel, namePlaceholder, nameLabel, isRequiredLabel, successMessageText, onFormClose, onFormOpen, onSubmitError, onSubmitSuccess, }?: OptionalFeedbackConfiguration);
    /**
     * Setup and initialize feedback container
     */
    setupOnce(): void;
    /**
     * Allows user to open the dialog box. Creates a new widget if
     * `autoInject` was false, otherwise re-uses the default widget that was
     * created during initialization of the integration.
     */
    openDialog(): void;
    /**
     * Closes the dialog for the default widget, if it exists
     */
    closeDialog(): void;
    /**
     * Adds click listener to attached element to open a feedback dialog
     */
    attachTo(el: Element | string, optionOverrides?: OptionalFeedbackConfiguration): FeedbackWidget | null;
    /**
     * Creates a new widget. Accepts partial options to override any options passed to constructor.
     */
    createWidget(optionOverrides?: OptionalFeedbackConfiguration & {
        shouldCreateActor?: boolean;
    }): FeedbackWidget | null;
    /**
     * Removes a single widget
     */
    removeWidget(widget: FeedbackWidget | null | undefined): boolean;
    /**
     * Returns the default (first-created) widget
     */
    getWidget(): FeedbackWidget | null;
    /**
     * Removes the Feedback integration (including host, shadow DOM, and all widgets)
     */
    remove(): void;
    /**
     * Initializes values of protected properties
     */
    protected _initialize(): void;
    /**
     * Clean-up the widget if it already exists in the DOM. This shouldn't happen
     * in prod, but can happen in development with hot module reloading.
     */
    protected _cleanupWidgetIfExists(): void;
    /**
     * Creates a new widget, after ensuring shadow DOM exists
     */
    protected _createWidget(options: FeedbackInternalOptions & {
        shouldCreateActor?: boolean;
    }): FeedbackWidget | null;
    /**
     * Ensures that shadow DOM exists and is added to the DOM
     */
    protected _ensureShadowHost<T>(options: FeedbackInternalOptions, cb: (createShadowHostResult: ReturnType<typeof createShadowHost>) => T): T | null;
}
//# sourceMappingURL=integration.d.ts.map