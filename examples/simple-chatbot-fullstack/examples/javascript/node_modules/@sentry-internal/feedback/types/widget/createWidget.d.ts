import type { FeedbackInternalOptions, FeedbackWidget } from '../types';
interface CreateWidgetParams {
    /**
     * Shadow DOM to append to
     */
    shadow: ShadowRoot;
    /**
     * Feedback integration options
     */
    options: FeedbackInternalOptions & {
        shouldCreateActor?: boolean;
    };
    /**
     * An element to attach to, that when clicked, will open a dialog
     */
    attachTo?: Element;
    /**
     * If false, will not create an actor
     */
    shouldCreateActor?: boolean;
}
/**
 * Creates a new widget. Returns public methods that control widget behavior.
 */
export declare function createWidget({ shadow, options: { shouldCreateActor, ...options }, attachTo, }: CreateWidgetParams): FeedbackWidget;
export {};
//# sourceMappingURL=createWidget.d.ts.map