import type { FeedbackComponent, FeedbackFormData, FeedbackInternalOptions, FeedbackTextConfiguration } from '../types';
export interface FormComponentProps extends Pick<FeedbackInternalOptions, 'showName' | 'showEmail' | 'isNameRequired' | 'isEmailRequired' | Exclude<keyof FeedbackTextConfiguration, 'buttonLabel' | 'formTitle' | 'successMessageText'>> {
    /**
     * A default name value to render the input with. Empty strings are ok.
     */
    defaultName: string;
    /**
     * A default email value to render the input with. Empty strings are ok.
     */
    defaultEmail: string;
    onCancel?: (e: Event) => void;
    onSubmit?: (feedback: FeedbackFormData) => void;
}
interface FormComponent extends FeedbackComponent<HTMLFormElement> {
    /**
     * Shows the error message
     */
    showError: (message: string) => void;
    /**
     * Hides the error message
     */
    hideError: () => void;
}
/**
 * Creates the form element
 */
export declare function Form({ nameLabel, namePlaceholder, emailLabel, emailPlaceholder, messageLabel, messagePlaceholder, isRequiredLabel, cancelButtonLabel, submitButtonLabel, showName, showEmail, isNameRequired, isEmailRequired, defaultName, defaultEmail, onCancel, onSubmit, }: FormComponentProps): FormComponent;
export {};
//# sourceMappingURL=Form.d.ts.map