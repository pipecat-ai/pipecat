import type { FeedbackComponent, FeedbackInternalOptions } from '../types';
import type { FormComponentProps } from './Form';
export interface DialogProps extends FormComponentProps, Pick<FeedbackInternalOptions, 'formTitle' | 'showBranding' | 'colorScheme'> {
    onClosed?: () => void;
}
export interface DialogComponent extends FeedbackComponent<HTMLDialogElement> {
    /**
     * Shows the error message
     */
    showError: (message: string) => void;
    /**
     * Hides the error message
     */
    hideError: () => void;
    /**
     * Opens and shows the dialog and form
     */
    open: () => void;
    /**
     * Closes the dialog and form
     */
    close: () => void;
    /**
     * Check if dialog is currently opened
     */
    checkIsOpen: () => boolean;
}
/**
 * Feedback dialog component that has the form
 */
export declare function Dialog({ formTitle, showBranding, showName, showEmail, isNameRequired, isEmailRequired, colorScheme, defaultName, defaultEmail, onClosed, onCancel, onSubmit, ...textLabels }: DialogProps): DialogComponent;
//# sourceMappingURL=Dialog.d.ts.map