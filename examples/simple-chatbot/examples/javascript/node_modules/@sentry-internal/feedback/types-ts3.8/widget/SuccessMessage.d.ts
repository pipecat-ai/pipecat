import { FeedbackComponent } from '../types';
export interface SuccessMessageProps {
    message: string;
    onRemove?: () => void;
}
interface SuccessMessageComponent extends FeedbackComponent<HTMLDivElement> {
    /**
     * Removes the component
     */
    remove: () => void;
}
/**
 * Feedback dialog component that has the form
 */
export declare function SuccessMessage({ message, onRemove }: SuccessMessageProps): SuccessMessageComponent;
export {};
//# sourceMappingURL=SuccessMessage.d.ts.map
