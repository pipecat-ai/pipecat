import { TransportMakeRequestResponse } from '@sentry/types';
import { FeedbackFormData, SendFeedbackOptions } from '../types';
import { DialogComponent } from '../widget/Dialog';
/**
 * Handles UI behavior of dialog when feedback is submitted, calls
 * `sendFeedback` to send feedback.
 */
export declare function handleFeedbackSubmit(dialog: DialogComponent | null, feedback: FeedbackFormData, options?: SendFeedbackOptions): Promise<TransportMakeRequestResponse | void>;
//# sourceMappingURL=handleFeedbackSubmit.d.ts.map
