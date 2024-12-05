import type { TransportMakeRequestResponse } from '@sentry/types';
import type { FeedbackFormData, SendFeedbackOptions } from '../types';
import type { DialogComponent } from '../widget/Dialog';
/**
 * Handles UI behavior of dialog when feedback is submitted, calls
 * `sendFeedback` to send feedback.
 */
export declare function handleFeedbackSubmit(dialog: DialogComponent | null, feedback: FeedbackFormData, options?: SendFeedbackOptions): Promise<TransportMakeRequestResponse | void>;
//# sourceMappingURL=handleFeedbackSubmit.d.ts.map