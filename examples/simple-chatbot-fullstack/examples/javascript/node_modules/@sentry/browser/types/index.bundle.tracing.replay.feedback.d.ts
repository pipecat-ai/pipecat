import { Feedback, feedbackIntegration } from '@sentry-internal/feedback';
import { BrowserTracing, Span, addExtensionMethods } from '@sentry-internal/tracing';
import { InternalReplay, internalReplayIntegration } from '@sentry/replay';
import { bundleBrowserTracingIntegration as browserTracingIntegration } from './helpers';
export { Feedback, InternalReplay as Replay, feedbackIntegration, internalReplayIntegration as replayIntegration, BrowserTracing, browserTracingIntegration, Span, addExtensionMethods, };
export * from './index.bundle.base';
//# sourceMappingURL=index.bundle.tracing.replay.feedback.d.ts.map