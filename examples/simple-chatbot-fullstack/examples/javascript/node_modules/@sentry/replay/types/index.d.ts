import { Replay as InternalReplay, replayIntegration as internalReplayIntegration } from './integration';
import type { CanvasManagerInterface as InternalCanvasManagerInterface, CanvasManagerOptions as InternalCanvasManagerOptions, ReplayBreadcrumbFrame as InternalReplayBreadcrumbFrame, ReplayBreadcrumbFrameEvent as InternalReplayBreadcrumbFrameEvent, ReplayConfiguration as InternalReplayConfiguration, ReplayEventType as InternalReplayEventType, ReplayEventWithTime as InternalReplayEventWithTime, ReplayFrame as InternalReplayFrame, ReplayFrameEvent as InternalReplayFrameEvent, ReplayOptionFrameEvent as InternalReplayOptionFrameEvent, ReplaySpanFrame as InternalReplaySpanFrame, ReplaySpanFrameEvent as InternalReplaySpanFrameEvent } from './types';
import { getReplay as internalGetReplay } from './util/getReplay';
/** @deprecated Use the export from `@sentry/replay` or from framework-specific SDKs like `@sentry/react` or `@sentry/vue` */
declare const getReplay: typeof internalGetReplay;
/** @deprecated Use the export from `@sentry/replay` or from framework-specific SDKs like `@sentry/react` or `@sentry/vue` */
declare const replayIntegration: (options?: InternalReplayConfiguration | undefined) => InternalReplay;
/** @deprecated Use the export from `@sentry/replay` or from framework-specific SDKs like `@sentry/react` or `@sentry/vue` */
declare class Replay extends InternalReplay {
}
export { replayIntegration, getReplay, Replay, internalReplayIntegration, internalGetReplay, InternalReplay };
/** @deprecated Use the export from `@sentry/replay` or from framework-specific SDKs like `@sentry/react` or `@sentry/vue` */
type ReplayConfiguration = InternalReplayConfiguration;
/** @deprecated Use the export from `@sentry/replay` or from framework-specific SDKs like `@sentry/react` or `@sentry/vue` */
type ReplayEventType = InternalReplayEventType;
/** @deprecated Use the export from `@sentry/replay` or from framework-specific SDKs like `@sentry/react` or `@sentry/vue` */
type ReplayEventWithTime = InternalReplayEventWithTime;
/** @deprecated Use the export from `@sentry/replay` or from framework-specific SDKs like `@sentry/react` or `@sentry/vue` */
type ReplayBreadcrumbFrame = InternalReplayBreadcrumbFrame;
/** @deprecated Use the export from `@sentry/replay` or from framework-specific SDKs like `@sentry/react` or `@sentry/vue` */
type ReplayBreadcrumbFrameEvent = InternalReplayBreadcrumbFrameEvent;
/** @deprecated Use the export from `@sentry/replay` or from framework-specific SDKs like `@sentry/react` or `@sentry/vue` */
type ReplayOptionFrameEvent = InternalReplayOptionFrameEvent;
/** @deprecated Use the export from `@sentry/replay` or from framework-specific SDKs like `@sentry/react` or `@sentry/vue` */
type ReplayFrame = InternalReplayFrame;
/** @deprecated Use the export from `@sentry/replay` or from framework-specific SDKs like `@sentry/react` or `@sentry/vue` */
type ReplayFrameEvent = InternalReplayFrameEvent;
/** @deprecated Use the export from `@sentry/replay` or from framework-specific SDKs like `@sentry/react` or `@sentry/vue` */
type ReplaySpanFrame = InternalReplaySpanFrame;
/** @deprecated Use the export from `@sentry/replay` or from framework-specific SDKs like `@sentry/react` or `@sentry/vue` */
type ReplaySpanFrameEvent = InternalReplaySpanFrameEvent;
/** @deprecated Use the export from `@sentry/replay` or from framework-specific SDKs like `@sentry/react` or `@sentry/vue` */
type CanvasManagerInterface = InternalCanvasManagerInterface;
/** @deprecated Use the export from `@sentry/replay` or from framework-specific SDKs like `@sentry/react` or `@sentry/vue` */
type CanvasManagerOptions = InternalCanvasManagerOptions;
export type { CanvasManagerInterface, CanvasManagerOptions, ReplayBreadcrumbFrame, ReplayBreadcrumbFrameEvent, ReplayConfiguration, ReplayEventType, ReplayEventWithTime, ReplayFrame, ReplayFrameEvent, ReplayOptionFrameEvent, ReplaySpanFrame, ReplaySpanFrameEvent, InternalCanvasManagerInterface, InternalCanvasManagerOptions, InternalReplayBreadcrumbFrame, InternalReplayBreadcrumbFrameEvent, InternalReplayConfiguration, InternalReplayEventType, InternalReplayEventWithTime, InternalReplayFrame, InternalReplayFrameEvent, InternalReplayOptionFrameEvent, InternalReplaySpanFrame, InternalReplaySpanFrameEvent, };
export * from './types/deprecated';
//# sourceMappingURL=index.d.ts.map