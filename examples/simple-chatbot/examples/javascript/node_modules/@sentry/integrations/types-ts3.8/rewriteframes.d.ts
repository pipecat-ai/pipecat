import { Event, Integration, IntegrationClass, StackFrame } from '@sentry/types';
type StackFrameIteratee = (frame: StackFrame) => StackFrame;
interface RewriteFramesOptions {
    root?: string;
    prefix?: string;
    iteratee?: StackFrameIteratee;
}
export declare const rewriteFramesIntegration: (options?: RewriteFramesOptions | undefined) => import("@sentry/types").IntegrationFnResult;
/**
 * Rewrite event frames paths.
 * @deprecated Use `rewriteFramesIntegration()` instead.
 */
export declare const RewriteFrames: IntegrationClass<Integration & {
    processEvent: (event: Event) => Event;
}> & (new (options?: {
    root?: string;
    prefix?: string;
    iteratee?: StackFrameIteratee;
}) => Integration);
export {};
//# sourceMappingURL=rewriteframes.d.ts.map
