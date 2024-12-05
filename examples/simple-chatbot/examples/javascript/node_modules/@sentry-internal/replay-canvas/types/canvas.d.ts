import type { InternalCanvasManagerInterface, InternalCanvasManagerOptions } from '@sentry/replay';
import type { Integration, IntegrationClass } from '@sentry/types';
interface ReplayCanvasOptions {
    enableManualSnapshot?: boolean;
    maxCanvasSize?: [width: number, height: number];
    quality: 'low' | 'medium' | 'high';
}
type GetCanvasManager = (options: InternalCanvasManagerOptions) => InternalCanvasManagerInterface;
export interface ReplayCanvasIntegrationOptions {
    enableManualSnapshot?: boolean;
    maxCanvasSize?: number;
    recordCanvas: true;
    getCanvasManager: GetCanvasManager;
    sampling: {
        canvas: number;
    };
    dataURLOptions: {
        type: string;
        quality: number;
    };
}
/** Exported only for type safe tests. */
export declare const _replayCanvasIntegration: (options?: Partial<ReplayCanvasOptions>) => {
    name: string;
    setupOnce(): void;
    getOptions(): ReplayCanvasIntegrationOptions;
    snapshot(canvasElement?: HTMLCanvasElement): Promise<void>;
};
/**
 * Add this in addition to `replayIntegration()` to enable canvas recording.
 */
export declare const replayCanvasIntegration: (options?: Partial<ReplayCanvasOptions> | undefined) => import("@sentry/types").IntegrationFnResult;
/**
 * @deprecated Use `replayCanvasIntegration()` instead
 */
export declare const ReplayCanvas: IntegrationClass<Integration & {
    getOptions: () => ReplayCanvasIntegrationOptions;
}>;
export {};
//# sourceMappingURL=canvas.d.ts.map