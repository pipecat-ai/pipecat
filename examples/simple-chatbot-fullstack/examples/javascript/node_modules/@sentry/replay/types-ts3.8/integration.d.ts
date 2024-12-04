import { Integration } from '@sentry/types';
import { ReplayConfiguration, SendBufferedReplayOptions } from './types';
export declare const replayIntegration: (options?: ReplayConfiguration) => Replay;
/**
 * The main replay integration class, to be passed to `init({  integrations: [] })`.
 * @deprecated Use `replayIntegration()` instead.
 */
export declare class Replay implements Integration {
    /**
     * @inheritDoc
     */
    static id: string;
    /**
     * @inheritDoc
     */
    name: string;
    /**
     * Options to pass to `rrweb.record()`
     */
    private readonly _recordingOptions;
    /**
     * Initial options passed to the replay integration, merged with default values.
     * Note: `sessionSampleRate` and `errorSampleRate` are not required here, as they
     * can only be finally set when setupOnce() is called.
     *
     * @private
     */
    private readonly _initialOptions;
    private _replay?;
    constructor({ flushMinDelay, flushMaxDelay, minReplayDuration, maxReplayDuration, stickySession, useCompression, workerUrl, _experiments, sessionSampleRate, errorSampleRate, maskAllText, maskAllInputs, blockAllMedia, mutationBreadcrumbLimit, mutationLimit, slowClickTimeout, slowClickIgnoreSelectors, networkDetailAllowUrls, networkDetailDenyUrls, networkCaptureBodies, networkRequestHeaders, networkResponseHeaders, mask, maskAttributes, unmask, block, unblock, ignore, maskFn, beforeAddRecordingEvent, beforeErrorSampling, blockClass, blockSelector, maskInputOptions, maskTextClass, maskTextSelector, ignoreClass, }?: ReplayConfiguration);
    /*If replay has already been initialized
    Update _isInitialized */
    protected _isInitialized: boolean;
    /**
     * Setup and initialize replay container
     */
    setupOnce(): void;
    /**
     * Start a replay regardless of sampling rate. Calling this will always
     * create a new session. Will throw an error if replay is already in progress.
     *
     * Creates or loads a session, attaches listeners to varying events (DOM,
     * PerformanceObserver, Recording, Sentry SDK, etc)
     */
    start(): void;
    /**
     * Start replay buffering. Buffers until `flush()` is called or, if
     * `replaysOnErrorSampleRate` > 0, until an error occurs.
     */
    startBuffering(): void;
    /**
     * Currently, this needs to be manually called (e.g. for tests). Sentry SDK
     * does not support a teardown
     */
    stop(): Promise<void>;
    /**
     * If not in "session" recording mode, flush event buffer which will create a new replay.
     * Unless `continueRecording` is false, the replay will continue to record and
     * behave as a "session"-based replay.
     *
     * Otherwise, queue up a flush.
     */
    flush(options?: SendBufferedReplayOptions): Promise<void>;
    /**
     * Get the current session ID.
     */
    getReplayId(): string | undefined;
    /**
     * Initializes replay.
     */
    protected _initialize(): void;
    /** Setup the integration. */
    private _setup;
    /** Get canvas options from ReplayCanvas integration, if it is also added. */
    private _maybeLoadFromReplayCanvasIntegration;
}
//# sourceMappingURL=integration.d.ts.map
