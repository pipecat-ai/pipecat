import { Session, SessionOptions } from '../types';
/**
 * Get or create a session, when initializing the replay.
 * Returns a session that may be unsampled.
 */
export declare function loadOrCreateSession({ traceInternals, sessionIdleExpire, maxReplayDuration, previousSessionId, }: {
    sessionIdleExpire: number;
    maxReplayDuration: number;
    traceInternals?: boolean;
    previousSessionId?: string;
}, sessionOptions: SessionOptions): Session;
//# sourceMappingURL=loadOrCreateSession.d.ts.map
