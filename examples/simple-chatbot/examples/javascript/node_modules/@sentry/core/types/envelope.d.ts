import type { DsnComponents, Event, EventEnvelope, SdkMetadata, Session, SessionAggregates, SessionEnvelope } from '@sentry/types';
/** Creates an envelope from a Session */
export declare function createSessionEnvelope(session: Session | SessionAggregates, dsn?: DsnComponents, metadata?: SdkMetadata, tunnel?: string): SessionEnvelope;
/**
 * Create an Envelope from an event.
 */
export declare function createEventEnvelope(event: Event, dsn?: DsnComponents, metadata?: SdkMetadata, tunnel?: string): EventEnvelope;
//# sourceMappingURL=envelope.d.ts.map