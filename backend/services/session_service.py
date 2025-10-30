"""Session service for managing voice pipeline sessions"""

import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime

from ..models.session import (
    Session,
    SessionCreate,
    SessionStatus,
    SessionType,
    SessionMetrics,
)


class SessionService:
    """Service for session management and tracking"""

    def __init__(self):
        # In-memory storage (replace with database in production)
        self.sessions: Dict[str, Session] = {}

    def create_session(
        self,
        session_create: SessionCreate,
        remote_addr: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Session:
        """Create a new session"""
        session_id = str(uuid.uuid4())

        session = Session(
            id=session_id,
            user_id=session_create.user_id,
            session_type=session_create.session_type,
            voice_config=session_create.voice_config,
            system_prompt=session_create.system_prompt,
            metadata=session_create.metadata,
            status=SessionStatus.PENDING,
            started_at=datetime.utcnow(),
            remote_addr=remote_addr,
            user_agent=user_agent,
        )

        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID"""
        return self.sessions.get(session_id)

    def update_session_status(
        self, session_id: str, status: SessionStatus
    ) -> Optional[Session]:
        """Update session status"""
        session = self.sessions.get(session_id)
        if not session:
            return None

        session.status = status

        # Set ended_at when session completes
        if status in [
            SessionStatus.COMPLETED,
            SessionStatus.FAILED,
            SessionStatus.TIMEOUT,
        ]:
            session.ended_at = datetime.utcnow()
            if session.ended_at:
                session.duration_seconds = (
                    session.ended_at - session.started_at
                ).total_seconds()

        return session

    def add_session_message(
        self, session_id: str, direction: str = "sent"
    ) -> Optional[Session]:
        """Increment message counter"""
        session = self.sessions.get(session_id)
        if not session:
            return None

        if direction == "sent":
            session.messages_sent += 1
        elif direction == "received":
            session.messages_received += 1

        return session

    def add_session_error(self, session_id: str, error: str) -> Optional[Session]:
        """Add error to session"""
        session = self.sessions.get(session_id)
        if not session:
            return None

        session.errors.append(f"{datetime.utcnow().isoformat()}: {error}")
        return session

    def update_twilio_info(
        self, session_id: str, call_sid: str, stream_sid: str
    ) -> Optional[Session]:
        """Update Twilio specific information"""
        session = self.sessions.get(session_id)
        if not session:
            return None

        session.call_sid = call_sid
        session.stream_sid = stream_sid
        return session

    def get_session_metrics(self, session_id: str) -> Optional[SessionMetrics]:
        """Get real-time session metrics"""
        session = self.sessions.get(session_id)
        if not session:
            return None

        uptime = (datetime.utcnow() - session.started_at).total_seconds()

        return SessionMetrics(
            session_id=session.id,
            status=session.status,
            uptime_seconds=uptime,
            messages_sent=session.messages_sent,
            messages_received=session.messages_received,
        )

    def list_sessions(
        self,
        user_id: Optional[str] = None,
        status: Optional[SessionStatus] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Session]:
        """List sessions with optional filters"""
        sessions = list(self.sessions.values())

        # Filter by user_id
        if user_id:
            sessions = [s for s in sessions if s.user_id == user_id]

        # Filter by status
        if status:
            sessions = [s for s in sessions if s.status == status]

        # Sort by start time (newest first)
        sessions.sort(key=lambda s: s.started_at, reverse=True)

        # Pagination
        return sessions[skip : skip + limit]

    def get_active_sessions(self) -> List[Session]:
        """Get all active sessions"""
        return [s for s in self.sessions.values() if s.status == SessionStatus.ACTIVE]

    def get_session_count(self) -> int:
        """Get total session count"""
        return len(self.sessions)

    def get_active_session_count(self) -> int:
        """Get active session count"""
        return len(self.get_active_sessions())

    def delete_session(self, session_id: str) -> bool:
        """Delete session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def cleanup_old_sessions(self, days: int = 7) -> int:
        """Clean up sessions older than specified days"""
        cutoff = datetime.utcnow().timestamp() - (days * 24 * 60 * 60)
        deleted = 0

        for session_id, session in list(self.sessions.items()):
            if (
                session.status
                in [SessionStatus.COMPLETED, SessionStatus.FAILED, SessionStatus.TIMEOUT]
                and session.started_at.timestamp() < cutoff
            ):
                del self.sessions[session_id]
                deleted += 1

        return deleted


# Global session service instance
_session_service: Optional[SessionService] = None


def get_session_service() -> SessionService:
    """Get or create the global session service instance"""
    global _session_service
    if _session_service is None:
        _session_service = SessionService()
    return _session_service
