from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session
from app.database import get_session as get_db_session
from app.schema import Session as SessionModel
from app.crud import create_session, get_session as get_session_by_id, update_session
from app.models import TranscriptTurn, FreezeEvent
from uuid import UUID

router = APIRouter()

@router.post("/sessions/", response_model=SessionModel)
def create_new_session(session: SessionModel, db: Session = Depends(get_db_session)):
    return create_session(db, session)

@router.get("/sessions/", response_model=List[SessionModel])
def list_sessions(db: Session = Depends(get_db_session)):
    from app.crud import get_all_sessions
    return get_all_sessions(db)

@router.get("/sessions/{session_id}", response_model=SessionModel)
def read_session(session_id: UUID, db: Session = Depends(get_db_session)):
    session = get_session_by_id(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@router.patch("/sessions/{session_id}/transcript", response_model=SessionModel)
def update_transcript(session_id: UUID, transcript: List[TranscriptTurn], db: Session = Depends(get_db_session)):
    updated = update_session(db, session_id, {"transcript": transcript})
    if not updated:
        raise HTTPException(status_code=404, detail="Session not found")
    return updated

@router.patch("/sessions/{session_id}/freeze_events", response_model=SessionModel)
def update_freeze_events(session_id: UUID, freeze_events: List[FreezeEvent], db: Session = Depends(get_db_session)):
    updated = update_session(db, session_id, {"freeze_events": freeze_events})
    if not updated:
        raise HTTPException(status_code=404, detail="Session not found")
    return updated
