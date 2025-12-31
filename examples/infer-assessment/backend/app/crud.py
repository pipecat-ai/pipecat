from typing import Optional
from sqlmodel import Session, select
from app.schema import Session as SessionModel
from uuid import UUID

def create_session(db: Session, session_data: SessionModel) -> SessionModel:
    db.add(session_data)
    db.commit()
    db.refresh(session_data)
    return session_data

def get_session(db: Session, session_id: UUID) -> Optional[SessionModel]:
    statement = select(SessionModel).where(SessionModel.id == session_id)
    return db.exec(statement).first()

def update_session(db: Session, session_id: UUID, updates: dict) -> SessionModel | None:
    session = get_session(db, session_id)
    if not session:
        return None
    
    for key, value in updates.items():
        setattr(session, key, value)
        
    db.add(session)
    db.commit()
    db.refresh(session)
    return session

def get_all_sessions(db: Session) -> list[SessionModel]:
    statement = select(SessionModel).order_by(SessionModel.created_at.desc())
    return db.exec(statement).all()