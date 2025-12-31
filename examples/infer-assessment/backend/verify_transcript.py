from sqlmodel import Session, select, create_engine
from app.schema import Session as SessionModel
# from app.database import sqlite_url # This might be relative, let's hardcode for safety or import if possible
import os

# Assume running from backend/
sqlite_url = "sqlite:///database.db"
engine = create_engine(sqlite_url)

with Session(engine) as session:
    statement = select(SessionModel).order_by(SessionModel.created_at.desc()).limit(1)
    result = session.exec(statement).first()
    
    if result:
        print(f"Session ID: {result.id}")
        print(f"Audio URL: {result.audio_url}")
        print(f"Transcript Count: {len(result.transcript)}")
        print("Transcript Content:")
        for turn in result.transcript:
            print(turn)
    else:
        print("No sessions found.")
