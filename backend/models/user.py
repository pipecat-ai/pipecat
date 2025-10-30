"""User models for authentication and management"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class UserRole(str, Enum):
    """User roles for permission management"""
    ADMIN = "admin"
    USER = "user"
    DEVELOPER = "developer"


class UserBase(BaseModel):
    """Base user model with common fields"""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    full_name: Optional[str] = None
    role: UserRole = UserRole.USER
    is_active: bool = True


class UserCreate(UserBase):
    """User creation model"""
    password: str = Field(..., min_length=8)


class UserUpdate(BaseModel):
    """User update model - all fields optional"""
    email: Optional[EmailStr] = None
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    full_name: Optional[str] = None
    password: Optional[str] = Field(None, min_length=8)
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


class User(UserBase):
    """User model returned to clients"""
    id: str
    api_key: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        from_attributes = True


class UserInDB(User):
    """User model stored in database"""
    hashed_password: str


class Token(BaseModel):
    """JWT token response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Token payload data"""
    user_id: str
    username: str
    role: UserRole
