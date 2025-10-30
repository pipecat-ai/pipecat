"""User service for authentication and user management"""

import uuid
import secrets
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import JWTError, jwt

from ..models.user import (
    User,
    UserCreate,
    UserUpdate,
    UserInDB,
    UserRole,
    Token,
    TokenData,
)
from ..config import settings

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class UserService:
    """Service for user management and authentication"""

    def __init__(self):
        # In-memory storage (replace with database in production)
        self.users: Dict[str, UserInDB] = {}
        self.api_keys: Dict[str, str] = {}  # api_key -> user_id

        # Create default admin user
        self._create_default_admin()

    def _create_default_admin(self):
        """Create default admin user for initial setup"""
        admin_id = str(uuid.uuid4())
        admin = UserInDB(
            id=admin_id,
            email="admin@localhost",
            username="admin",
            full_name="Admin User",
            role=UserRole.ADMIN,
            is_active=True,
            hashed_password=self.hash_password("admin123"),
            api_key=self.generate_api_key(),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metadata={},
        )
        self.users[admin_id] = admin
        self.api_keys[admin.api_key] = admin_id

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def generate_api_key() -> str:
        """Generate a secure API key"""
        return f"pk_{secrets.token_urlsafe(32)}"

    def create_user(self, user_create: UserCreate) -> User:
        """Create a new user"""
        # Check if username or email already exists
        for user in self.users.values():
            if user.username == user_create.username:
                raise ValueError("Username already exists")
            if user.email == user_create.email:
                raise ValueError("Email already exists")

        # Create user
        user_id = str(uuid.uuid4())
        api_key = self.generate_api_key()

        user_in_db = UserInDB(
            id=user_id,
            email=user_create.email,
            username=user_create.username,
            full_name=user_create.full_name,
            role=user_create.role,
            is_active=user_create.is_active,
            hashed_password=self.hash_password(user_create.password),
            api_key=api_key,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metadata={},
        )

        self.users[user_id] = user_in_db
        self.api_keys[api_key] = user_id

        return User(**user_in_db.dict())

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        user = self.users.get(user_id)
        if user:
            return User(**user.dict())
        return None

    def get_user_by_username(self, username: str) -> Optional[UserInDB]:
        """Get user by username"""
        for user in self.users.values():
            if user.username == username:
                return user
        return None

    def get_user_by_api_key(self, api_key: str) -> Optional[User]:
        """Get user by API key"""
        user_id = self.api_keys.get(api_key)
        if user_id:
            return self.get_user(user_id)
        return None

    def update_user(self, user_id: str, user_update: UserUpdate) -> Optional[User]:
        """Update user"""
        if user_id not in self.users:
            return None

        user = self.users[user_id]
        update_data = user_update.dict(exclude_unset=True)

        # Hash password if updated
        if "password" in update_data:
            update_data["hashed_password"] = self.hash_password(update_data.pop("password"))

        # Update fields
        for field, value in update_data.items():
            setattr(user, field, value)

        user.updated_at = datetime.utcnow()
        return User(**user.dict())

    def delete_user(self, user_id: str) -> bool:
        """Delete user"""
        if user_id in self.users:
            user = self.users[user_id]
            # Remove API key mapping
            if user.api_key in self.api_keys:
                del self.api_keys[user.api_key]
            # Remove user
            del self.users[user_id]
            return True
        return False

    def list_users(self, skip: int = 0, limit: int = 100) -> List[User]:
        """List users with pagination"""
        users = list(self.users.values())
        return [User(**u.dict()) for u in users[skip : skip + limit]]

    def authenticate_user(self, username: str, password: str) -> Optional[UserInDB]:
        """Authenticate user with username and password"""
        user = self.get_user_by_username(username)
        if not user:
            return None
        if not self.verify_password(password, user.hashed_password):
            return None
        if not user.is_active:
            return None
        return user

    def create_access_token(
        self, user: UserInDB, expires_delta: Optional[timedelta] = None
    ) -> Token:
        """Create JWT access token"""
        if expires_delta is None:
            expires_delta = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

        expire = datetime.utcnow() + expires_delta

        token_data = {
            "sub": user.id,
            "username": user.username,
            "role": user.role.value,
            "exp": expire,
        }

        access_token = jwt.encode(
            token_data, settings.SECRET_KEY, algorithm=settings.ALGORITHM
        )

        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=int(expires_delta.total_seconds()),
        )

    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify JWT token and return token data"""
        try:
            payload = jwt.decode(
                token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
            )
            user_id: str = payload.get("sub")
            username: str = payload.get("username")
            role: str = payload.get("role")

            if user_id is None or username is None:
                return None

            return TokenData(user_id=user_id, username=username, role=UserRole(role))
        except JWTError:
            return None

    def regenerate_api_key(self, user_id: str) -> Optional[str]:
        """Regenerate API key for user"""
        if user_id not in self.users:
            return None

        user = self.users[user_id]

        # Remove old API key
        if user.api_key in self.api_keys:
            del self.api_keys[user.api_key]

        # Generate new API key
        new_api_key = self.generate_api_key()
        user.api_key = new_api_key
        user.updated_at = datetime.utcnow()

        # Add new API key mapping
        self.api_keys[new_api_key] = user_id

        return new_api_key


# Global user service instance
_user_service: Optional[UserService] = None


def get_user_service() -> UserService:
    """Get or create the global user service instance"""
    global _user_service
    if _user_service is None:
        _user_service = UserService()
    return _user_service
