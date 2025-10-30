"""Authentication middleware and dependencies"""

from typing import Optional
from fastapi import Depends, HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..models.user import User, UserRole, TokenData
from ..services.user_service import get_user_service, UserService

# Security scheme for JWT
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    user_service: UserService = Depends(get_user_service),
) -> User:
    """
    Get current user from JWT token

    Args:
        credentials: HTTP Bearer credentials
        user_service: User service dependency

    Returns:
        Current user

    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    token_data = user_service.verify_token(credentials.credentials)
    if token_data is None:
        raise credentials_exception

    user = user_service.get_user(token_data.user_id)
    if user is None:
        raise credentials_exception

    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Get current active user

    Args:
        current_user: Current user dependency

    Returns:
        Current active user

    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user",
        )
    return current_user


def require_role(required_role: UserRole):
    """
    Dependency factory for role-based access control

    Args:
        required_role: Required user role

    Returns:
        Dependency function

    Example:
        @app.get("/admin", dependencies=[Depends(require_role(UserRole.ADMIN))])
    """

    async def role_checker(current_user: User = Depends(get_current_active_user)):
        if current_user.role != required_role and current_user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation requires {required_role.value} role",
            )
        return current_user

    return role_checker


async def verify_api_key(
    x_api_key: Optional[str] = Header(None),
    user_service: UserService = Depends(get_user_service),
) -> Optional[User]:
    """
    Verify API key from header

    Args:
        x_api_key: API key from X-API-Key header
        user_service: User service dependency

    Returns:
        User if API key is valid, None otherwise

    Raises:
        HTTPException: If API key is invalid
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    user = user_service.get_user_by_api_key(x_api_key)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user",
        )

    return user


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    ),
    x_api_key: Optional[str] = Header(None),
    user_service: UserService = Depends(get_user_service),
) -> Optional[User]:
    """
    Get user from JWT token or API key (optional)

    Args:
        credentials: HTTP Bearer credentials (optional)
        x_api_key: API key from header (optional)
        user_service: User service dependency

    Returns:
        User if authenticated, None otherwise
    """
    # Try JWT token first
    if credentials:
        token_data = user_service.verify_token(credentials.credentials)
        if token_data:
            user = user_service.get_user(token_data.user_id)
            if user and user.is_active:
                return user

    # Try API key
    if x_api_key:
        user = user_service.get_user_by_api_key(x_api_key)
        if user and user.is_active:
            return user

    return None
