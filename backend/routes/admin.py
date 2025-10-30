"""Admin API routes for user and session management"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from ..models.user import User, UserCreate, UserUpdate, UserRole, Token
from ..models.session import Session, SessionStatus, SessionListResponse, SessionMetrics
from ..services.user_service import get_user_service, UserService
from ..services.session_service import get_session_service, SessionService
from ..services.pipeline_service import get_pipeline_service, PipelineService
from ..middleware.auth import (
    get_current_active_user,
    require_role,
)

router = APIRouter(prefix="/api/admin", tags=["admin"])


# ==================== Authentication ====================


@router.post("/auth/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    user_service: UserService = Depends(get_user_service),
):
    """
    Login with username and password to get JWT token

    Args:
        form_data: OAuth2 password form
        user_service: User service dependency

    Returns:
        JWT access token
    """
    user = user_service.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = user_service.create_access_token(user)
    return token


@router.get("/auth/me", response_model=User)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user),
):
    """Get current authenticated user information"""
    return current_user


# ==================== User Management ====================


@router.post("/users", response_model=User, dependencies=[Depends(require_role(UserRole.ADMIN))])
async def create_user(
    user_create: UserCreate,
    user_service: UserService = Depends(get_user_service),
    current_user: User = Depends(get_current_active_user),
):
    """
    Create a new user (Admin only)

    Args:
        user_create: User creation data
        user_service: User service dependency
        current_user: Current authenticated user

    Returns:
        Created user
    """
    try:
        user = user_service.create_user(user_create)
        return user
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("/users", response_model=List[User])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    user_service: UserService = Depends(get_user_service),
    current_user: User = Depends(get_current_active_user),
):
    """
    List all users with pagination

    Args:
        skip: Number of users to skip
        limit: Maximum number of users to return
        user_service: User service dependency
        current_user: Current authenticated user

    Returns:
        List of users
    """
    # Non-admin users can only see themselves
    if current_user.role != UserRole.ADMIN:
        return [current_user]

    users = user_service.list_users(skip=skip, limit=limit)
    return users


@router.get("/users/{user_id}", response_model=User)
async def get_user(
    user_id: str,
    user_service: UserService = Depends(get_user_service),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get user by ID

    Args:
        user_id: User ID
        user_service: User service dependency
        current_user: Current authenticated user

    Returns:
        User
    """
    # Non-admin users can only access their own data
    if current_user.role != UserRole.ADMIN and current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this user",
        )

    user = user_service.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return user


@router.patch("/users/{user_id}", response_model=User)
async def update_user(
    user_id: str,
    user_update: UserUpdate,
    user_service: UserService = Depends(get_user_service),
    current_user: User = Depends(get_current_active_user),
):
    """
    Update user

    Args:
        user_id: User ID
        user_update: User update data
        user_service: User service dependency
        current_user: Current authenticated user

    Returns:
        Updated user
    """
    # Non-admin users can only update themselves (except role)
    if current_user.role != UserRole.ADMIN:
        if current_user.id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to update this user",
            )
        if user_update.role is not None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot change own role",
            )

    user = user_service.update_user(user_id, user_update)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return user


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    user_service: UserService = Depends(get_user_service),
    current_user: User = Depends(require_role(UserRole.ADMIN)),
):
    """
    Delete user (Admin only)

    Args:
        user_id: User ID
        user_service: User service dependency
        current_user: Current authenticated user

    Returns:
        Success message
    """
    if not user_service.delete_user(user_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return {"message": "User deleted successfully"}


@router.post("/users/{user_id}/regenerate-api-key", response_model=dict)
async def regenerate_api_key(
    user_id: str,
    user_service: UserService = Depends(get_user_service),
    current_user: User = Depends(get_current_active_user),
):
    """
    Regenerate API key for user

    Args:
        user_id: User ID
        user_service: User service dependency
        current_user: Current authenticated user

    Returns:
        New API key
    """
    # Users can regenerate their own API key
    if current_user.role != UserRole.ADMIN and current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to regenerate API key for this user",
        )

    api_key = user_service.regenerate_api_key(user_id)
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return {"api_key": api_key}


# ==================== Session Management ====================


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    user_id: Optional[str] = None,
    status_filter: Optional[SessionStatus] = None,
    skip: int = 0,
    limit: int = 100,
    session_service: SessionService = Depends(get_session_service),
    current_user: User = Depends(get_current_active_user),
):
    """
    List sessions with optional filters

    Args:
        user_id: Filter by user ID
        status_filter: Filter by session status
        skip: Number of sessions to skip
        limit: Maximum number of sessions to return
        session_service: Session service dependency
        current_user: Current authenticated user

    Returns:
        Paginated list of sessions
    """
    # Non-admin users can only see their own sessions
    if current_user.role != UserRole.ADMIN:
        user_id = current_user.id

    sessions = session_service.list_sessions(
        user_id=user_id,
        status=status_filter,
        skip=skip,
        limit=limit,
    )

    total = session_service.get_session_count()

    return SessionListResponse(
        sessions=sessions,
        total=total,
        page=skip // limit if limit > 0 else 0,
        page_size=limit,
    )


@router.get("/sessions/{session_id}", response_model=Session)
async def get_session(
    session_id: str,
    session_service: SessionService = Depends(get_session_service),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get session by ID

    Args:
        session_id: Session ID
        session_service: Session service dependency
        current_user: Current authenticated user

    Returns:
        Session
    """
    session = session_service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    # Non-admin users can only access their own sessions
    if current_user.role != UserRole.ADMIN and session.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this session",
        )

    return session


@router.get("/sessions/{session_id}/metrics", response_model=SessionMetrics)
async def get_session_metrics(
    session_id: str,
    session_service: SessionService = Depends(get_session_service),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get real-time session metrics

    Args:
        session_id: Session ID
        session_service: Session service dependency
        current_user: Current authenticated user

    Returns:
        Session metrics
    """
    session = session_service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    # Non-admin users can only access their own sessions
    if current_user.role != UserRole.ADMIN and session.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this session",
        )

    metrics = session_service.get_session_metrics(session_id)
    if not metrics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session metrics not found",
        )

    return metrics


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    session_service: SessionService = Depends(get_session_service),
    pipeline_service: PipelineService = Depends(get_pipeline_service),
    current_user: User = Depends(require_role(UserRole.ADMIN)),
):
    """
    Delete session (Admin only)

    Args:
        session_id: Session ID
        session_service: Session service dependency
        pipeline_service: Pipeline service dependency
        current_user: Current authenticated user

    Returns:
        Success message
    """
    # Stop pipeline if active
    await pipeline_service.stop_pipeline(session_id)

    if not session_service.delete_session(session_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    return {"message": "Session deleted successfully"}


# ==================== System Statistics ====================


@router.get("/stats")
async def get_system_stats(
    session_service: SessionService = Depends(get_session_service),
    pipeline_service: PipelineService = Depends(get_pipeline_service),
    user_service: UserService = Depends(get_user_service),
    current_user: User = Depends(require_role(UserRole.ADMIN)),
):
    """
    Get system statistics (Admin only)

    Args:
        session_service: Session service dependency
        pipeline_service: Pipeline service dependency
        user_service: User service dependency
        current_user: Current authenticated user

    Returns:
        System statistics
    """
    total_sessions = session_service.get_session_count()
    active_sessions = session_service.get_active_session_count()
    total_users = len(user_service.list_users(limit=10000))

    return {
        "total_sessions": total_sessions,
        "active_sessions": active_sessions,
        "total_users": total_users,
        "active_pipelines": pipeline_service.get_session_count(),
    }
