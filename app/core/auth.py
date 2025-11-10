"""
Authentication Dependencies
FastAPI dependencies for JWT verification

app/core/auth.py
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.db.base import get_async_session
from app.db.models import User
from app.services.auth import AuthService

import logging
logger = logging.getLogger(__name__)


# Security scheme
security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    session: AsyncSession = Depends(get_async_session)
) -> User:
    """
    Get current authenticated user from JWT token
    
    Args:
        credentials: HTTP Bearer token from Authorization header
        session: Database session
        
    Returns:
        Current User object
        
    Raises:
        HTTPException 401: If token is invalid or user not found
    """
    
    # Extract token
    token = credentials.credentials
    
    # Verify token
    user_id = AuthService.verify_access_token(token)
    
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user from database
    user = await AuthService.get_user_by_id(session, user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current active user
    Alias for get_current_user (for clarity)
    """
    return current_user


# Optional: For future use
async def get_optional_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    session: AsyncSession = Depends(get_async_session)
) -> Optional[User]:
    """
    Get current user if authenticated, None otherwise
    Use for endpoints that work with or without authentication
    """
    
    if not credentials:
        return None
    
    try:
        token = credentials.credentials
        user_id = AuthService.verify_access_token(token)
        
        if user_id:
            user = await AuthService.get_user_by_id(session, user_id)
            if user and user.is_active:
                return user
    except Exception as e:
        logger.debug(f"Optional auth failed: {e}")
    
    return None