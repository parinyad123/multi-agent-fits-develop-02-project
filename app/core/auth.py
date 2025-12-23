"""
Authentication Dependencies
FastAPI dependencies for JWT verification with flexible auth support

app/core/auth.py
"""

from fastapi import Depends, HTTPException, status, Cookie
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.db.base import get_async_session
from app.db.models import User
from app.services.auth import AuthService

import logging
logger = logging.getLogger(__name__)

# Security scheme (auto_error=False for fallback to cookie)
security = HTTPBearer(auto_error=False)

# async def get_current_user(
#     credentials: HTTPAuthorizationCredentials = Depends(security),
#     session: AsyncSession = Depends(get_async_session)
# ) -> User:
#     """
#     Get current authenticated user from JWT token
    
#     Args:
#         credentials: HTTP Bearer token from Authorization header
#         session: Database session
        
#     Returns:
#         Current User object
        
#     Raises:
#         HTTPException 401: If token is invalid or user not found
#     """
    
#     # Extract token
#     token = credentials.credentials
#     logger.info(f"Received token: {token[:20]}...")
    
#     # Verify token
#     user_id = AuthService.verify_access_token(token)
#     logger.info(f"Decoded user_id: {user_id}")
    
#     if not user_id:
#         logger.info("Token verification failed")
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Invalid or expired token",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
    
#     # Get user from database
#     user = await AuthService.get_user_by_id(session, user_id)
#     logger.info(f"Found user: {user.username if user else 'None'}")
    
#     if not user:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="User not found",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
    
#     # Check if user is active
#     if not user.is_active:
#         raise HTTPException(
#             status_code=status.HTTP_403_FORBIDDEN,
#             detail="User account is inactive"
#         )
    
#     return user

async def get_current_user_flexible(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    access_token_cookie: Optional[str] = Cookie(None, alias="access_token"),
    session: AsyncSession = Depends(get_async_session)
) -> User:
    """
    Flexible authentication: Try Bearer token first, fallback to Cookie
    
    Authentication priority:
    1. Bearer token from Authorization header (for API clients, Postman, mobile apps)
    2. HTTP-only cookie (for web browsers)
    
    Args:
        credentials: Bearer token from Authorization header
        access_token_cookie: Token from HTTP-only cookie
        session: Database session
        
    Returns:
        Current User object
        
    Raises:
        HTTPException 401: If no valid authentication found
        HTTPException 403: If user account is inactive
    """
    
    token = None
    auth_method = None
    
    # Priority 1: Try Bearer token first (for API clients/Postman)
    if credentials and credentials.credentials:
        token = credentials.credentials
        auth_method = "bearer"
        logger.debug(f"Authentication attempt via Bearer token: {token[:20]}...")
    
    # Priority 2: Fallback to cookie (for browsers)
    elif access_token_cookie:
        token = access_token_cookie
        auth_method = "cookie"
        logger.debug(f"Authentication attempt via Cookie: {token[:20]}...")
    
    # No authentication provided
    if not token:
        logger.warning("No authentication credentials provided (no Bearer token or Cookie)")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide Bearer token or valid session cookie.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify token
    user_id = AuthService.verify_access_token(token)
    
    if not user_id:
        logger.warning(f"Invalid or expired token from {auth_method}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user from database
    user = await AuthService.get_user_by_id(session, user_id)
    
    if not user:
        logger.error(f"User not found in database: user_id={user_id}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user is active
    if not user.is_active:
        logger.warning(f"Inactive user attempted access: {user.username}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    
    logger.info(f"✅ Authentication successful via {auth_method}: user={user.username} (id={user.user_id})")
    
    return user



async def get_current_active_user(
    current_user: User = Depends(get_current_user_flexible)
) -> User:
    """
    Get current active user (alias for clarity and backward compatibility)
    
    This is an alias for get_current_user_flexible() to maintain
    backward compatibility with existing code.
    """
    return current_user


# ============================================
# Optional: For future use (public endpoints that work with/without auth)
# ============================================

async def get_optional_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    access_token_cookie: Optional[str] = Cookie(None, alais="access_token"),
    session: AsyncSession = Depends(get_async_session)
) -> Optional[User]:
    """
    Get current user if authenticated, None otherwise
    
    Use for endpoints that work with or without authentication
    (e.g., public content that shows extra features for logged-in users)
    
    Returns:
        User object if authenticated, None otherwise
    """
    
    if not credentials and not access_token_cookie:
        return None
    
    try:
        token = None
        
        if credentials and credentials.credentials:
            token = credentials.credentials
        elif access_token_cookie:
            token = access_token_cookie
        
        if token:
            user_id = AuthService.verify_access_token(token)
            
            if user_id:
                user = await AuthService.get_user_by_id(session, user_id)
                if user and user.is_active:
                    return user
    except Exception as e:
        logger.debug(f"Optional auth failed: {e}")
    
    return None