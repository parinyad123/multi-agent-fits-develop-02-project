"""
Authentication API Routes
Minimal auth endpoints

app/api/v1/auth/routes.py
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_async_session
from app.services.simple_auth import SimpleAuthService
from app.core.auth import get_current_active_user
from app.api.v1.auth.schemas import (
    RegisterRequest,
    LoginRequest,
    TokenResponse,
    UserResponse,
    MessageResponse
)
from app.core.config import settings

import logging
logger = logging.getLogger(__name__)


router = APIRouter()


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register new user",
    description="Create a new user account with email and password"
)
async def register(
    request: RegisterRequest,
    session: AsyncSession = Depends(get_async_session)
):
    """
    Register a new user
    
    - **email**: Valid email address (must be unique)
    - **password**: Password (minimum 8 characters)
    - **username**: Optional username
    """
    
    try:
        user = await SimpleAuthService.create_user(
            session=session,
            email=request.email,
            password=request.password,
            username=request.username
        )
        
        await session.commit()
        
        logger.info(f"✅ User registered: {user.email}")
        return user
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"❌ Registration error: {e}", exc_info=True)
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login",
    description="Authenticate with email and password to get access token"
)
async def login(
    request: LoginRequest,
    session: AsyncSession = Depends(get_async_session)
):
    """
    Login to get access token
    
    - **email**: User email
    - **password**: User password
    
    Returns JWT access token valid for 8 hours
    """
    
    try:
        # Authenticate user
        user = await SimpleAuthService.authenticate_user(
            session=session,
            email=request.email,
            password=request.password
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create access token
        access_token = SimpleAuthService.create_access_token(user.user_id)
        
        await session.commit()
        
        logger.info(f"✅ User logged in: {user.email}")
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=settings.access_token_expire_hours * 3600  # Convert to seconds
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Login error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user",
    description="Get information about the currently authenticated user"
)
async def get_me(
    current_user = Depends(get_current_active_user)
):
    """
    Get current user information
    
    Requires authentication (Bearer token in Authorization header)
    """
    
    return current_user


@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Logout",
    description="Logout (client should delete token)"
)
async def logout(
    current_user = Depends(get_current_active_user)
):
    """
    Logout current user
    
    Note: Since we're using stateless JWT tokens, logout is handled client-side.
    The client should delete the token from storage.
    
    This endpoint is provided for consistency and future extensions.
    """
    
    logger.info(f"User logged out: {current_user.email}")
    
    return MessageResponse(
        message="Logged out successfully. Please delete your token."
    )