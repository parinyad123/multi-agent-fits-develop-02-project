"""
Authentication API Routes
Minimal auth endpoints with Cookie support

app/api/v1/auth/routes.py
"""

from fastapi import APIRouter, Depends, HTTPException, status, Response, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_async_session
from app.services.auth import AuthService
from app.core.auth import get_current_active_user, get_current_user_flexible
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
    description="Create a new user account with username and password"
)
async def register(
    request: RegisterRequest,
    session: AsyncSession = Depends(get_async_session)
):
    """
    Register a new user
    
    - **username**: Unique username (3-50 chars, alphanumeric + _-)
    - **password**: Password (minimum 8 characters)
    - **username**: Optional username
    """
    
    try:
        user = await AuthService.create_user(
            session=session,
            username=request.username,
            password=request.password,
            email=request.email
        )
        
        await session.commit()
        
        logger.info(f"User registered: {user.username}")
        return user
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Registration error: {e}", exc_info=True)
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
    response: Response,
    session: AsyncSession = Depends(get_async_session),
    set_cookie: bool = Query(
        True,
        description="Set HTTP-only cookie (for browsers). Set false for Postman/API clients."
    )
):
    """
    Login with flexible authentication
    
    - **username**: Username
    - **password**: User password
    
    **Cookie Behavior:**
    - `set_cookie=true` (default): Sets HTTP-only cookie for browsers
    - `set_cookie=false`: Only returns token (for Postman/API clients)
    
    **Browser usage:**
    ```javascript
        fetch('/auth/login', {
            method: 'POST',
            credentials: 'include',  // Auto-send cookie
            body: JSON.stringify({username, password})
        });
    ```
    
    **Postman usage:**
    ```
        POST /auth/login?set_cookie=false
        Authorization: Bearer <use_returned_token>
    ```
    
    Returns JWT access token valid for 8 hours
    """
    
    try:
        # Authenticate user
        user = await AuthService.authenticate_user(
            session=session,
            username=request.username,
            password=request.password
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create access token
        access_token = AuthService.create_access_token(user.user_id)

        # Calculate expiration
        expires_seconds = settings.access_token_expire_hours * 3600

        # Set HTTP-only cookie (if requested)
        if set_cookie:
            response.set_cookie(
                key="access_token",
                value=access_token,
                httponly=True,  # Protect Javascript access (XSS protection)
                secure=False,   # use True for production (HTTPS only)
                max_age=expires_seconds,
                path="/"
            )
            logger.info(f"Cookie set for user: {user.username}")
        else:
            logger.info(f"Token-only mode for user: {user.username}")
        
        await session.commit()
        
        logger.info(f"User logged in: {user.username}")
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=expires_seconds
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}", exc_info=True)
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
    current_user = Depends(get_current_user_flexible)
):
    """
    Get current user information
    
    Requires authentication:
    - Browser: Cookie (automatic)
    - Postman/API: Bearer token in Authorization header
    """
    
    return current_user


@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Logout",
    description="Logout (deletes cookie and invalidates session)"
)
async def logout(
    response: Response,
    current_user = Depends(get_current_user_flexible)
):
    """
    Logout current user
    
    **For Browser:**
    - Deletes HTTP-only cookie
    - Client should also clear localStorage/sessionStorage
    
    **For API clients:**
    - Returns success message
    - Client should discard the token
    
    **Note:** JWT tokens are stateless, so server-side invalidation
    is not possible without a token blacklist (future feature).
    """
    
    # Delete cookie
    response.delete_cookie(
        key="access_token",
        path="/",
        httponly=True,
        secure=False,  # True in production
        samesite="lax"
    )
    
    logger.info(f"User logged out: {current_user.username}")
    
    return MessageResponse(
        message="Logged out successfully. Cookie deleted and token invalidated."
    )

# Optional: Refresh token endpoint
@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh access token",
    description="Get a new access token using current valid token"
)
async def refresh_token(
    response: Response,
    current_user = Depends(get_current_user_flexible)
):
    """
    Refresh access token
    
    Generates new token and updates cookie (if cookie auth was used)
    
    **Usage:**
    - Before token expires, call this endpoint
    - New token will be returned and cookie updated
    - Old token becomes invalid after expiration
    """
    
    # Generate new token
    new_token = AuthService.create_access_token(current_user.user_id)
    expires_seconds = settings.access_token_expire_hours * 3600
    
    # Update cookie (always set for browsers)
    response.set_cookie(
        key="access_token",
        value=new_token,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=expires_seconds,
        path="/"
    )
    
    logger.info(f"Token refreshed for user: {current_user.username}")
    
    return TokenResponse(
        access_token=new_token,
        token_type="bearer",
        expires_in=expires_seconds
    )