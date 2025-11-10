"""
Simple Authentication Service

app/services/auth.py
"""

from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID
import logging

from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.models import User
from app.core.config import settings

logger = logging.getLogger(__name__)

class AuthService:
    """
    Minimal authentication service
    
    Features:
    - User registration
    - Login with email/password
    - JWT token generation/verification
    - Basic user management
    """

    @staticmethod
    async def create_user(
        session: AsyncSession,
        email: str,
        password: str,
        username: Optional[str] = None
    ) -> User:
        """
        Create new user
        
        Args:
            session: Database session
            email: User email (must be unique)
            password: Plain text password (will be hashed)
            username: Optional username
            
        Returns:
            Created User object
            
        Raises:
            ValueError: If email already exists
        """
        
        # Check if email already exists
        result = await session.execute(
            select(User).where(User.email == email)
        )
        existing_user = result.scalar_one_or_none()
        
        if existing_user:
            raise ValueError(f"Email already registered: {email}")
        
        # Create user
        user = User(
            email=email,
            username=username,
            is_active=True
        )
        user.set_password(password)
        
        session.add(user)
        await session.flush()
        
        logger.info(f"User created: {user.user_id} ({email})")
        return user
    
    @staticmethod
    async def authenticate_user(
        session: AsyncSession,
        email: str,
        password: str
    ) -> Optional[User]:
        """
        Authenticate user by email/password
        
        Args:
            session: Database session
            email: User email
            password: Plain text password
            
        Returns:
            User object if authentication successful, None otherwise
        """
        
        # Get user by email
        result = await session.execute(
            select(User).where(User.email == email)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            logger.warning(f"Login failed: email not found ({email})")
            return None
        
        # Check if active
        if not user.is_active:
            logger.warning(f"Login failed: user inactive ({email})")
            return None
        
        # Verify password
        if not user.verify_password(password):
            logger.warning(f"Login failed: wrong password ({email})")
            return None
        
        # Update last login
        user.last_login_at = datetime.now()
        await session.flush()
        
        logger.info(f"User authenticated: {user.user_id} ({email})")
        return user
    
    @staticmethod
    def verify_access_token(token: str) -> Optional[UUID]:
        """
        Verify JWT access token
        
        Args:
            token: JWT token string
            
        Returns:
            User UUID if token is valid, None otherwise
        """
        
        try:
            payload = jwt.decode(
                token,
                settings.secret_key,
                algorithms=[settings.algorithm]
            )
            
            user_id_str: str = payload.get("sub")
            if user_id_str is None:
                return None
            
            return UUID(user_id_str)
            
        except JWTError as e:
            logger.debug(f"Token verification failed: {e}")
            return None
        
    @staticmethod
    async def get_user_by_id(
        session: AsyncSession,
        user_id: UUID
    ) -> Optional[User]:
        """
        Get user by ID
        
        Args:
            session: Database session
            user_id: User UUID
            
        Returns:
            User object if found, None otherwise
        """
        
        result = await session.execute(
            select(User).where(User.user_id == user_id)
        )
        return result.scalar_one_or_none()