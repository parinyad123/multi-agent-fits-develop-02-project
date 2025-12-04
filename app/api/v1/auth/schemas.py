"""
Authentication Schemas
Request/Response models for auth endpoints

app/api/v1/auth/schemas.py
"""

from pydantic import BaseModel, EmailStr, Field, field_validator
from uuid import UUID
from datetime import datetime
from typing import Optional
import re

# ==========================================
# Request Schemas
# ==========================================

class RegisterRequest(BaseModel):
    """User registration request"""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=100)
    email: Optional[EmailStr] = None

    @field_validator('username')
    @classmethod
    def validate_username(cls, v: str) -> str:
        """
        Validate username format:
        - Only alphanumeric, underscore, hyphen
        - Must start with letter or number
        - No consecutive special chars
        """
        if not re.match(r'[a-zA-Z0-9][a-zA-Z0-9_-]', v):
            raise ValueError(
                'Username must start with letter/number and contain only '
                'alphanumric characters, underscores, or hyphens'
            )
        
        # No allowed contiguogs special chars 
        if '--' in v or '--' in v or '_-' in v or '-_' in v:
            raise ValueError('Username cannot contain consecutive special characters')

        return v.lower()
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "securepassword123",
                "username": "john_doe"
            }
        }


class LoginRequest(BaseModel):
    """User login request"""
    # email: EmailStr
    username: str
    password: str

    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "securepassword123"
            }
        }


# ==========================================
# Response Schemas
# ==========================================

class TokenResponse(BaseModel):
    """Token response after login"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    
    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 28800
            }
        }


class UserResponse(BaseModel):
    """User information response"""
    user_id: UUID
    username: str
    email: Optional[str]
    is_active: bool
    created_at: datetime
    last_login_at: Optional[datetime]
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "user_id": "123e4567-e89b-12d3-a456-426614174000",
                "email": "user@example.com",
                "username": "john_doe",
                "is_active": True,
                "created_at": "2025-11-09T10:00:00Z",
                "last_login_at": "2025-11-09T14:30:00Z"
            }
        }


class MessageResponse(BaseModel):
    """Generic message response"""
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Operation successful"
            }
        }
