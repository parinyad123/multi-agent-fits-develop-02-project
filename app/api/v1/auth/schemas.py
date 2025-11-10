"""
Authentication Schemas
Request/Response models for auth endpoints

app/api/v1/auth/schemas.py
"""

from pydantic import BaseModel, EmailStr, Field
from uuid import UUID
from datetime import datetime
from typing import Optional

# ==========================================
# Request Schemas
# ==========================================

class RegisterRequest(BaseModel):
    """User registration request"""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    
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
    email: EmailStr
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
    email: str
    username: Optional[str]
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
