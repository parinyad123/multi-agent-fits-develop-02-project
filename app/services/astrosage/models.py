"""
multi-agent-fits-dev-02/app/services/astrosage/models.py

Data models for AstroSage Service
"""

from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

# ==========================================
# Expertise Levels
# ==========================================

class ExpertiseLevel(str, Enum):
    """User expertise levels for tailored responses"""
    BEGINNER = "beginner"           # High school / introductory
    INTERMEDIATE = "intermediate"   # Undergraduate / amateur astronomer
    ADVANCED = "advanced"           # Graduate student / professional
    EXPERT = "expert"               # Research scientist / postdoc


# ==========================================
# Conversation Models
# ==========================================

class ConversationPair(BaseModel):
    """A pair of user-assistant messages"""
    user_message: str
    assistant_message: str
    timestamp: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_message": "What causes red shift in galaxies?",
                "assistant_message": "Red shift occurs when light from distant galaxies...",
                "timestamp": "2025-01-15T10:30:00Z"
            }
        }


# ==========================================
# Request Models
# ==========================================

class AstroSageRequest(BaseModel):
    """Request to AstroSage service"""
    user_id: UUID
    session_id: str
    user_query: str
    expertise_level: ExpertiseLevel = ExpertiseLevel.INTERMEDIATE
    
    # Optional context
    conversation_history: Optional[List[ConversationPair]] = None
    analysis_results: Optional[Dict[str, Any]] = None
    
    # LLM configuration overrides
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "123e4567-e89b-12d3-a456-426614174000",
                "session_id": "session_abc123",
                "user_query": "What is the significance of a power law index of 1.5?",
                "expertise_level": "intermediate",
                "analysis_results": {
                    "power_law": {"A": 1.23e-3, "b": 1.52, "n": 2.1e-5}
                }
            }
        }

# ==========================================
# Response Models
# ==========================================

class AstroSageResponse(BaseModel):
    """Response from AstroSage service"""
    response_id: UUID = Field(default_factory=uuid4)
    content: str
    model_used: str = "AstroSage-Llama-3.1-8B"
    
    # Metadata
    tokens_used: Optional[int] = None
    response_time: float
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Status
    success: bool = True
    error: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "response_id": "789e0123-e89b-12d3-a456-426614174000",
                "content": "A power law index of 1.5 is typical for...",
                "model_used": "AstroSage-Llama-3.1-8B",
                "tokens_used": 450,
                "response_time": 2.34,
                "success": True
            }
        }

# ==========================================
# LLM Configuration
# ==========================================

class LLMConfig(BaseModel):
    """Configuration for LLM API calls"""
    temperature: float = 0.2
    max_tokens: int = 600
    top_p: float = 0.95
    repeat_penalty: float = 1.05
    presence_penalty: float = 0.1
    frequency_penalty: float = 0.05
    
    def to_api_dict(self) -> dict:
        """Convert to API request format"""
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty
        }


# ==========================================
# Error Models
# ==========================================

class AstroSageError(Exception):
    """Base exception for AstroSage service"""
    pass


class AstroSageConnectionError(AstroSageError):
    """Cannot connect to AstroSage service"""
    pass


class AstroSageTimeoutError(AstroSageError):
    """Request timed out"""
    pass


class AstroSageInvalidResponseError(AstroSageError):
    """Invalid response from service"""
    pass