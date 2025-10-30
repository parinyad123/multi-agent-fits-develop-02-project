"""
multi-agent-fits-dev-02/app/services/astrosage/__init__.py

AstroSage Service - AI assistant for astrophysics questions and analysis interpretation
"""

from app.services.astrosage.client import AstroSageClient
from app.services.astrosage.models import (
    AstroSageRequest,
    AstroSageResponse,
    ExpertiseLevel,
    ConversationPair,
    AstroSageError,
    AstroSageConnectionError,
    AstroSageTimeoutError,
    AstroSageInvalidResponseError
)
from app.services.astrosage.expertise_adapter import ExpertiseAdapter
from app.services.astrosage.conversation_manager import ConversationManager
from app.services.astrosage.prompt_builder import PromptBuilder

__all__ = [
    'AstroSageClient',
    'AstroSageRequest',
    'AstroSageResponse',
    'ExpertiseLevel',
    'ConversationPair',
    'AstroSageError',
    'AstroSageConnectionError',
    'AstroSageTimeoutError',
    'AstroSageInvalidResponseError',
    'ExpertiseAdapter',
    'ConversationManager',
    'PromptBuilder'
]