"""
multi-agent-fits-dev-02/app/services/astrosage/expertise_adapter.py

Handle user expertise levels and adapt responses accordingly
"""

import logging
from typing import Dict, Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.services.astrosage.models import ExpertiseLevel, LLMConfig
from app.db.models import User

logger = logging.getLogger(__name__)

class ExpertiseAdapter:
    """
    Manage user expertise levels and provide appropriate configurations
    """

    # Configuration for each expertise level
    EXPERTISE_CONFIGS = {
        ExpertiseLevel.BEGINNER: {
            "temperature": 0.7,
            "max_tokens": 800,
            "top_p": 0.95,
            "repeat_penalty": 1.05,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.05,
            "guidelines": {
                "include_analogies": True,
                "simplify_equations": True,
                "reference_level": "popular science",
                "tone": "friendly and encouraging"
            }
        },
        ExpertiseLevel.INTERMEDIATE: {
            "temperature": 0.5,
            "max_tokens": 1000,
            "top_p": 0.92,
            "repeat_penalty": 1.05,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.05,
            "guidelines": {
                "include_analogies": False,
                "simplify_equations": False,
                "reference_level": "undergraduate textbook",
                "tone": "clear and informative"
            }
        },
        ExpertiseLevel.ADVANCED: {
            "temperature": 0.3,
            "max_tokens": 1200,
            "top_p": 0.90,
            "repeat_penalty": 1.05,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.05,
            "guidelines": {
                "include_citations": True,
                "reference_level": "research papers",
                "tone": "technical and precise"
            }
        },
        ExpertiseLevel.EXPERT: {
            "temperature": 0.2,
            "max_tokens": 1500,
            "top_p": 0.88,
            "repeat_penalty": 1.05,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.05,
            "guidelines": {
                "include_citations": True,
                "technical_depth": "maximum",
                "reference_level": "peer review",
                "tone": "colleague-level discourse"
            }
        }
    }

    @classmethod
    async def get_user_expertise(
        cls, 
        user_id: UUID, 
        db_session: AsyncSession
    ) -> ExpertiseLevel:
        """
        Get user's expertise level from database
        
        Args:
            user_id: User UUID
            db_session: Database session
        
        Returns:
            ExpertiseLevel (default: INTERMEDIATE)
        """
        try:
            result = await db_session.execute(
                select(User).where(User.user_id == user_id)
            )
            user = result.scalar_one_or_none()
            
            if not user:
                logger.warning(f"User {user_id} not found, using default expertise level")
                return ExpertiseLevel.INTERMEDIATE
            
            # Get from preferences JSONB field
            if user.preferences and 'expertise_level' in user.preferences:
                level_str = user.preferences['expertise_level']
                try:
                    return ExpertiseLevel(level_str)
                except ValueError:
                    logger.warning(
                        f"Invalid expertise level '{level_str}' for user {user_id}, "
                        f"using default"
                    )
                    return ExpertiseLevel.INTERMEDIATE
            
            # Default level
            logger.info(f"No expertise level set for user {user_id}, using INTERMEDIATE")
            return ExpertiseLevel.INTERMEDIATE
            
        except Exception as e:
            logger.error(f"Error getting user expertise for {user_id}: {e}", exc_info=True)
            return ExpertiseLevel.INTERMEDIATE
    
    @classmethod
    async def update_user_expertise(
        cls,
        user_id: UUID,
        level: ExpertiseLevel,
        db_session: AsyncSession
    ) -> bool:
        """
        Update user's expertise level
        
        Args:
            user_id: User UUID
            level: New expertise level
            db_session: Database session
        
        Returns:
            True if successful, False otherwise
        """
        try:
            result = await db_session.execute(
                select(User).where(User.user_id == user_id)
            )
            user = result.scalar_one_or_none()
            
            if not user:
                logger.error(f"User {user_id} not found, cannot update expertise")
                return False
            
            # Initialize preferences if needed
            if user.preferences is None:
                user.preferences = {}
            
            # Update expertise level
            user.preferences['expertise_level'] = level.value
            await db_session.flush()
            
            logger.info(f"Updated expertise level for user {user_id} to {level.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating expertise for {user_id}: {e}", exc_info=True)
            return False
    
    @classmethod
    def get_llm_config(cls, level: ExpertiseLevel) -> LLMConfig:
        """
        Get LLM configuration for expertise level
        
        Args:
            level: Expertise level
        
        Returns:
            LLMConfig object
        """
        config_dict = cls.EXPERTISE_CONFIGS.get(level, cls.EXPERTISE_CONFIGS[ExpertiseLevel.INTERMEDIATE])
        
        return LLMConfig(
            temperature=config_dict["temperature"],
            max_tokens=config_dict["max_tokens"],
            top_p=config_dict["top_p"],
            repeat_penalty=config_dict["repeat_penalty"],
            presence_penalty=config_dict["presence_penalty"],
            frequency_penalty=config_dict["frequency_penalty"]
        )
    
    @classmethod
    def get_guidelines(cls, level: ExpertiseLevel) -> Dict[str, Any]:
        """
        Get response guidelines for expertise level
        
        Args:
            level: Expertise level
        
        Returns:
            Dictionary of guidelines
        """
        config = cls.EXPERTISE_CONFIGS.get(level, cls.EXPERTISE_CONFIGS[ExpertiseLevel.INTERMEDIATE])
        return config.get("guidelines", {})
    
    @classmethod
    def get_system_prompt_modifier(cls, level: ExpertiseLevel) -> str:
        """
        Get system prompt modification text for expertise level
        
        Args:
            level: Expertise level
        
        Returns:
            Additional system prompt text
        """
        modifiers = {
            ExpertiseLevel.BEGINNER: """
                Your audience is NEW to astrophysics. Please:
                - Use simple, everyday analogies (e.g., "think of a black hole like a drain in a bathtub")
                - Avoid or explain all technical jargon
                - Break complex ideas into small, digestible steps
                - Encourage curiosity and questions
                - Use encouraging, friendly language
                """,
            ExpertiseLevel.INTERMEDIATE: """
                Your audience has BASIC astrophysics knowledge (undergraduate level). Please:
                - Use standard scientific terminology freely
                - Provide equations when relevant, with brief explanations
                - Reference observational missions and data when appropriate
                - Balance technical accuracy with accessibility
                """,
            ExpertiseLevel.ADVANCED: """
                Your audience has ADVANCED astrophysics knowledge (graduate student level). Please:
                - Use technical language and notation freely
                - Discuss cutting-edge research and ongoing debates
                - Reference recent papers and missions by name
                - Assume strong mathematical and physics background
                """,
            ExpertiseLevel.EXPERT: """
                Your audience is at RESEARCH level (postdoc/professor). Please:
                - Engage as a peer-level colleague
                - Reference specific papers, instruments, and methodologies
                - Discuss error analysis, systematics, and caveats
                - Critique assumptions and suggest research directions
                """
        }
        
        return modifiers.get(level, modifiers[ExpertiseLevel.INTERMEDIATE])