# app/agents/rewrite/session_title_generator.py

"""
Session Title Generator
Hybrid approach: GPT with rule-based fallback
"""

from openai import AsyncOpenAI
from typing import Optional, Tuple
import logging
from datetime import datetime
import asyncio

from app.core.config import settings

logger = logging.getLogger(__name__)


class SessionTitleGenerator:
    """
    Generate session titles from first user query
    """

    # GPT model configuration
    MODEL = "gpt-4o-mini"
    TEMPERATURE = 0.3
    MAX_TOKENS = 30 # short title only
    TIMEOUT = 3.0   # 3 seconds timeout

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        logger.info("SessionTitleGenerator initialized")

    async def generate_title(
            self,
            user_query: str,
            routing_strategy: str,
            analysis_types: Optional[list] = None,
            filename: Optional[str] = None, 
            use_gpt: bool = True
    ) -> str:
        """
        Generate session title

        Args:
            user_query: First user message
            routing_strategy: "analysis", "astrosage", or "mixed"
            analysis_types: List of analysis types (if analysis)
            filename: FITS filename (if available)
            use_gpt: Try GPT first (with fallback)      
        """

        # Try GPT first
        if use_gpt:
            try:
                title = await asyncio.wait_for(
                    self._generate_gpt_title(
                        user_query,
                        routing_strategy,
                        analysis_types,
                        filename
                    ),
                    timeout=self.TIMEOUT
                )

                logger.info(f"GPT title gernerated: '{title}'")
                return (title, "gpt")

            except asyncio.TimeoutError:
                logger.warning(
                    f"GPT title generation timeout ({self.TIMEOUT}s), "
                    "using rule-based fallback"
                )

            except Exception as e:
                logger.error(f"GPT title generation failed: {e}, using fallback")

        # Fallback to rule-based
        title = self._generate_rule_bases_title(
            user_query,
            routing_strategy,
            analysis_types,
            filename
        )

        logger.info(f"Rule-based title generated: '{title}'")
        return (title, "rule_based")
    
    async def _generate_gpt_title(
            self, 
            user_query: str,
            routing_strategy: str,
            analysis_types: Optional[list],
            filename: Optional[str]
    ) -> str:
        """Gerate title using GPT"""

        # Build prompt
        prompt = self._build_gpt_prompt(
            user_query,
            routing_strategy,
            analysis_types,
            filename
        )

        # Call GPT
        response = await self.client.chat.completions.create(
            model=self.MODEL,
            massage=[
                {
                    "role": "system",
                    "content": (
                        "You are a title generator for X-ray astronomy analysis sessions. "
                        "Generate SHORT, DESCRIPTIVE titles (max 50 characters). "
                        "Use technical terms when appropriate. "
                        "Format: 'Action + Object' (e.g., 'Power Law Fit', 'PSD Analysis'). "
                        "NO quotes, NO colons, NO prefixes like 'Title:'."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=self.TEMPERATURE,
            max_tokens=self.MAX_TOKENS
        )

        # Extract and clean title
        title = response.choices[0].message.content.strip()

        # Remove common prefixes/suffixes
        # title = title.replace("Title:", "").strip()
        # title = title.strip('"\'')

        return title

    def _build_gpt_prompt(
        self,
        user_query: str,
        routing_strategy: str,
        analysis_types: Optional[list],
        filename: Optional[str]
    ) -> str:
        """Build prompt for GPT"""
        
        prompt_parts = []
        
        # User query
        prompt_parts.append(f"User query: {user_query}")
        
        # Context
        if routing_strategy:
            prompt_parts.append(f"Task type: {routing_strategy}")
        
        if analysis_types:
            prompt_parts.append(f"Analysis: {', '.join(analysis_types)}")
        
        if filename:
            prompt_parts.append(f"File: {filename}")
        
        prompt_parts.append("\nGenerate a short, descriptive title (max 50 chars):")
        
        return "\n".join(prompt_parts)



# Singleton instance
_title_generator = None

def get_title_generator() -> SessionTitleGenerator:
    """Get singleton title generator instance"""
    global _title_generator
    
    if _title_generator is None:
        _title_generator = SessionTitleGenerator()
    
    return _title_generator