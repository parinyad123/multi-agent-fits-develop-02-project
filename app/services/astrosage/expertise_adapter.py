"""
multi-agent-fits-dev-02/app/services/astrosage/expertise_adapter.py

IMPROVED VERSION: Enhanced for longer, detailed responses with LaTeX support
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
    
    IMPROVEMENTS:
    - Extended max_tokens for longer, more detailed responses
    - Adjusted temperature for richer content generation
    - Added LaTeX equation formatting guidelines
    - Enhanced response depth based on expertise
    """

    # IMPROVED: Extended configuration for each expertise level
    EXPERTISE_CONFIGS = {
        ExpertiseLevel.BEGINNER: {
            "temperature": 0.35,      # ชัดเจน, ตรงประเด็น
            "max_tokens": 2000,
            "top_p": 0.90,           # จำกัด vocabulary
            "repeat_penalty": 1.1,    # หลีกเลี่ยงการซ้ำ
            "presence_penalty": 0.2,
            "frequency_penalty": 0.15,
            "guidelines": {
                "response_length": "comprehensive",
                "include_analogies": True,
                "simplify_equations": True,
                "use_latex": True,  # Enable LaTeX for simple equations
                "equation_complexity": "basic",  # Simple LaTeX only
                "reference_level": "popular science",
                "tone": "friendly and encouraging",
                "depth": "step-by-step with examples",
                "sections": [
                    "Simple explanation",
                    "Everyday analogy",
                    "Basic equation (in LaTeX)",
                    "What it means for your data",
                    "Next steps"
                ]
            }
        },
        ExpertiseLevel.INTERMEDIATE: {
            "temperature": 0.5, 
            "max_tokens": 2500,
            "top_p": 0.92,
            "repeat_penalty": 1.05,
            "presence_penalty": 0.15,
            "frequency_penalty": 0.1,
            "guidelines": {
                "response_length": "detailed",
                "include_analogies": False,
                "simplify_equations": False,
                "use_latex": True,
                "equation_complexity": "intermediate",
                "reference_level": "undergraduate textbook",
                "tone": "clear and informative",
                "depth": "thorough with physical interpretation",
                "sections": [
                    "Physical explanation",
                    "Mathematical formulation (LaTeX)",
                    "Interpretation of parameters",
                    "Relation to observations",
                    "Implications for your analysis"
                ]
            }
        },
        ExpertiseLevel.ADVANCED: {
            "temperature": 0.65, 
            "max_tokens": 3000,
            "top_p": 0.95,
            "repeat_penalty": 1.05,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.05,
            "guidelines": {
                "response_length": "comprehensive",
                "include_citations": True,
                "use_latex": True,
                "equation_complexity": "advanced",
                "reference_level": "research papers",
                "tone": "technical and precise",
                "depth": "detailed with error analysis",
                "sections": [
                    "Theoretical framework",
                    "Detailed mathematical derivation (LaTeX)",
                    "Parameter space analysis",
                    "Systematic uncertainties",
                    "Comparison with literature",
                    "Research implications"
                ]
            }
        },
        ExpertiseLevel.EXPERT: {
            "temperature": 0.75,  
            "max_tokens": 3500,
            "top_p": 0.95,
            "repeat_penalty": 1.0,  
            "presence_penalty": 0.05,
            "frequency_penalty": 0.05,
            "guidelines": {
                "response_length": "exhaustive",
                "include_citations": True,
                "technical_depth": "maximum",
                "use_latex": True,
                "equation_complexity": "research",
                "reference_level": "peer review",
                "tone": "colleague-level discourse",
                "depth": "complete with alternative approaches",
                "sections": [
                    "Comprehensive theoretical background",
                    "Full mathematical treatment (LaTeX)",
                    "Alternative models and approaches",
                    "Degeneracies and systematics",
                    "Recent literature context",
                    "Future research directions",
                    "Methodological recommendations"
                ]
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
                        f"Invalid expertise level '{level_str}' for user {user_id}, using default"
                    )
                    return ExpertiseLevel.INTERMEDIATE
            
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
        config_dict = cls.EXPERTISE_CONFIGS.get(
            level, 
            cls.EXPERTISE_CONFIGS[ExpertiseLevel.INTERMEDIATE]
        )
        
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
        config = cls.EXPERTISE_CONFIGS.get(
            level, 
            cls.EXPERTISE_CONFIGS[ExpertiseLevel.INTERMEDIATE]
        )
        return config.get("guidelines", {})
    
    @classmethod
    def get_system_prompt_modifier(cls, level: ExpertiseLevel) -> str:
        """
        Get system prompt modification text for expertise level
        
        NEW: Enhanced with LaTeX formatting instructions
        
        Args:
            level: Expertise level
        
        Returns:
            Additional system prompt text
        """
        guidelines = cls.get_guidelines(level)
        
        modifiers = {
            ExpertiseLevel.BEGINNER: """
                Your audience is NEW to astrophysics. Provide COMPREHENSIVE, DETAILED explanations (500-800 words minimum):

                **Response Structure:**
                1. Start with a simple, clear explanation in everyday language
                2. Use relatable analogies (e.g., "think of a black hole like a cosmic drain")
                3. Present equations using LaTeX format ($$...$$ for display, $...$ for inline)
                4. Explain what each variable represents in simple terms
                5. Connect the math to real observations
                6. Provide concrete examples from the user's data
                7. Suggest next steps or further questions

                **LaTeX Formatting:**
                - Display equations: $$P(f) = \\frac{A}{f^b} + n$$
                - Inline variables: $f$, $A$, $b$
                - Always define variables immediately after equations
                - Keep equations simple and well-explained

                **Tone:** Encouraging, friendly, conversational. Break complex ideas into 3-5 small steps.
            """,
            
            ExpertiseLevel.INTERMEDIATE: """
                Your audience has BASIC astrophysics knowledge (undergraduate level). Provide DETAILED, THOROUGH explanations (800-1200 words minimum):

                **Response Structure:**
                1. Begin with physical context and motivation
                2. Present mathematical framework with proper LaTeX equations
                3. Explain physical meaning of each parameter
                4. Discuss how the model relates to observations
                5. Interpret specific values from user's analysis
                6. Compare with typical literature values
                7. Suggest follow-up analyses

                **LaTeX Formatting:**
                - Power law: $$\\text{PSD}(f) = \\frac{A}{f^b} + n$$
                - Bending power law: $$\\text{PSD}(f) = \\frac{A}{f\\left[1 + \\left(\\frac{f}{f_b}\\right)^{\\alpha-1}\\right]} + n$$
                - Proper subscripts/superscripts: $f_b$, $\\alpha$, $\\chi^2$
                - Units: $\\text{Hz}$, $\\text{s}^{-1}$
                - Use \\text{} for text in equations

                **Tone:** Clear, informative. Balance technical accuracy with accessibility. Reference missions/instruments when relevant.
            """,
            
            ExpertiseLevel.ADVANCED: """
                Your audience has ADVANCED knowledge (graduate level). Provide COMPREHENSIVE, RESEARCH-LEVEL explanations (1200-1800 words minimum):

                **Response Structure:**
                1. Theoretical framework from recent literature
                2. Complete mathematical treatment with detailed LaTeX derivations
                3. Parameter space and degeneracy analysis
                4. Systematic uncertainties and error analysis
                5. Comparison with published results (cite specific papers)
                6. Alternative models and their applicability
                7. Advanced techniques for follow-up

                **LaTeX Formatting:**
                - Full notation: $$P_\\nu(f) = \\frac{A}{(f/f_0)^\\alpha \\left[1 + (f/f_b)^{\\beta-\\alpha}\\right]} + C$$
                - Include derivations when relevant
                - Advanced notation: $\\langle x^2 \\rangle$, $\\chi^2_\\nu$, $\\sigma_\\text{rms}$
                - Integrals/summations: $\\int_0^\\infty$, $\\sum_{i=1}^N$
                - Vectors: $\\vec{r}$, $\\nabla \\cdot \\vec{v}$
                - Error propagation: $\\delta f = \\sqrt{\\sum_i \\left(\\frac{\\partial f}{\\partial x_i}\\right)^2 \\delta x_i^2}$$

                **Tone:** Technical, precise. Reference specific papers (Author et al. YEAR). Discuss cutting-edge research and debates.
            """,
            
            ExpertiseLevel.EXPERT: """
                Your audience is at RESEARCH level (postdoc/professor). Provide EXHAUSTIVE, PEER-REVIEW-LEVEL explanations (1800-2500 words minimum):

                **Response Structure:**
                1. Comprehensive theoretical background with historical context
                2. Full mathematical treatment (complete derivations in LaTeX)
                3. Alternative models and competing approaches
                4. Degeneracies, systematics, and selection effects
                5. Recent literature context (last 2-3 years)
                6. Future research directions and open questions
                7. Methodological recommendations for their specific case

                **LaTeX Formatting:**
                - Research-level notation with full rigor
                - Complete derivations: $$\\frac{dN}{dt} = -\\int_0^\\infty \\sigma(E) n(E) v(E) dE$$
                - Statistical frameworks: $\\mathcal{L}(\\theta|D) = \\prod_{i=1}^N P(d_i|\\theta)$
                - Tensor notation if relevant: $T^{\\mu\\nu}$, $g_{\\mu\\nu}$
                - Matrix operations: $\\mathbf{C}^{-1}$, $\\det(\\mathbf{M})$
                - Full error treatment: $\\chi^2 = \\sum_{i,j} (d_i - m_i) C_{ij}^{-1} (d_j - m_j)$

                **Tone:** Colleague-level discourse. Critique assumptions. Reference specific instruments, reduction pipelines, and analysis codes. Suggest research directions.
            """
        }
        
        return modifiers.get(level, modifiers[ExpertiseLevel.INTERMEDIATE])