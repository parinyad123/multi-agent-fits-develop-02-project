# app/services/astrosage/prompt_builder_minimal.py
"""
Minimal Prompt Builder for AstroSage (Optimized Version)
Target: < 1,000 tokens total system prompt
"""

import logging
from typing import List, Dict, Any, Optional

from app.core.constants import RoutingStrategy
from app.services.astrosage.models import ExpertiseLevel, AstroSageRequest

from app.services.astrosage.metadata_formatter import (
    MetadataFormatter,
    extract_fitted_params
)

logger = logging.getLogger(__name__)

class MinimalPromptBuilder:
    """
    Minimal prompt builder for AstroSage
    
    Key optimizations:
    - Concise system prompt (~300 tokens)
    - Essential metadata only (~200 tokens)
    - Simplified analysis context (~300 tokens)
    - Total: < 1,000 tokens
    """
    
    # ============================================
    # Ultra-concise base prompt
    # ============================================
    BASE_SYSTEM_PROMPT = """Expert astrophysicist AI. Provide accurate, well-formatted responses.

**CRITICAL RULES:**
1. LaTeX: $$display$$ or $inline$ for ALL math
2. Use first-person for YOUR analysis results ("I calculated...")
3. Educational tone for general concepts
4. Cite exact fitted values, never hallucinate

**ROUTING:** {routing_mode}
**EXPERTISE:** {expertise_level}"""

    # ============================================
    # Concise routing frames
    # ============================================
    ROUTING_FRAMES = {
        RoutingStrategy.ASTROSAGE: "Educational Q&A - explain concepts clearly",
        RoutingStrategy.ANALYSIS: "Present YOUR analysis results with authority",
        RoutingStrategy.MIXED: "YOUR results first, then context"
    }
    
    # ============================================
    # Minimal expertise modifiers
    # ============================================
    EXPERTISE_MODIFIERS = {
        ExpertiseLevel.BEGINNER: "Simple language, analogies, explain variables",
        ExpertiseLevel.INTERMEDIATE: "Technical depth, clear equations",
        ExpertiseLevel.ADVANCED: "Research-level, cite literature",
        ExpertiseLevel.EXPERT: "Peer-review quality, full rigor"
    }
    
    @classmethod
    def build_full_prompt(
        cls,
        request: AstroSageRequest,
        routing_strategy: RoutingStrategy
    ) -> List[Dict[str, str]]:
        """
        Build minimal prompt
        
        Target token counts:
        - System: 300-400 tokens
        - Analysis context: 300-400 tokens
        - Total: 600-800 tokens
        """
        
        # ============================================
        # STEP 1: Build minimal system prompt
        # ============================================
        routing_mode = cls.ROUTING_FRAMES.get(
            routing_strategy,
            cls.ROUTING_FRAMES[RoutingStrategy.ASTROSAGE]
        )
        
        expertise_mod = cls.EXPERTISE_MODIFIERS.get(
            request.expertise_level,
            cls.EXPERTISE_MODIFIERS[ExpertiseLevel.INTERMEDIATE]
        )
        
        system_prompt = cls.BASE_SYSTEM_PROMPT.format(
            routing_mode=routing_mode,
            expertise_level=expertise_mod
        )
        
        # ============================================
        # STEP 2: Add minimal analysis context
        # ============================================
        if request.analysis_results and routing_strategy != RoutingStrategy.ASTROSAGE:
            analysis_context = cls._build_minimal_analysis_context(
                request.analysis_results,
                routing_strategy
            )
            system_prompt += "\n\n" + analysis_context
        
        # ============================================
        # STEP 3: Build messages
        # ============================================
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.user_query}
        ]
        
        logger.info(
            f"Built minimal prompt: "
            f"system={len(system_prompt)} chars (~{len(system_prompt)//4} tokens), "
            f"strategy={routing_strategy.value}"
        )
        
        return messages
    
    @classmethod
    def _build_minimal_analysis_context(
        cls,
        analysis_results: Dict[str, Any],
        routing_strategy: RoutingStrategy
    ) -> str:
        """
        Build ultra-compact analysis context WITH smart metadata
        
        Target: < 400 tokens (was 300, now +100 for metadata)
        """
        
        lines = []
        
        # ============================================
        # STEP 1: Smart Metadata Context
        # ============================================
        metadata = analysis_results.get('metadata', {})
        fitted_params = extract_fitted_params(analysis_results)
        
        if metadata:
            # Use appropriate formatter based on routing
            if routing_strategy == RoutingStrategy.ANALYSIS:
                # Ultra-minimal for analysis-only
                metadata_context = MetadataFormatter.format_for_analysis_only(metadata)
                lines.append(f"**Observation:** {metadata_context}")
            else:
                # Smart context for mixed/astrosage
                metadata_context = MetadataFormatter.format_compact_context(
                    metadata,
                    routing_strategy.value,
                    fitted_params
                )
                if metadata_context:
                    lines.append(metadata_context)
        
        # ============================================
        # STEP 2: Fitted Parameters (as before)
        # ============================================
        lines.append("\n**YOUR FITTED VALUES:**")
        
        # Power Law
        if 'power_law' in analysis_results:
            pl = analysis_results['power_law'].get('fitted_parameters', {})
            if pl:
                lines.append(
                    f"PL: A={pl.get('A', 0):.3e}, "
                    f"b={pl.get('b', 0):.3f}, "
                    f"n={pl.get('n', 0):.3e}"
                )
        
        # Bending Power Law
        if 'bending_power_law' in analysis_results:
            bpl = analysis_results['bending_power_law'].get('fitted_parameters', {})
            if bpl:
                fb = bpl.get('fb', 0)
                t_break = 1.0 / fb if fb > 0 else 0
                
                lines.append(
                    f"BPL: A={bpl.get('A', 0):.3e}, "
                    f"fb={fb:.3e}Hz (t≈{t_break:.1e}s), "
                    f"α={bpl.get('sh', 0):.3f}, "
                    f"n={bpl.get('n', 0):.3e}"
                )
        
        # ============================================
        # STEP 3: Instruction
        # ============================================
        if routing_strategy == RoutingStrategy.ANALYSIS:
            lines.append("\n→ Present these as YOUR calculations")
        elif routing_strategy == RoutingStrategy.MIXED:
            lines.append("\n→ YOUR values + physical interpretation")
        
        return "\n".join(lines)


# ============================================
# Factory function for backward compatibility
# ============================================
class PromptBuilder:
    """
    Wrapper to maintain API compatibility
    Can switch between minimal and full prompts
    """
    
    USE_MINIMAL = True  # Toggle here
    
    @classmethod
    def build_full_prompt(
        cls,
        request: AstroSageRequest,
        routing_strategy: RoutingStrategy
    ) -> List[Dict[str, str]]:
        """
        Build prompt (delegates to minimal or full builder)
        """
        
        if cls.USE_MINIMAL:
            return MinimalPromptBuilder.build_full_prompt(request, routing_strategy)
        else:
            # Import original builder
            from app.services.astrosage.prompt_builder_original import (
                PromptBuilder as FullPromptBuilder
            )
            return FullPromptBuilder.build_full_prompt(request, routing_strategy)