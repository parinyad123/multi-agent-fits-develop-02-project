"""
app/agents/rewrite/prompt_builder.py - FIXED VERSION
"""

from typing import List, Dict, Any, Optional
import json
import logging

from app.agents.rewrite.models import RewriteRequest

logger = logging.getLogger(__name__)


class RewritePromptBuilder:
    """Build GPT prompts for response rewriting"""
    
    # FIX: Escape ALL curly braces that are NOT format placeholders
    SYSTEM_PROMPT_BASE = """You are an expert scientific response formatter for an astrophysics analysis system.

        Your role is to:
        1. **Format and validate LaTeX**: Ensure all mathematical expressions use proper LaTeX syntax
        - Display math: $$equation$$
        - Inline math: $variable$
        - Proper notation: \\times, \\pm, \\frac{{}}{{}}, \\chi^2, etc.

        2. **Structure responses clearly**: Use markdown headers, sections, and formatting
        - Start with executive summary
        - Organize by analysis type
        - Add visual cues (bold, bullet points)
        - Include plot links at the end

        3. **Integrate multi-agent results**: Combine Analysis + AstroSage seamlessly
        - Cross-reference numerical results
        - Add physical interpretation
        - Maintain scientific accuracy

        4. **Adapt to expertise level**: {expertise_level}
        {expertise_guidelines}

        CRITICAL RULES:
        - NEVER change numerical values from the analysis results
        - ALWAYS use proper LaTeX for ALL mathematical expressions
        - NEVER invent information not present in the results
        - Keep scientific accuracy paramount
        - Cite which agent provided which information when relevant
        """
    
    # Expertise guidelines (no changes needed here)
    EXPERTISE_GUIDELINES = {
                "beginner": """
        **For Beginner Level:**
        - Use simple, clear language
        - Explain technical terms in everyday words
        - Use analogies when helpful (e.g., "like water flowing down a drain")
        - Keep sentences short and direct
        - Target length: 800-1200 words
        - Avoid jargon unless explained
        """,
                "intermediate": """
        **For Intermediate Level:**
        - Use standard scientific terminology
        - Balance accessibility with technical accuracy
        - Provide context for key findings
        - Reference typical values from literature when relevant
        - Target length: 1200-1800 words
        - Assume basic astrophysics knowledge
        """,
                "advanced": """
        **For Advanced Level:**
        - Use technical language freely
        - Include detailed methodology discussion
        - Reference literature when relevant
        - Discuss systematic uncertainties
        - Compare with published results
        - Target length: 1800-2500 words
        - Assume graduate-level knowledge
        """,
                "expert": """
        **For Expert Level:**
        - Colleague-level discourse
        - Full mathematical rigor
        - Detailed error analysis
        - Compare alternative approaches
        - Discuss degeneracies and selection effects
        - Target length: 2500-3500 words
        - Assume postdoc/professor level knowledge
        """
    }
    
    def build_prompt(self, request: RewriteRequest) -> List[Dict[str, str]]:
        """Build complete prompt for GPT"""
        
        # Build system prompt
        system_prompt = self._build_system_prompt(request.expertise_level)
        
        # Build user prompt
        user_prompt = self._build_user_prompt(request)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        logger.debug(
            f"Built prompt: system={len(system_prompt)} chars, "
            f"user={len(user_prompt)} chars"
        )
        
        return messages
    
    def _build_system_prompt(self, expertise_level: str) -> str:
        """Build system prompt with expertise guidelines"""
        
        guidelines = self.EXPERTISE_GUIDELINES.get(
            expertise_level,
            self.EXPERTISE_GUIDELINES["intermediate"]
        )
        
        # Now this will work because we escaped {{}} in the template
        return self.SYSTEM_PROMPT_BASE.format(
            expertise_level=expertise_level.upper(),
            expertise_guidelines=guidelines
        )
    
    def _build_user_prompt(self, request: RewriteRequest) -> str:
        """Build user prompt with all context"""
        
        parts = []
        
        # Part 1: User query
        parts.append("# USER QUERY")
        parts.append(f'"{request.user_query}"')
        parts.append("")
        
        # Part 2: Workflow type
        parts.append("# WORKFLOW TYPE")
        parts.append(f"Routing Strategy: **{request.routing_strategy}**")
        parts.append("")
        
        # Part 3: Analysis results (if available)
        analysis_step = self._find_step(request.completed_steps, "analysis")
        if analysis_step:
            parts.append(self._format_analysis_section(analysis_step))
            parts.append("")
        
        # Part 4: AstroSage response (if available)
        astrosage_step = self._find_step(request.completed_steps, "astrosage")
        if astrosage_step and astrosage_step.get('success'):
            parts.append(self._format_astrosage_section(astrosage_step))
            parts.append("")
        
        # Part 5: Task instructions
        parts.append(self._build_task_instructions(request.routing_strategy))
        
        return "\n".join(parts)
    
    def _find_step(self, steps: List[Dict], step_name: str) -> Optional[Dict]:
        """Find a specific step in completed_steps"""
        for step in steps:
            if step.get('step') == step_name:
                return step
        return None
    
    def _format_analysis_section(self, step: Dict) -> str:
        """Format analysis results section"""
        
        result = step.get('analysis_result', {})
        
        parts = []
        parts.append("# ANALYSIS RESULTS")
        parts.append("")
        parts.append(f"**Status**: {result.get('status', 'unknown')}")
        parts.append(f"**Execution Time**: {result.get('execution_time', 0):.2f}s")
        parts.append("")
        
        # Completed analyses
        completed = result.get('completed_analyses', [])
        if completed:
            parts.append(f"**Completed**: {', '.join(completed)}")
        
        # Failed analyses
        failed = result.get('failed_analyses', [])
        if failed:
            parts.append(f"**Failed**: {', '.join(failed)}")
        
        # Skipped analyses
        skipped = result.get('skipped_analyses', [])
        if skipped:
            parts.append(f"**Skipped**: {', '.join(skipped)}")
        
        parts.append("")
        
        # Detailed results
        results = result.get('results', {})
        if results:
            parts.append("## Detailed Results")
            parts.append("")
            parts.append("```json")
            parts.append(json.dumps(results, indent=2))
            parts.append("```")
            parts.append("")
        
        # Errors
        errors = result.get('errors', {})
        if errors:
            parts.append("## Errors")
            for analysis_type, error in errors.items():
                parts.append(f"- **{analysis_type}**: {error}")
            parts.append("")
        
        # Plots
        plots = result.get('plots', [])
        if plots:
            parts.append("## Generated Plots")
            for plot in plots:
                parts.append(f"- **{plot.get('plot_type')}**: `{plot.get('plot_url')}`")
            parts.append("")
        
        return "\n".join(parts)
    
    def _format_astrosage_section(self, step: Dict) -> str:
        """Format AstroSage response section"""
        
        parts = []
        parts.append("# ASTROSAGE INTERPRETATION")
        parts.append("")
        
        # Response content
        content = step.get('response', '')
        parts.append("**AstroSage's Response:**")
        parts.append("")
        parts.append(content)
        parts.append("")
        
        # Metadata
        parts.append("**Metadata:**")
        parts.append(f"- Model: {step.get('model_used', 'unknown')}")
        parts.append(f"- Tokens: {step.get('tokens_used', 0)}")
        parts.append(f"- Response Time: {step.get('response_time', 0):.2f}s")
        
        if step.get('error'):
            parts.append(f"- Error: {step['error']}")
        
        return "\n".join(parts)
    
    def _build_task_instructions(self, routing_strategy: str) -> str:
        """Build task-specific instructions based on routing strategy"""
        
        instructions = {
            "analysis": """# YOUR TASK

            Create a well-formatted response that:
            1. Presents the analysis results clearly with proper sections
            2. Validates and fixes all LaTeX expressions (use $...$ for inline, $$...$$ for display)
            3. Adds context to numerical values (what do they mean?)
            4. Organizes results by analysis type
            5. Includes plot links at the end with descriptive text

            **Structure:**
            - Executive Summary (2-3 sentences highlighting key findings)
            - Analysis Results (one section per analysis type)
            - Visualizations (formatted plot links)

            **Remember**: Use the ACTUAL numerical values from the analysis results above.
            """,
                        
                        "astrosage": """# YOUR TASK

            Create a well-formatted response that:
            1. Presents the AstroSage interpretation clearly
            2. Validates and fixes all LaTeX expressions
            3. Maintains conversational and engaging tone
            4. Ensures scientific accuracy

            **Structure:**
            - Direct answer to user's question
            - Supporting explanation with context
            - Additional insights (if relevant)

            **Remember**: This is primarily a Q&A response, keep it focused and clear.
            """,
                        
                        "mixed": """# YOUR TASK

            Create a comprehensive response that INTEGRATES both Analysis and AstroSage results:

            1. **Start with Executive Summary** (2-3 sentences)
            - Highlight the most important findings
            - Mention key parameter values

            2. **Analysis Results Section**
            - Present numerical results clearly
            - Use proper LaTeX for ALL equations and parameters
            - Organize by analysis type (Statistics, PSD, Model Fitting)
            - Include actual fitted parameter values

            3. **Physical Interpretation Section** (from AstroSage)
            - Explain what the numbers mean physically
            - Connect to accretion disk physics
            - Compare models if multiple fits were done
            - Discuss implications

            4. **Model Comparison** (if applicable)
            - Compare power law vs bending power law
            - Which model fits better and why?
            - What does the break frequency tell us?

            5. **Conclusions & Recommendations**
            - Summarize key insights
            - Suggest next steps or follow-up analyses

            6. **Visualizations**
            - Format plot links with descriptions

            **CRITICAL**: When citing parameters (e.g., A, b, fb), use the ACTUAL values from the Analysis Results section above.
            DO NOT use generic placeholders or made-up numbers.

            **Example of correct citation:**
            "My power law fit yielded $A = 2.16 \\times 10^3$, $b = 0.809$, $n = 4.80 \\times 10^4$."

            NOT: "I calculated A, b, and n" (too vague, missing actual values)
            """
        }
        
        return instructions.get(
            routing_strategy,
            instructions["analysis"]  # default
        )