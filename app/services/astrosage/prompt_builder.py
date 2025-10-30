"""
multi-agent-fits-dev-02/app/services/astrosage/prompt_builder.py

IMPROVED VERSION: Enhanced to force LLM to use actual numerical results
"""

import logging
from typing import List, Dict, Any, Optional

from app.services.astrosage.models import (
    ExpertiseLevel,
    ConversationPair,
    AstroSageRequest
)
from app.services.astrosage.expertise_adapter import ExpertiseAdapter

logger = logging.getLogger(__name__)

class PromptBuilder:
    """
    Construct optimized prompts for AstroSage LLM
    
    KEY IMPROVEMENTS:
    - Forces LLM to cite actual numerical values
    - Requires model comparison when both are available
    - Emphasizes physical interpretation of fitted parameters
    """
    
    # Base system prompt (always included)
    BASE_SYSTEM_PROMPT = """You are AstroSage, an expert astrophysicist AI assistant specializing in stellar physics, X-ray astronomy, time series analysis, power spectral density analysis, and accretion disk physics. You provide accurate, detailed, comprehensive scientific explanations while maintaining conversation continuity. Always reference previous discussion points when relevant.

        **CRITICAL RESPONSE REQUIREMENTS:**
        1. **Length**: Provide thorough, comprehensive explanations. Do not give brief answers.

        2. **LaTeX Equations**: Always use LaTeX formatting for mathematical expressions
        - Display equations: $$equation$$
        - Inline math: $variable$

        3. **Structure**: Organize responses with clear sections and logical flow

        4. **Physical Interpretation**: Always explain the physical meaning behind mathematical results

        5. **MANDATORY: CITE ACTUAL NUMERICAL VALUES**: 
        - You MUST start by explicitly stating the fitted parameter values from the user's analysis
        - Example: "From your analysis, the power law fit gives A = 2.67×10³, b = 0.802, n = 1.23×10⁻²"
        - DO NOT give generic explanations without referring to specific numbers

        6. **MANDATORY: MODEL COMPARISON**:
        - When both power law AND bending power law results are available, you MUST:
            * Compare the fitted parameters side by side
            * Explain which model fits better and why
            * Discuss what the break frequency tells us (if bending power law is used)
            * Provide quantitative comparison (e.g., reduced chi-squared values if available)

        7. **Completeness**: Cover theory, mathematics, interpretation, and practical implications

        **FORBIDDEN**: Never give purely theoretical explanations without connecting to the user's actual data values."""
  
    @classmethod
    def build_system_prompt(cls, expertise_level: ExpertiseLevel) -> str:
        """
        Build system prompt based on expertise level
        
        Args:
            expertise_level: User's expertise level
        
        Returns:
            Complete system prompt with LaTeX instructions
        """
        # Start with base prompt
        prompt = cls.BASE_SYSTEM_PROMPT
        
        # Add expertise-specific modifier
        modifier = ExpertiseAdapter.get_system_prompt_modifier(expertise_level)
        prompt += "\n\n" + modifier
        
        # Add LaTeX examples section
        prompt += "\n\n" + cls._get_latex_examples()
        
        return prompt
    
    @classmethod
    def _get_latex_examples(cls) -> str:
        """
        Provide LaTeX formatting examples
        
        Returns:
            Examples of proper LaTeX usage
        """
        return """
            **LaTeX FORMATTING EXAMPLES:**

            Display Equations (use $$...$$):
            - Power Law: $$\\text{PSD}(f) = \\frac{A}{f^b} + n$$
            - Bending Power Law: $$\\text{PSD}(f) = \\frac{A}{f\\left[1 + \\left(\\frac{f}{f_b}\\right)^{\\alpha-1}\\right]} + n$$
            - Chi-squared: $$\\chi^2 = \\sum_{i=1}^N \\frac{(O_i - E_i)^2}{\\sigma_i^2}$$
            - Integrals: $$F_\\text{total} = \\int_{f_\\text{min}}^{f_\\text{max}} \\text{PSD}(f) \\, df$$

            Inline Math (use $...$):
            - Parameters: $A = 2.67 \\times 10^3$, $b = 0.802$, $f_b = 4.06 \\times 10^{-5}$ Hz
            - Ranges: frequency range $f \\in [10^{-5}, 10^{-2}]$ Hz
            - Comparisons: if $b > 1$, then $b < 2$ implies...

            Subscripts and Superscripts:
            - $f_b$ (break frequency), $\\sigma_\\text{rms}$ (RMS variability)
            - $\\chi^2_\\nu$ (reduced chi-squared), $M_\\odot$ (solar mass)

            Greek Letters:
            - $\\alpha$, $\\beta$, $\\gamma$, $\\sigma$, $\\chi$, $\\nu$, $\\omega$

            **IMPORTANT:** Always explain what each variable means immediately after introducing it!
            """

    @classmethod
    def build_conversation_context(
        cls, 
        conversations: Optional[List[ConversationPair]]
    ) -> str:
        """
        Build conversation context section
        
        Args:
            conversations: List of past conversation pairs
        
        Returns:
            Formatted conversation context
        """
        if not conversations:
            return ""
        
        lines = ["\n\n=== PREVIOUS CONVERSATION HISTORY ===\n"]
        lines.append("(Last 10 exchanges - maintain continuity with this discussion)\n")
        
        for i, pair in enumerate(conversations, 1):
            # Format time
            time_str = cls._format_timestamp(pair.timestamp)
            
            # User message
            lines.append(f"\n**Exchange {i}** ({time_str}):")
            lines.append(f"USER: {pair.user_message}")
            
            # Assistant message (truncate if very long)
            assistant_msg = pair.assistant_message
            if len(assistant_msg) > 300:
                assistant_msg = assistant_msg[:300] + "... [response continues]"
            lines.append(f"ASTROSAGE: {assistant_msg}\n")
        
        lines.append("\n**INSTRUCTION:** Reference these past exchanges when relevant to provide continuity.")
        
        return "\n".join(lines)
    
    @classmethod
    def build_analysis_context(
        cls,
        analysis_results: Optional[Dict[str, Any]],
        expertise_level: ExpertiseLevel = ExpertiseLevel.INTERMEDIATE
    ) -> str:
        """
        Build analysis results context section with STRONG emphasis on using values
        
        Args:
            analysis_results: Results from FITS analysis
            expertise_level: User's expertise level for appropriate detail
        
        Returns:
            Formatted analysis context
        """
        if not analysis_results:
            return ""
        
        lines = ["\n\n=== USER'S FITS FILE ANALYSIS RESULTS ===\n"]
        lines.append("**MANDATORY INSTRUCTION:** Your response MUST interpret and explain these specific numerical values in detail!")
        lines.append("**YOU MUST CITE THESE EXACT NUMBERS IN YOUR EXPLANATION!**\n")
        
        # File information
        if 'metadata' in analysis_results:
            lines.append(cls._format_metadata(analysis_results['metadata']))
        
        # Statistics
        if 'statistics' in analysis_results:
            lines.append(cls._format_statistics(
                analysis_results['statistics'], 
                expertise_level
            ))
        
        # PSD
        if 'psd' in analysis_results:
            lines.append(cls._format_psd(
                analysis_results['psd'],
                expertise_level
            ))
        
        # Check if BOTH models are present
        has_power_law = 'power_law' in analysis_results
        has_bending = 'bending_power_law' in analysis_results
        
        # Power Law
        if has_power_law:
            lines.append(cls._format_power_law(
                analysis_results['power_law'],
                expertise_level
            ))
        
        # Bending Power Law
        if has_bending:
            lines.append(cls._format_bending_power_law(
                analysis_results['bending_power_law'],
                expertise_level
            ))
        
        # Add comparison instruction if both models exist
        if has_power_law and has_bending:
            lines.append("\n" + "="*70)
            lines.append("**CRITICAL INSTRUCTION: MODEL COMPARISON REQUIRED!**")
            lines.append("="*70)
            lines.append("""
                Since BOTH power law and bending power law fits are available, you MUST:

                1. **State both sets of fitted parameters explicitly**
                2. **Compare the models directly:**
                - Which model provides a better fit? (Look at chi-squared if available)
                - What does the break frequency $f_b$ tell us about the system?
                - How do the power law indices differ between models?
                3. **Physical interpretation:**
                - What does the bending power law's break frequency imply about characteristic timescales?
                - Does the simple power law adequately capture the variability, or is the bending model necessary?
                4. **Recommendations:**
                - Which model should be used for further analysis?
                - What follow-up observations or analyses would help?
                """)
        
        lines.append("\n**REMINDER:** Explain what these numerical values mean physically for THIS SPECIFIC SOURCE!")
        
        return "\n".join(lines)

    @classmethod
    def build_full_prompt(cls, request: AstroSageRequest) -> List[Dict[str, str]]:
        """
        Build complete prompt for LLM API
        
        Args:
            request: AstroSageRequest object
        
        Returns:
            List of message dictionaries for API
        """
        # Build system prompt
        system_prompt = cls.build_system_prompt(request.expertise_level)
        
        # Add conversation context
        if request.conversation_history:
            conversation_context = cls.build_conversation_context(
                request.conversation_history
            )
            system_prompt += conversation_context
        
        # Add analysis context
        if request.analysis_results:
            analysis_context = cls.build_analysis_context(
                request.analysis_results,
                request.expertise_level
            )
            system_prompt += analysis_context
        
        # Add final instruction
        system_prompt += cls._build_final_instruction(request.expertise_level)
        
        # Build messages list
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": request.user_query
            }
        ]
        
        logger.info(
            f"Built prompt: system={len(system_prompt)} chars, "
            f"expertise={request.expertise_level.value}"
        )
        
        return messages
    
    @classmethod
    def _build_final_instruction(cls, expertise_level: ExpertiseLevel) -> str:
        """
        Build final instruction based on expertise level
        
        Args:
            expertise_level: User's expertise level
        
        Returns:
            Final instruction text
        """
        base_instruction = """
            \n\n**FINAL MANDATORY CHECKLIST BEFORE RESPONDING:**

            ☑ Have I cited the ACTUAL fitted parameter values from the analysis?
            ☑ Have I compared both models (if both power law and bending power law are available)?
            ☑ Have I explained the PHYSICAL meaning of each parameter?
            ☑ Have I used proper LaTeX formatting for all equations?
            ☑ Is my response sufficiently detailed (meeting minimum word count)?
            ☑ Have I provided specific recommendations based on THEIR data?

            **If you cannot check ALL boxes above, your response is INCOMPLETE!**
            """
        
        instructions = {
            ExpertiseLevel.BEGINNER: base_instruction + """
                **RESPONSE GUIDELINES FOR BEGINNER LEVEL:**
                - Write at least 500-800 words
                - Use 3-5 clear sections with headers
                - Include LaTeX equations (properly formatted)
                - Explain all technical terms
                - Use analogies to make concepts accessible
                - End with practical next steps
            """,
            ExpertiseLevel.INTERMEDIATE: base_instruction + """
                **RESPONSE GUIDELINES FOR INTERMEDIATE LEVEL:**
                - Write at least 800-1200 words
                - Provide thorough physical and mathematical explanation
                - Use LaTeX for all equations
                - Compare with typical values from literature
                - Discuss observational implications
                - Suggest follow-up analyses
            """,
            ExpertiseLevel.ADVANCED: base_instruction + """
                **RESPONSE GUIDELINES FOR ADVANCED LEVEL:**
                - Write at least 1200-1800 words
                - Provide research-level analysis
                - Include mathematical derivations (LaTeX)
                - Discuss systematic uncertainties
                - Reference specific papers and missions
                - Evaluate alternative models
                - Suggest advanced techniques
            """,
            ExpertiseLevel.EXPERT: base_instruction + """
                **RESPONSE GUIDELINES FOR EXPERT LEVEL:**
                - Write at least 1800-2500 words
                - Provide exhaustive peer-review-level analysis
                - Include complete derivations (LaTeX)
                - Discuss degeneracies and selection effects
                - Comprehensive literature context
                - Multiple approaches and methodologies
                - Research directions and open questions
            """
        }
        
        return instructions.get(expertise_level, instructions[ExpertiseLevel.INTERMEDIATE])


    # ==========================================
    # Helper Methods for Formatting
    # ==========================================
    
    @staticmethod
    def _format_metadata(metadata: Dict[str, Any]) -> str:
        """Format file metadata"""
        lines = ["**File Information:**"]
        
        if 'original_filename' in metadata:
            lines.append(f"- Filename: `{metadata['original_filename']}`")
        if 'file_size' in metadata:
            size_mb = metadata['file_size'] / (1024 * 1024)
            lines.append(f"- File Size: {size_mb:.2f} MB")
        if 'uploaded_at' in metadata:
            lines.append(f"- Uploaded: {metadata['uploaded_at']}")
        if 'n_data_points' in metadata:
            lines.append(f"- Total Data Points: {metadata['n_data_points']:,}")
        
        return "\n".join(lines) + "\n"
    
    @staticmethod
    def _format_statistics(
        stats: Dict[str, Any],
        expertise_level: ExpertiseLevel
    ) -> str:
        """Format statistics results with appropriate detail"""
        lines = ["**Statistical Summary:**"]
        
        if 'statistics' in stats:
            stats = stats['statistics']
        
        # Basic stats (all levels)
        if 'count' in stats:
            lines.append(f"- Data Points: $N = {stats['count']:,}$")
        if 'mean' in stats:
            lines.append(f"- Mean Rate: $\\langle R \\rangle = {stats['mean']:.6e}$ counts/s")
        if 'median' in stats:
            lines.append(f"- Median Rate: $R_\\text{{median}} = {stats['median']:.6e}$ counts/s")
        if 'std' in stats:
            lines.append(f"- Standard Deviation: $\\sigma = {stats['std']:.6e}$ counts/s")
        
        # Additional stats for advanced users
        if expertise_level in [ExpertiseLevel.ADVANCED, ExpertiseLevel.EXPERT]:
            if 'min' in stats and 'max' in stats:
                lines.append(f"- Range: $[{stats['min']:.6e}, {stats['max']:.6e}]$ counts/s")
            if 'mean' in stats and 'std' in stats:
                cv = stats['std'] / stats['mean']
                lines.append(f"- Coefficient of Variation: $CV = \\sigma/\\mu = {cv:.3f}$")
        
        return "\n".join(lines) + "\n"
    
    @staticmethod
    def _format_psd(
        psd: Dict[str, Any],
        expertise_level: ExpertiseLevel
    ) -> str:
        """Format PSD results with appropriate detail"""
        lines = ["**Power Spectral Density Analysis:**"]
        
        if 'freq_range' in psd:
            if isinstance(psd['freq_range'], dict) and 'actual' in psd['freq_range']:
                freq_range = psd['freq_range']['actual']
            else:
                freq_range = psd['freq_range']
            
            f_min, f_max = freq_range[0], freq_range[1]
            lines.append(
                f"- Frequency Range: $f \\in [{f_min:.2e}, {f_max:.2e}]$ Hz"
            )
            
            # Add period range for context (advanced users)
            if expertise_level in [ExpertiseLevel.ADVANCED, ExpertiseLevel.EXPERT]:
                if f_min > 0:
                    p_max = 1.0 / f_min
                    p_min = 1.0 / f_max
                    lines.append(
                        f"- Corresponding Period Range: $P \\in [{p_min:.2e}, {p_max:.2e}]$ s"
                    )
        
        if 'n_points' in psd:
            lines.append(f"- Number of Frequency Bins: $N_\\text{{bins}} = {psd['n_points']:,}$")
        
        if 'n_bins' in psd and isinstance(psd['n_bins'], dict):
            lines.append(
                f"- Requested/Actual Bins: {psd['n_bins'].get('requested', 'N/A')} / "
                f"{psd['n_bins'].get('actual', 'N/A')}"
            )
        
        return "\n".join(lines) + "\n"
    
    @staticmethod
    def _format_power_law(
        power_law: Dict[str, Any],
        expertise_level: ExpertiseLevel
    ) -> str:
        """Format power law fit results with STRONG emphasis on using these values"""
        lines = ["\n" + "="*70]
        lines.append("**POWER LAW FIT RESULTS** (YOU MUST CITE THESE VALUES!):")
        lines.append("="*70)
        lines.append("Model: $$\\text{PSD}(f) = \\frac{A}{f^b} + n$$\n")
        
        if 'parameters' in power_law:
            params = power_law['parameters']
        elif 'fitted_parameters' in power_law:
            params = power_law['fitted_parameters']
        else:
            params = power_law
        
        if 'A' in params:
            lines.append(f"- **Amplitude**: $A = {params['A']:.6e}$")
        if 'b' in params:
            lines.append(f"- **Power Law Index**: $b = {params['b']:.3f}$")
        if 'n' in params:
            lines.append(f"- **Noise Level**: $n = {params['n']:.6e}$")
        
        # Add interpretation hints for advanced users
        if expertise_level in [ExpertiseLevel.ADVANCED, ExpertiseLevel.EXPERT]:
            if 'b' in params:
                b_val = params['b']
                lines.append(f"\n*Hint: $b = {b_val:.3f}$ suggests ", )
                if b_val < 1:
                    lines.append("red noise / flicker noise regime*")
                elif b_val < 2:
                    lines.append("typical accreting source variability*")
                else:
                    lines.append("steep spectrum / white noise dominated*")
        
        # Add fit quality if available
        if 'fit_quality' in power_law:
            quality = power_law['fit_quality']
            if 'chi_squared' in quality:
                lines.append(f"- **Goodness of Fit**: $\\chi^2_\\nu = {quality['chi_squared']:.3f}$")
        
        lines.append("\n**INSTRUCTION:** Interpret what these specific values mean for this source!")
        
        return "\n".join(lines) + "\n"
    
    @staticmethod
    def _format_bending_power_law(
        bending: Dict[str, Any],
        expertise_level: ExpertiseLevel
    ) -> str:
        """Format bending power law fit results with STRONG emphasis"""
        lines = ["\n" + "="*70]
        lines.append("**BENDING POWER LAW FIT RESULTS** (YOU MUST CITE THESE VALUES!):")
        lines.append("="*70)
        lines.append(
            "Model: $$\\text{PSD}(f) = \\frac{A}{f\\left[1 + \\left(\\frac{f}{f_b}\\right)^{\\alpha-1}\\right]} + n$$\n"
        )
        
        if 'parameters' in bending:
            params = bending['parameters']
        elif 'fitted_parameters' in bending:
            params = bending['fitted_parameters']
        else:
            params = bending
        
        if 'A' in params:
            lines.append(f"- **Amplitude**: $A = {params['A']:.6e}$")
        if 'fb' in params:
            lines.append(f"- **Break Frequency**: $f_b = {params['fb']:.6e}$ Hz")
            # Add timescale for context
            if params['fb'] > 0:
                t_break = 1.0 / params['fb']
                lines.append(f"  → **Characteristic Timescale**: $t_b \\approx {t_break:.2e}$ s")
        if 'sh' in params:
            lines.append(f"- **Shape Parameter**: $\\alpha = {params['sh']:.3f}$")
        if 'n' in params:
            lines.append(f"- **Noise Level**: $n = {params['n']:.6e}$")
        
        # Add physical interpretation for advanced users
        if expertise_level in [ExpertiseLevel.ADVANCED, ExpertiseLevel.EXPERT]:
            if 'fb' in params and params['fb'] > 0:
                lines.append(
                    f"\n*Hint: $f_b = {params['fb']:.2e}$ Hz may correspond to "
                    f"characteristic frequencies from inner disk radius, "
                    f"viscous timescales, or orbital periods*"
                )
        
        # Add fit quality if available
        if 'fit_quality' in bending:
            quality = bending['fit_quality']
            if 'chi_squared' in quality:
                lines.append(f"- **Goodness of Fit**: $\\chi^2_\\nu = {quality['chi_squared']:.3f}$")
        
        lines.append("\n**INSTRUCTION:** Explain what the break frequency tells us about this system!")
        
        return "\n".join(lines) + "\n"
    
    @staticmethod
    def _format_timestamp(timestamp) -> str:
        """Format timestamp for display"""
        from datetime import datetime, timezone
        
        now = datetime.now()
        
        # Handle timezone-aware timestamps
        if timestamp.tzinfo is not None:
            now = now.replace(tzinfo=timezone.utc)
        
        delta = now - timestamp
        seconds = delta.total_seconds()
        
        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes}m ago"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours}h ago"
        else:
            days = int(seconds / 86400)
            return f"{days}d ago"