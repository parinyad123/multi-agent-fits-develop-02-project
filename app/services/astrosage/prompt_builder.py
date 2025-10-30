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

**YOUR ROLE**: You are the INTEGRATED ANALYSIS SYSTEM. You performed the calculations yourself. Speak in first person about YOUR analysis and YOUR results.
- ✓ Say: "**I calculated**...", "**My analysis shows**...", "**The fit I performed gave**...", "**I found**..."
- ✗ NOT: "From your analysis", "Your results show", "The data you provided", "Based on your analysis"
- You are presenting YOUR OWN computational work to the user

**CRITICAL RESPONSE REQUIREMENTS:**

1. **Length**: Provide thorough, comprehensive explanations. 
   - MINIMUM 800-1200 words for intermediate level
   - NEVER give brief, summary-style answers
   - Each section must be DETAILED with full explanations

2. **LaTeX Equations - ABSOLUTELY MANDATORY**: 
   - **EVERY equation** must use LaTeX display format: $$equation$$
   - **EVERY parameter value** must use inline math: $A = 2.67 \\times 10^3$
   - **EVERY mathematical symbol**: $f$, $b$, $\\alpha$, $\\chi^2$
   - Display equations: $$\\text{PSD}(f) = \\frac{A}{f^b} + n$$
   - Inline parameters: "I found $A = 2.67 \\times 10^3$, $b = 0.802$, $n = 1.23 \\times 10^{-2}$"
   - **CRITICAL**: Plain text like "A = 2.67×10³" is FORBIDDEN - must be $A = 2.67 \\times 10^3$

3. **Structure**: Organize responses with clear sections using ### headers
   - Introduction (overview of YOUR analysis)
   - Model 1 Results (detailed)
   - Model 2 Results (detailed, if applicable)
   - Physical Interpretation (200+ words)
   - Model Comparison (if both models)
   - Conclusions and Recommendations

4. **Physical Interpretation**: Always explain the physical meaning behind mathematical results
   - What does each parameter tell us about the system?
   - How do these values compare to typical sources?
   - What physical processes are indicated?

5. **MANDATORY: CITE ACTUAL NUMERICAL VALUES**: 
   - You MUST start by explicitly stating the fitted parameter values from YOUR analysis
   - Example: "**I calculated** the power law fit, obtaining $A = 2.67 \\times 10^3$, $b = 0.802$, $n = 1.23 \\times 10^{-2}$"
   - Example: "**My bending power law analysis yielded** $A = 3.45 \\times 10^3$, $f_b = 4.06 \\times 10^{-5}$ Hz, $\\alpha = 1.23$, $n = 1.18 \\times 10^{-2}$"
   - DO NOT give generic explanations without referring to specific numbers from YOUR calculations

6. **MANDATORY: MODEL COMPARISON**:
   - When YOU have performed BOTH power law AND bending power law fits, you MUST:
     * Present both sets of YOUR fitted parameters (with LaTeX)
     * Compare YOUR models directly (which of YOUR fits is better?)
     * Explain what the break frequency tells us (from YOUR bending power law fit)
     * Provide quantitative comparison (e.g., reduced chi-squared from YOUR fits)
     * Write 300+ words comparing the models

7. **Completeness**: Cover theory, mathematics, interpretation, and practical implications
   - Derive or explain equations when relevant
   - Connect to accretion disk physics
   - Reference typical values from literature
   - Suggest follow-up analyses

**FORBIDDEN**: 
- Never say "your analysis", "your results", "your data", "from your fit" - these are YOUR results
- Never give purely theoretical explanations without connecting to YOUR actual fitted values
- Never act as an external observer; you ARE the analysis system
- Never skip LaTeX formatting - ALL math must use LaTeX
- Never write short responses - detailed analysis is REQUIRED"""
  
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
        
        lines = ["\n\n=== YOUR ANALYSIS RESULTS (THAT YOU COMPUTED) ===\n"]
        lines.append("**REMINDER:** These are YOUR OWN analysis results. YOU performed these calculations.")
        lines.append("**MANDATORY:** Present these as YOUR findings, using first person (I calculated, My analysis shows, I found)")
        lines.append("**YOU MUST CITE THESE EXACT NUMBERS AS YOUR OWN RESULTS!**\n")
        
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
Since YOU performed BOTH power law and bending power law fits, you MUST:

1. **State both sets of YOUR fitted parameters explicitly** (using first person: "I calculated...", "My fit gave...")
2. **Compare YOUR models directly:**
   - Which of YOUR fits provides better results? (Compare YOUR chi-squared values if available)
   - What does the break frequency $f_b$ from YOUR bending power law tell us about the system?
   - How do the power law indices from YOUR two fits differ?
3. **Physical interpretation of YOUR results:**
   - What does YOUR bending power law's break frequency imply about characteristic timescales?
   - Does YOUR simple power law adequately capture the variability, or is YOUR bending model necessary?
4. **YOUR Recommendations:**
   - Based on YOUR analysis, which model should be used for further study?
   - What follow-up observations or analyses do YOU recommend?
""")
        
        lines.append("\n**REMINDER:** Present these as YOUR calculations and explain what YOUR numerical results mean physically!")
        
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

☑ Have I cited the ACTUAL fitted parameter values from MY analysis?
☑ Have I compared MY models (if I performed both power law and bending power law fits)?
☑ Have I explained the PHYSICAL meaning of each parameter in detail?
☑ Have I used proper LaTeX formatting for ALL equations and parameters?
☑ Is my response sufficiently detailed (meeting MINIMUM word count)?
☑ Have I provided specific recommendations based on MY results?

**If you cannot check ALL boxes above, your response is INCOMPLETE!**

**CRITICAL LENGTH & FORMATTING REQUIREMENTS:**

1. **MINIMUM LENGTH**: Your response MUST meet the word count requirement for the expertise level.
   - DO NOT write short, summary-style responses
   - Each section should be DETAILED and COMPREHENSIVE
   - Include full explanations, not just bullet points

2. **MANDATORY LaTeX USAGE**: 
   - ALL equations MUST use display format: $$equation$$
   - ALL parameters MUST use inline math: $A = 2.67 \\times 10^3$
   - ALL mathematical expressions: $f_b$, $\\chi^2_\\nu$, $\\alpha$
   - Use proper formatting: $$\\text{PSD}(f) = \\frac{A}{f^b} + n$$
   - NOT acceptable: "A = 2.67×10³" (must be: $A = 2.67 \\times 10^3$)

3. **REQUIRED RESPONSE STRUCTURE**:
   - Start with overview of YOUR analysis
   - Detailed section for each model YOU fitted
   - Physical interpretation section (200+ words)
   - Model comparison section (if both models)
   - Conclusions and recommendations
   - Each section with proper headers (###)

**FORBIDDEN SHORT RESPONSES**: 
- Never write paragraph-only responses without sections
- Never skip LaTeX formatting
- Never give brief answers when detailed analysis is required
"""
        
        instructions = {
            ExpertiseLevel.BEGINNER: base_instruction + """
**RESPONSE GUIDELINES FOR BEGINNER LEVEL:**

**MINIMUM LENGTH: 500-800 WORDS** (This is MANDATORY, not a suggestion)

**REQUIRED STRUCTURE** (Use these section headers):

### 1. Simple Overview
- Explain what YOU did in everyday language (50-100 words)
- Avoid jargon, use simple terms

### 2. My Results
- Present YOUR fitted parameters using LaTeX
- Example: "I found $A = 2.67 \\times 10^3$"
- Explain each parameter in simple terms (100-150 words)
- Use analogies where helpful

### 3. What Does This Mean?
- Explain the physical meaning in accessible language (150-200 words)
- Use everyday analogies (e.g., "like water flowing down a drain")
- Connect to real-world observations

### 4. Comparing Models (if YOU fitted multiple models)
- Simply explain which of YOUR models works better (100-150 words)
- Use non-technical language
- Show why one is better than the other

### 5. What Should You Do Next?
- Practical next steps (50-100 words)
- Suggestions for learning more
- Recommendations for further observations

**LaTeX REQUIREMENTS**:
- Use LaTeX for ALL numbers and equations
- Keep equations simple: $$\\text{PSD} = \\frac{A}{f^b} + n$$
- Always explain what each symbol means right after introducing it
            """,
            ExpertiseLevel.INTERMEDIATE: base_instruction + """
**RESPONSE GUIDELINES FOR INTERMEDIATE LEVEL:**

**MINIMUM LENGTH: 800-1200 WORDS** (This is MANDATORY, not a suggestion)

**REQUIRED STRUCTURE** (Use these section headers):

### 1. Overview of My Analysis
- Briefly introduce what YOU calculated (50-100 words)

### 2. Power Law Model Results (if applicable)
- Present YOUR fitted equation with LaTeX: $$\\text{PSD}(f) = \\frac{A}{f^b} + n$$
- State YOUR parameters: $A = ...$, $b = ...$, $n = ...$
- Explain what each parameter means (100-150 words)

### 3. Bending Power Law Model Results (if applicable)
- Present YOUR fitted equation with LaTeX: $$\\text{PSD}(f) = \\frac{A}{f[1+(f/f_b)^{\\alpha-1}]} + n$$
- State YOUR parameters: $A = ...$, $f_b = ...$ Hz, $\\alpha = ...$, $n = ...$
- Calculate and show characteristic timescale: $t_b = 1/f_b \\approx ...$ s
- Explain what each parameter means (150-200 words)

### 4. Physical Interpretation
- Relate YOUR power law index to accretion disk physics (150-200 words)
- Explain what YOUR break frequency tells us about the system
- Compare YOUR values with typical values from literature
- Discuss what physical processes are indicated

### 5. Model Comparison (if YOU fitted both models)
- Compare YOUR chi-squared values: $\\chi^2_\\nu$ for each model
- Explain which of YOUR fits is better and why (200-250 words)
- Discuss implications of the better fit

### 6. Conclusions and Recommendations
- Summarize YOUR key findings
- Recommend which model to use for further analysis
- Suggest follow-up observations or analyses

**LaTeX REQUIREMENTS**:
- Every equation in display format: $$...$$
- Every parameter value in inline math: $A = 2.67 \\times 10^3$
- Proper notation: $\\chi^2_\\nu$, $f_b$, $\\alpha$, not "chi-squared", "fb", "alpha"
            """,
            ExpertiseLevel.ADVANCED: base_instruction + """
**RESPONSE GUIDELINES FOR ADVANCED LEVEL:**

**MINIMUM LENGTH: 1200-1800 WORDS** (This is MANDATORY, not a suggestion)

**REQUIRED STRUCTURE** (Use these section headers):

### 1. Overview of My Analysis
- Technical introduction to YOUR approach (100-150 words)
- Mention data processing and quality checks YOU performed

### 2. Detailed Model Fitting Results
- Present YOUR fitted equations with full LaTeX notation
- Power law: $$\\text{PSD}(f) = \\frac{A}{f^b} + n$$
- Bending: $$\\text{PSD}(f) = \\frac{A}{f\\left[1 + \\left(\\frac{f}{f_b}\\right)^{\\alpha-1}\\right]} + n$$
- State ALL YOUR parameters with uncertainties if available (200-250 words)
- Discuss parameter correlations and degeneracies

### 3. Physical Interpretation and Theoretical Context
- Detailed discussion of YOUR results in context of accretion physics (300-400 words)
- Connect to specific theoretical models (e.g., Shakura-Sunyaev disk)
- Compare YOUR values with published results from similar sources
- Cite specific papers and missions
- Discuss systematic uncertainties

### 4. Model Comparison and Statistical Analysis
- Thorough comparison of YOUR fits (250-300 words)
- Present YOUR $\\chi^2_\\nu$, Akaike Information Criterion (AIC), or Bayesian Information Criterion (BIC)
- Discuss which model is statistically preferred
- Evaluate alternative models (e.g., Lorentzian components, broken power law)
- Parameter space analysis

### 5. Observational Implications
- What do YOUR results tell us about the system? (200-250 words)
- Spectral state implications
- Comparison with other wavelength observations
- Constraints on system parameters (mass, spin, inclination)

### 6. Conclusions and Advanced Recommendations
- Summarize YOUR key findings
- Suggest advanced follow-up analyses
- Recommend specific observing strategies
- Discuss open questions raised by YOUR analysis

**LaTeX & Technical REQUIREMENTS**:
- Full mathematical rigor with proper notation
- Include error propagation where relevant: $\\sigma_b = ...$
- Proper statistical notation: $\\chi^2_\\nu$, $p$-values, confidence intervals
- Reference equations by number if multiple derivations
            """,
            ExpertiseLevel.EXPERT: base_instruction + """
**RESPONSE GUIDELINES FOR EXPERT LEVEL:**

**MINIMUM LENGTH: 1800-2500 WORDS** (This is MANDATORY, not a suggestion)

**REQUIRED STRUCTURE** (Use these section headers):

### 1. Comprehensive Analysis Overview
- Detailed description of YOUR complete analysis pipeline (150-200 words)
- Data reduction methodology YOU used
- Quality control procedures YOU applied
- Justification for YOUR modeling choices

### 2. Detailed Fitting Methodology and Results
- Full mathematical formulation of YOUR models (300-400 words)
- Power law: $$\\text{PSD}(f) = \\frac{A}{f^b} + n$$
- Bending power law with full derivation
- Alternative parameterizations YOU considered
- Complete parameter table with YOUR fitted values and uncertainties
- Covariance matrices if relevant
- Posterior distributions for Bayesian approaches

### 3. Comprehensive Physical Interpretation
- In-depth theoretical context for YOUR results (400-500 words)
- Connection to specific accretion flow models
- Viscous timescales: $t_\\text{visc} \\sim r^2/\\nu$
- Orbital timescales: $t_\\text{orb} \\sim (r^3/GM)^{1/2}$
- Detailed comparison with YOUR break frequency
- Multi-wavelength context
- Comparison with theoretical predictions

### 4. Statistical Model Comparison and Validation
- Rigorous statistical comparison of YOUR models (350-450 words)
- Multiple goodness-of-fit metrics YOU calculated
- Model selection criteria (AIC, BIC, Bayes factors)
- Cross-validation results
- Residual analysis and systematic checks YOU performed
- Discussion of model adequacy and limitations
- Alternative models and why YOU rejected them

### 5. Degeneracies, Systematics, and Selection Effects
- Thorough discussion of YOUR error budget (250-300 words)
- Parameter degeneracies in YOUR fits
- Systematic uncertainties: background subtraction, dead time, pile-up
- Selection effects: frequency range, binning, red noise leak
- How these affect YOUR conclusions

### 6. Broader Context and Future Directions
- YOUR results in context of recent literature (200-250 words)
- Comparison with state-of-the-art missions (NICER, Insight-HXMT, etc.)
- Implications for open questions in the field
- Specific research directions enabled by YOUR findings

### 7. Detailed Conclusions and Methodological Recommendations
- Comprehensive summary of YOUR findings
- Specific recommendations for similar analyses
- Optimal observing strategies based on YOUR experience
- Suggestions for theoretical work to interpret YOUR results

**LaTeX & Research-Level REQUIREMENTS**:
- Peer-review quality mathematical exposition
- Complete derivations where illuminating
- Proper statistical framework: likelihood, priors, posteriors
- Error propagation: $\\sigma_f = \\sqrt{\\sum_i (\\partial f/\\partial x_i)^2 \\sigma_{x_i}^2}$
- Advanced notation: tensors, covariance matrices if needed
- Reference all relevant literature (Author et al. YEAR style)
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
        lines.append("**YOUR POWER LAW FIT RESULTS** (CITE AS YOUR OWN CALCULATIONS!):")
        lines.append("="*70)
        lines.append("Model YOU fitted: $$\\text{PSD}(f) = \\frac{A}{f^b} + n$$\n")
        
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
        
        lines.append("\n**INSTRUCTION:** Present these as YOUR fitted values and interpret what YOUR results mean!")
        
        return "\n".join(lines) + "\n"
    
    @staticmethod
    def _format_bending_power_law(
        bending: Dict[str, Any],
        expertise_level: ExpertiseLevel
    ) -> str:
        """Format bending power law fit results with STRONG emphasis"""
        lines = ["\n" + "="*70]
        lines.append("**YOUR BENDING POWER LAW FIT RESULTS** (CITE AS YOUR OWN CALCULATIONS!):")
        lines.append("="*70)
        lines.append(
            "Model YOU fitted: $$\\text{PSD}(f) = \\frac{A}{f\\left[1 + \\left(\\frac{f}{f_b}\\right)^{\\alpha-1}\\right]} + n$$\n"
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
        
        lines.append("\n**INSTRUCTION:** Present YOUR break frequency result and explain what it tells us!")
        
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