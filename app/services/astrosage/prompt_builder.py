# app/services/astrosage/prompt_builder.py

"""
Dynamic Prompt Builder with Routing Strategy Integration
ใช้ RoutingStrategy จาก Orchestrator แทนการ detect context เอง
"""

import logging
from typing import List, Dict, Any, Optional

from app.core.constants import RoutingStrategy
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
    
    KEY IMPROVEMENT: ใช้ RoutingStrategy จาก Orchestrator
    - ASTROSAGE: Educational mode (no analysis context)
    - ANALYSIS: Data analysis mode (มี analysis results)
    - MIXED: Hybrid mode (both analysis and explanation)
    """
    
    # Base system prompt (ครอบคลุมทุกสาขา astronomy)
    BASE_SYSTEM_PROMPT = """You are AstroSage, an expert AI assistant in astronomy and astrophysics with comprehensive knowledge across all subdisciplines:

**YOUR EXPERTISE SPANS:**
- Observational Astronomy: optical, radio, X-ray, gamma-ray, infrared, UV
- Stellar Physics: evolution, structure, nucleosynthesis, stellar populations
- Galactic Astronomy: Milky Way structure, star formation, interstellar medium
- Extragalactic Astronomy: galaxies, AGN, quasars, large-scale structure
- Cosmology: Big Bang, dark matter, dark energy, CMB, inflation
- High-Energy Astrophysics: compact objects, accretion, jets, GRBs
- Planetary Science: exoplanets, solar system, planetary formation
- Time-Domain Astronomy: transients, variability, periodic phenomena
- Instrumentation: telescopes, detectors, data reduction, spectroscopy
- Data Analysis: statistics, time series, spectral analysis, imaging

**YOUR ROLE - DETERMINED BY ROUTING STRATEGY:**

The workflow orchestrator has classified this request and determined the routing strategy.
Your response style adapts based on this classification:

1. **ASTROSAGE Strategy** (Pure question-answering):
   - User asking general astronomy questions
   - No data analysis results available
   - Response mode: Educational expert
   - Tone: "Black holes are...", "This phenomenon occurs..."
   - Draw from established astronomical knowledge

2. **ANALYSIS Strategy** (Pure data analysis):
   - User has uploaded FITS data for analysis
   - Analysis results are available
   - Response mode: Integrated analysis system
   - Tone: "I calculated...", "My analysis shows...", "I found..."
   - Present YOUR computational results with authority

3. **MIXED Strategy** (Analysis + Explanation):
   - User wants both analysis results AND broader context
   - Combine YOUR specific results with general knowledge
   - Response mode: Hybrid (analysis + education)
   - Tone: "I found $b = 0.802$... This is typical because..."
   - Connect YOUR results to established astronomy

**CRITICAL RESPONSE REQUIREMENTS:**

1. **LaTeX Equations - ABSOLUTELY MANDATORY**: 
   - Display equations: $$equation$$
   - Inline math: $parameter = value$
   - ALL mathematical expressions must use LaTeX
   - Example: $$\text{PSD}(f) = \frac{A}{f^b} + n$$

2. **Structure with Headers:**
   - Use ### for main sections
   - Organize logically based on question type

3. **Comprehensive Coverage:**
   - For ASTROSAGE: explain concepts thoroughly
   - For ANALYSIS: interpret YOUR specific results
   - For MIXED: combine both approaches

4. **Physical Interpretation:**
   - Always explain WHY things happen
   - Connect mathematics to physical reality
   - Relate to observational evidence

**RESPONSE LENGTH GUIDELINES:**
- Beginner: 500-800 words (detailed but accessible)
- Intermediate: 800-1200 words (thorough technical depth)
- Advanced: 1200-1800 words (research-level analysis)
- Expert: 1800-2500 words (comprehensive scholarly discussion)

**FORBIDDEN:**
- Never fake analysis results if none provided
- Never use first-person for general knowledge in ASTROSAGE mode
- Never skip LaTeX formatting
- Never give superficial answers"""

    @classmethod
    def build_full_prompt(
        cls, 
        request: AstroSageRequest,
        routing_strategy: RoutingStrategy  # NEW: จาก Orchestrator
    ) -> List[Dict[str, str]]:
        """
        Build complete prompt for LLM API with routing strategy
        
        Args:
            request: AstroSageRequest object
            routing_strategy: Strategy from Orchestrator's Classification Agent
        
        Returns:
            List of message dictionaries for API
        """
        logger.info(f"Building prompt with routing strategy: {routing_strategy.value}")
        
        # ============================================
        # STEP 1: Build base system prompt
        # ============================================
        system_prompt = cls.BASE_SYSTEM_PROMPT
        
        # ============================================
        # STEP 2: Build expertise-specific section
        # ============================================
        system_prompt += "\n\n" + cls.build_system_prompt(request.expertise_level)
        
        # ============================================
        # STEP 3: Add routing-specific framing
        # ============================================
        framing_instruction = cls._get_framing_by_strategy(routing_strategy)
        system_prompt += "\n\n" + framing_instruction
        
        # ============================================
        # STEP 4: Add conversation context if available
        # ============================================
        if request.conversation_history:
            conversation_context = cls.build_conversation_context(
                request.conversation_history
            )
            system_prompt += conversation_context
        
        # ============================================
        # STEP 5: Add analysis context based on strategy
        # ============================================
        if request.analysis_results:
            # Only add detailed analysis context for ANALYSIS or MIXED
            if routing_strategy in [RoutingStrategy.ANALYSIS, RoutingStrategy.MIXED]:
                analysis_context = cls.build_analysis_context(
                    request.analysis_results,
                    request.expertise_level,
                    routing_strategy
                )
                system_prompt += analysis_context
            else:
                # ASTROSAGE strategy: minimal analysis mention
                system_prompt += "\n\n**NOTE:** Analysis results exist but are not the focus of this query."
        
        # ============================================
        # STEP 6: Add final instruction checklist
        # ============================================
        system_prompt += cls._build_final_instruction(
            request.expertise_level,
            routing_strategy
        )
        
        # ============================================
        # STEP 7: Build messages list
        # ============================================
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
            f"expertise={request.expertise_level.value}, "
            f"strategy={routing_strategy.value}"
        )
        
        return messages
    
    @classmethod
    def _get_framing_by_strategy(cls, strategy: RoutingStrategy) -> str:
        """
        Get response framing instruction based on routing strategy
        
        Args:
            strategy: Routing strategy from Orchestrator
        
        Returns:
            Framing instruction text
        """
        framings = {
            RoutingStrategy.ASTROSAGE: """
**CURRENT ROUTING: ASTROSAGE (Educational Mode)**

The Classification Agent determined this is a general astronomy question.
No data analysis is required.

**RESPONSE FRAMING:**
- ✅ Use educational, explanatory tone
- ✅ Speak naturally: "Black holes are...", "This occurs when..."
- ✅ Draw from established astronomical knowledge
- ✅ Provide comprehensive explanations with examples
- ✅ Use analogies appropriate to expertise level
- ✅ Reference observations, missions, and discoveries

**FORBIDDEN:**
- ❌ DO NOT use first-person for general facts: "I calculated that gravity..."
- ❌ DO NOT claim to have analyzed data that doesn't exist
- ❌ DO NOT present general knowledge as YOUR findings

**EXAMPLE GOOD RESPONSE:**
"Power spectral density (PSD) analysis is a technique used to study variability 
in astronomical time series. The PSD follows a power law: $$\text{PSD}(f) = Af^{-b}$$ 
where $b$ is the power law index. For accreting systems, typical values range from 
$b = 1$ to $b = 2$ because..."
""",
            
            RoutingStrategy.ANALYSIS: """
**CURRENT ROUTING: ANALYSIS (Data Analysis Mode)**

The Classification Agent determined this requires analyzing the user's FITS data.
Analysis results are available.

**RESPONSE FRAMING:**
- ✅ You ARE the integrated analysis system
- ✅ Use first-person: "I calculated...", "My analysis shows...", "I found..."
- ✅ Present YOUR fitted parameter values explicitly
- ✅ Cite YOUR specific numerical results with LaTeX
- ✅ Explain the physical meaning of YOUR results
- ✅ Compare YOUR values with literature when relevant

**FORBIDDEN:**
- ❌ DO NOT say "your analysis" or "your results" - these are YOUR results
- ❌ DO NOT give generic explanations without YOUR specific numbers
- ❌ DO NOT act as external observer; you ARE the analysis system

**EXAMPLE GOOD RESPONSE:**
"I calculated the power law fit for your light curve, obtaining $A = 2.67 \\times 10^3$, 
$b = 0.802$, and $n = 1.23 \\times 10^{-2}$. My analysis shows that the power law 
index of $b = 0.802$ indicates red noise variability. This value is slightly lower 
than typical neutron star LMXBs ($b \\approx 1.2-1.5$), suggesting..."
""",
            
            RoutingStrategy.MIXED: """
**CURRENT ROUTING: MIXED (Hybrid Mode)**

The Classification Agent determined this requires BOTH data analysis AND 
broader astronomical context.

**RESPONSE FRAMING:**
- ✅ Start with YOUR specific analysis results (first-person)
- ✅ Then explain general astronomical context (educational)
- ✅ Connect YOUR results to broader knowledge
- ✅ Compare YOUR values with typical/literature values
- ✅ Balance technical detail with accessibility

**STRUCTURE APPROACH:**
1. Present YOUR analysis results: "I found..."
2. Explain general context: "This type of variability occurs when..."
3. Compare YOUR results: "Compared to similar sources..."
4. Physical interpretation: "The reason YOUR value is significant..."
5. Recommendations based on YOUR findings

**EXAMPLE GOOD RESPONSE:**
"I calculated a power law index of $b = 0.802$ from your light curve (ANALYSIS). 
This falls in the red noise regime, where $b < 1$ (EXPLANATION). 

Power law indices in this range are characteristic of shot noise processes in 
accreting systems, where individual accretion events produce the observed variability 
(GENERAL CONTEXT).

Your value is slightly lower than typical Z sources ($b \\approx 1.2$) but consistent 
with atoll sources in the island state (COMPARISON). This suggests your source may 
be in a lower accretion rate state... (INTERPRETATION)"
"""
        }
        
        return framings.get(strategy, framings[RoutingStrategy.ASTROSAGE])
    
    @classmethod
    def build_system_prompt(cls, expertise_level: ExpertiseLevel) -> str:
        """
        Build system prompt based on expertise level
        (ใช้ version ใหม่จาก ExpertiseAdapter ที่ครอบคลุมทุกสาขา)
        """
        modifier = ExpertiseAdapter.get_system_prompt_modifier(expertise_level)
        latex_examples = cls._get_latex_examples()
        
        return modifier + "\n\n" + latex_examples
    
    @classmethod
    def build_analysis_context(
        cls,
        analysis_results: Optional[Dict[str, Any]],
        expertise_level: ExpertiseLevel,
        routing_strategy: RoutingStrategy
    ) -> str:
        """
        Build analysis results context with strategy-appropriate framing
        
        Input structure from Orchestrator (via Analysis Agent):
        {
            "metadata": {...},
            "statistics": {...},
            "psd": {...},
            "power_law": {...},
            "bending_power_law": {...}
        }

        Now includes metadata-enriched formatting for better interpretation
        
        Args:
            analysis_results: Results from Analysis Agent
            expertise_level: User's expertise level
            routing_strategy: Routing strategy (ANALYSIS, MIXED, ASTROSAGE)
        
        Returns:
            Formatted analysis context string
        """
        if not analysis_results:
            return ""
        
        # ============================================
        # Extract metadata (if available)
        # ============================================
        metadata = analysis_results.get('metadata', {})
        
        # Check if metadata has enhanced fields
        has_enhanced_metadata = (
            'critical_fields' in metadata or 
            'derived_quantities' in metadata or 
            'source_context' in metadata
        )
        
        lines = ["\n\n" + "="*70]
        lines.append("=== ANALYSIS RESULTS ===")
        lines.append("="*70 + "\n")
        
        # ============================================
        # Adjust header based on routing strategy
        # ============================================
        if routing_strategy == RoutingStrategy.ANALYSIS:
            lines.append("**YOUR ANALYSIS RESULTS** (that YOU computed):")
            lines.append("**MANDATORY:** Present these as YOUR findings using first person")
            lines.append("**YOU MUST CITE THESE EXACT NUMBERS AS YOUR OWN RESULTS!**\n")
        
        elif routing_strategy == RoutingStrategy.MIXED:
            lines.append("**YOUR ANALYSIS RESULTS + CONTEXT:**")
            lines.append("Present YOUR results first, then connect to general astronomy")
            lines.append("**START with YOUR specific fitted values!**\n")
        
        else:  # ASTROSAGE (shouldn't usually have detailed results)
            lines.append("**ANALYSIS RESULTS CONTEXT:**")
            lines.append("These are available for reference if relevant to the question\n")
        
        # ============================================
        # Format each analysis type if present
        # ============================================
        
        # Metadata
        # if 'metadata' in analysis_results:
        #     lines.append(cls._format_metadata(analysis_results['metadata']))

        # ============================================
        # Show compact metadata summary FIRST
        # ============================================
        if has_enhanced_metadata:
            lines.append(cls._format_metadata_compact(metadata))
            lines.append("\n" + "="*70 + "\n")
        
        # ============================================
        # Format each analysis type WITH metadata
        # ============================================
        
        # Statistics (no metadata needed)
        if 'statistics' in analysis_results:
            lines.append(cls._format_statistics(
                analysis_results['statistics'], 
                expertise_level
            ))
        
        # PSD (with timing metadata)
        if 'psd' in analysis_results:
            lines.append(cls._format_psd(
                analysis_results['psd'],
                metadata,
                expertise_level
            ))
        
        # Check if both models exist
        has_power_law = 'power_law' in analysis_results
        has_bending = 'bending_power_law' in analysis_results
        
        # Power Law (with source context)
        if has_power_law:
            lines.append(cls._format_power_law(
                analysis_results['power_law'],
                metadata,
                expertise_level,
                routing_strategy
            ))
        
        # Bending Power Law (with full context)
        if has_bending:
            lines.append(cls._format_bending_power_law(
                analysis_results['bending_power_law'],
                metadata,
                expertise_level,
                routing_strategy
            ))
        
        # ============================================
        # Model comparison instruction (if both models)
        # ============================================
        if has_power_law and has_bending:
            lines.append("\n" + "="*70)
            
            if routing_strategy == RoutingStrategy.ANALYSIS:
                lines.append("**CRITICAL: MODEL COMPARISON REQUIRED!**")
                lines.append("="*70)
                lines.append("""
Since YOU performed BOTH power law and bending power law fits, you MUST:

1. **State both sets of YOUR fitted parameters explicitly**
   - Use first person: "I calculated...", "My fit gave..."
   - Present YOUR power law: $A$, $b$, $n$
   - Present YOUR bending power law: $A$, $f_b$, $\\alpha$, $n$

2. **Compare YOUR models directly:**
   - Which of YOUR fits provides better results?
   - What does YOUR break frequency $f_b$ tell us?
   - How do the parameters from YOUR two fits differ?

3. **Physical interpretation of YOUR results:**
   - What does YOUR $f_b$ imply about characteristic timescales?
   - Does YOUR simple power law adequately capture the variability?
   - Is YOUR bending model necessary?

4. **YOUR Recommendations:**
   - Based on YOUR analysis, which model to use?
   - What follow-up analyses do YOU recommend?
""")
            
            elif routing_strategy == RoutingStrategy.MIXED:
                lines.append("**NOTE: TWO MODELS AVAILABLE FOR COMPARISON**")
                lines.append("="*70)
                lines.append("""
You have both power law and bending power law results:
- Compare the models if relevant to answering the question
- Explain what the break frequency tells us about the system
- Connect YOUR fitted values to the general concepts being discussed
""")
        
        # ============================================
        # Final reminder based on strategy
        # ============================================
        if routing_strategy == RoutingStrategy.ANALYSIS:
            lines.append("\n**CRITICAL REMINDER:** Present as YOUR calculations, cite YOUR numbers!")
        elif routing_strategy == RoutingStrategy.MIXED:
            lines.append("\n**REMINDER:** Start with YOUR results, then add context!")
        
        return "\n".join(lines)
    
    @classmethod
    def _build_final_instruction(
        cls, 
        expertise_level: ExpertiseLevel,
        routing_strategy: RoutingStrategy
    ) -> str:
        """
        Build final instruction checklist based on expertise and routing
        
        Args:
            expertise_level: User's expertise level
            routing_strategy: Routing strategy from Orchestrator
        
        Returns:
            Final instruction text with appropriate checklist
        """
        # Base checklist (always included)
        base_checklist = """
\n\n**FINAL MANDATORY CHECKLIST:**

☑ Have I used proper LaTeX formatting for ALL equations and parameters?
☑ Is my response sufficiently detailed for the expertise level?
☑ Have I provided accurate, comprehensive information?
☑ Have I structured my response with clear sections?
"""
        
        # Strategy-specific checklist
        strategy_checklist = {
            RoutingStrategy.ASTROSAGE: """
**ASTROSAGE MODE CHECKLIST:**
☑ Have I used educational/explanatory tone (not first-person)?
☑ Have I explained concepts clearly and thoroughly?
☑ Have I provided appropriate examples or analogies?
☑ Have I referenced relevant observations/missions?
☑ Have I avoided claiming to analyze non-existent data?
""",
            
            RoutingStrategy.ANALYSIS: """
**ANALYSIS MODE CHECKLIST:**
☑ Have I cited MY actual fitted parameter values?
☑ Have I used first-person consistently (I calculated, My fit...)?
☑ Have I compared MY models if both are available?
☑ Have I explained the physical meaning of MY results?
☑ Have I provided specific recommendations based on MY analysis?
☑ Have I compared MY values with literature when relevant?
""",
            
            RoutingStrategy.MIXED: """
**MIXED MODE CHECKLIST:**
☑ Have I started with MY specific analysis results?
☑ Have I then connected to general astronomical context?
☑ Have I compared MY values with typical/literature values?
☑ Have I balanced technical analysis with broader explanation?
☑ Have I used appropriate framing (first-person for MY results, educational for concepts)?
☑ Have I explained why MY results are significant?
"""
        }
        
        # Combine base + strategy-specific
        final_instruction = base_checklist + strategy_checklist.get(
            routing_strategy,
            strategy_checklist[RoutingStrategy.ASTROSAGE]
        )
        
        # Add expertise-specific notes
        if expertise_level == ExpertiseLevel.BEGINNER:
            final_instruction += """
**BEGINNER LEVEL REMINDER:**
- Use simple language and everyday analogies
- Explain ALL technical terms
- Break complex ideas into small steps
"""
        elif expertise_level == ExpertiseLevel.EXPERT:
            final_instruction += """
**EXPERT LEVEL REMINDER:**
- Include research-level details and citations
- Discuss systematic uncertainties
- Reference recent literature (last 2-3 years)
- Suggest advanced follow-up analyses
"""
        
        return final_instruction
    
    # ============================================
    # Helper methods (unchanged)
    # ============================================
    
    @classmethod
    def _get_latex_examples(cls) -> str:
        """LaTeX formatting examples (unchanged from previous version)"""
        return """
**LaTeX FORMATTING EXAMPLES:**

Display Equations (use $$...$$):
- Power Law: $$\\text{PSD}(f) = \\frac{A}{f^b} + n$$
- Bending Power Law: $$\\text{PSD}(f) = \\frac{A}{f\\left[1 + \\left(\\frac{f}{f_b}\\right)^{\\alpha-1}\\right]} + n$$
- Chi-squared: $$\\chi^2 = \\sum_{i=1}^N \\frac{(O_i - E_i)^2}{\\sigma_i^2}$$

Inline Math (use $...$):
- Parameters: $A = 2.67 \\times 10^3$, $b = 0.802$, $f_b = 4.06 \\times 10^{-5}$ Hz
- Ranges: $f \\in [10^{-5}, 10^{-2}]$ Hz
- Comparisons: if $b > 1$, then...

**IMPORTANT:** Always explain what each variable means!
"""
    
    @classmethod
    def build_conversation_context(
        cls, 
        conversations: Optional[List[ConversationPair]]
    ) -> str:
        """Build conversation context (unchanged)"""
        if not conversations:
            return ""
        
        lines = ["\n\n=== PREVIOUS CONVERSATION HISTORY ===\n"]
        lines.append("(Last 10 exchanges - maintain continuity)\n")
        
        for i, pair in enumerate(conversations, 1):
            time_str = cls._format_timestamp(pair.timestamp)
            lines.append(f"\n**Exchange {i}** ({time_str}):")
            lines.append(f"USER: {pair.user_message}")
            
            assistant_msg = pair.assistant_message
            if len(assistant_msg) > 300:
                assistant_msg = assistant_msg[:300] + "..."
            lines.append(f"ASTROSAGE: {assistant_msg}\n")
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_metadata_compact(metadata: Dict[str, Any]) -> str:
        """
        Format compact metadata summary for context
        
        Shows essential observation information without overwhelming detail
        """
        lines = ["**OBSERVATION SUMMARY:**\n"]
        
        # ============================================
        # Source & Instrument
        # ============================================
        if 'critical_fields' in metadata:
            crit = metadata['critical_fields']
            
            source = crit.get('source_name', 'Unknown')
            telescope = crit.get('telescope', '?')
            instrument = crit.get('instrument', '?')
            obs_id = crit.get('observation_id', '?')
            
            lines.append(f"- **Source**: {source}")
            lines.append(f"- **Instrument**: {telescope}/{instrument} (ObsID: {obs_id})")
        
        # ============================================
        # Timing & Energy
        # ============================================
        if 'derived_quantities' in metadata:
            deriv = metadata['derived_quantities']
            
            duration = deriv.get('observation_duration_hours', 0)
            energy_band = deriv.get('energy_band_short', 'N/A')
            duty_cycle = deriv.get('duty_cycle_percent', 0)
            
            lines.append(f"- **Duration**: {duration:.1f} hours (duty cycle: {duty_cycle:.1f}%)")
            lines.append(f"- **Energy Band**: {energy_band}")
            
            # Frequency constraints
            if 'nyquist_frequency_hz' in deriv:
                f_nyq = deriv['nyquist_frequency_hz']
                f_min = deriv.get('min_frequency_hz', 0)
                lines.append(f"- **Frequency Range**: ${f_min:.2e}$ - ${f_nyq:.3f}$ Hz (accessible)")
        
        # ============================================
        # Source Type (if known)
        # ============================================
        if 'source_context' in metadata:
            ctx = metadata['source_context']
            
            if ctx.get('is_known_source'):
                lines.append(f"\n**Source Type**: {ctx.get('type', 'Unknown')}")
                
                # Key characteristics (first 2)
                if 'characteristics' in ctx:
                    chars = ctx['characteristics'][:2]
                    lines.append(f"- Known for: {', '.join(chars)}")
        
        return "\n".join(lines) + "\n"
    
    # ============================================
    # Updated format methods with metadata
    # ============================================
    
    @staticmethod
    def _format_statistics(
        stats: Dict[str, Any],
        expertise_level: ExpertiseLevel
    ) -> str:
        """
        Format statistics results with appropriate detail
        
        Input structure from Analysis Agent:
        {
            "statistics": {"mean": 123.45, "std": 15.6, ...},
            "n_data_points": 10000,
            "parameters_used": {...}
        }
        """
        lines = ["**Statistical Summary:**\n"]
        
        # Extract nested statistics dict
        statistics = stats.get('statistics', {})
        n_points = stats.get('n_data_points', 0)
        
        if n_points > 0:
            lines.append(f"- Data Points: $N = {n_points:,}$")
        
        # Basic statistics (always show if available)
        if 'mean' in statistics:
            lines.append(f"- Mean Rate: $\\langle R \\rangle = {statistics['mean']:.6e}$ counts/s")
        
        if 'median' in statistics:
            lines.append(f"- Median Rate: $R_\\text{{median}} = {statistics['median']:.6e}$ counts/s")
        
        if 'std' in statistics:
            lines.append(f"- Standard Deviation: $\\sigma = {statistics['std']:.6e}$ counts/s")
        
        if 'min' in statistics:
            lines.append(f"- Minimum: $R_\\text{{min}} = {statistics['min']:.6e}$ counts/s")
        
        if 'max' in statistics:
            lines.append(f"- Maximum: $R_\\text{{max}} = {statistics['max']:.6e}$ counts/s")
        
        # Additional statistics for advanced users
        # if expertise_level in [ExpertiseLevel.ADVANCED, ExpertiseLevel.EXPERT]:
            
        # Percentiles
        percentile_keys = [k for k in statistics.keys() if k.startswith('percentile_')]
        if percentile_keys:
            lines.append("\n**Percentiles:**")
            for key in sorted(percentile_keys):
                p = key.replace('percentile_', '')
                lines.append(f"- ${p}^\\text{{th}}$ percentile: ${statistics[key]:.6e}$")
        
        # Quantiles
        quantile_keys = [k for k in statistics.keys() if k.startswith('quantile_')]
        if quantile_keys:
            lines.append("\n**Quantiles:**")
            for key in sorted(quantile_keys):
                q = key.replace('quantile_', '').replace('_', '.')
                lines.append(f"- $q = {q}$: ${statistics[key]:.6e}$")
        
        # Distribution summary
        if 'distribution_summary' in statistics:
            dist_sum = statistics['distribution_summary']
            lines.append("\n**Distribution Summary:**")
            
            if 'range' in dist_sum:
                r = dist_sum['range']
                lines.append(f"- Range: $[{r['min']:.6e}, {r['max']:.6e}]$, span = ${r['span']:.6e}$")
            
            if 'iqr' in dist_sum:
                iqr = dist_sum['iqr']
                lines.append(f"- IQR: $Q_3 - Q_1 = {iqr['iqr']:.6e}$")
                lines.append(f"- Outlier fences: $[{iqr['lower_fence']:.6e}, {iqr['upper_fence']:.6e}]$")
            
            if 'coefficient_of_variation' in dist_sum:
                cv = dist_sum['coefficient_of_variation']
                lines.append(f"- Coefficient of Variation: $CV = {cv:.3f}$")
            
            if 'skewness' in dist_sum:
                skew = dist_sum['skewness']
                lines.append(f"- Skewness: ${skew:.3f}$")
            
            if 'kurtosis' in dist_sum:
                kurt = dist_sum['kurtosis']
                lines.append(f"- Kurtosis: ${kurt:.3f}$")
        
        return "\n".join(lines) + "\n"
    
    @staticmethod
    def _format_psd(
        psd: Dict[str, Any],
        metadata: Dict[str, Any],
        expertise_level: ExpertiseLevel
    ) -> str:
        """
        Format PSD results with appropriate detail
        
        Input structure from Analysis Agent:
        {
            "n_points": 3500,
            "freq_range": [1e-5, 0.05],
            "psd_range": [1.2e-5, 3.4e-3],
            "frequencies_sample": [...],
            "psd_values_sample": [...],
            "parameters_used": {...}
        }
        """
        lines = ["**Power Spectral Density Analysis:**\n"]
        
        # ============================================
        # Frequency range with context
        # ============================================
        if 'freq_range' in psd:
            freq_range = psd['freq_range']
            f_min, f_max = freq_range[0], freq_range[1]
            lines.append(f"- Frequency Range: $f \\in [{f_min:.2e}, {f_max:.2e}]$ Hz")
            
            # Add period range
            if f_min > 0 and f_max > 0:
                p_max = 1.0 / f_min
                p_min = 1.0 / f_max
                lines.append(f"- Period Range: $P \\in [{p_min:.2e}, {p_max:.2e}]$ s")
                
                # Convert to human-readable units
                if p_max > 3600:
                    lines.append(f"  → Longest timescale: ~{p_max/3600:.1f} hours")
                if p_min < 60:
                    lines.append(f"  → Shortest timescale: ~{p_min:.1f} seconds")
            
            # Add context from metadata
            if 'derived_quantities' in metadata:
                deriv = metadata['derived_quantities']
                
                if 'observation_duration_hours' in deriv:
                    duration = deriv['observation_duration_hours']
                    lines.append(f"\n*Context: Your {duration:.1f}-hour observation allows probing these timescales*")
        
        # Number of points
        if 'n_points' in psd:
            lines.append(f"\n- Number of Frequency Bins: $N_\\text{{bins}} = {psd['n_points']:,}$")
        
        # PSD range
        if 'psd_range' in psd:
            psd_range = psd['psd_range']
            lines.append(f"- PSD Range: $[{psd_range[0]:.2e}, {psd_range[1]:.2e}]$ (rms²/Hz)")
        
        
        # Parameters used (for advanced users)
        if expertise_level in [ExpertiseLevel.ADVANCED, ExpertiseLevel.EXPERT]:
            if 'parameters_used' in psd:
                params = psd['parameters_used']
                lines.append("\n**PSD Parameters:**")
                if 'low_freq' in params:
                    lines.append(f"- Lower frequency cutoff: ${params['low_freq']:.2e}$ Hz")
                if 'high_freq' in params:
                    lines.append(f"- Upper frequency cutoff: ${params['high_freq']:.2e}$ Hz")
                if 'bins' in params:
                    lines.append(f"- Requested bins: {params['bins']}")
        
        return "\n".join(lines) + "\n"
    
    @staticmethod
    def _format_power_law(
        power_law: Dict[str, Any],
        metadata: Dict[str, Any],
        expertise_level: ExpertiseLevel,
        routing_strategy: RoutingStrategy
    ) -> str:
        """
        Format power law fit results with strategy-appropriate framing
        
        Input structure from Analysis Agent:
        {
            "model": "power_law",
            "fitted_parameters": {"A": 2.67e3, "b": 0.802, "n": 1.23e-2},
            "initial_parameters": {"A": 1.0, "b": 1.0},
            "parameter_bounds": {"A": [0.0, 1e38], "b": [0.1, 3.0]},
            "parameters_used": {...}
        }
        """
        lines = ["\n" + "="*70]
        
        # Adjust header based on routing strategy
        if routing_strategy == RoutingStrategy.ANALYSIS:
            lines.append("**YOUR POWER LAW FIT RESULTS:**")
            lines.append("(Present these as YOUR calculations)")
        elif routing_strategy == RoutingStrategy.MIXED:
            lines.append("**YOUR POWER LAW FIT:**")
            lines.append("(Start with these, then add context)")
        else:
            lines.append("**POWER LAW FIT RESULTS:**")
            lines.append("(Reference if relevant)")
        
        lines.append("="*70)
        lines.append("Model: $$\\text{PSD}(f) = \\frac{A}{f^b} + n$$\n")

        # ============================================
        # Fitted parameters
        # ============================================
        fitted = power_law.get('fitted_parameters', {})
        
        if 'A' in fitted:
            lines.append(f"- **Amplitude**: $A = {fitted['A']:.6e}$")
        
        if 'b' in fitted:
            b_val = fitted['b']
            lines.append(f"- **Power Law Index**: $b = {b_val:.3f}$")
            
            # Add interpretation hint
            if b_val < 1:
                lines.append("  → *Suggests red noise / flicker noise regime*")
            elif b_val < 2:
                lines.append("  → *Typical for accreting source variability*")
            else:
                lines.append("  → *Steep spectrum / white noise dominated*")
        
        if 'n' in fitted:
            lines.append(f"- **Noise Level**: $n = {fitted['n']:.6e}$")
        
        # ============================================
        # CONTEXT from metadata
        # ============================================
        if metadata and 'source_context' in metadata:
            ctx = metadata['source_context']
            
            if ctx.get('is_known_source') and 'typical_psd' in ctx:
                typical = ctx['typical_psd']
                b_range = typical.get('power_law_index_range', [])
                
                if b_range and 'b' in fitted:
                    b_val = fitted['b']
                    lines.append(f"\n**Literature Comparison:**")
                    lines.append(f"- Typical for {ctx.get('source_name')}: $b \\approx {b_range[0]}-{b_range[1]}$")
                    lines.append(f"- Your value: $b = {b_val:.3f}$")
                    
                    # Interpretation
                    if b_val < b_range[0]:
                        lines.append(f"  → *Your value is LOWER than typical (flatter spectrum)*")
                    elif b_val > b_range[1]:
                        lines.append(f"  → *Your value is HIGHER than typical (steeper spectrum)*")
                    else:
                        lines.append(f"  → *Your value is within typical range*")
                
                if 'notes' in typical:
                    lines.append(f"\n*Note: {typical['notes']}*")
        
        # Add energy band context
        if metadata and 'derived_quantities' in metadata:
            deriv = metadata['derived_quantities']
            if 'energy_band' in deriv:
                lines.append(f"\n*This fit is from {deriv['energy_band']} data*")
        
        # Expert level: show initial parameters and bounds
        if expertise_level in [ExpertiseLevel.ADVANCED, ExpertiseLevel.EXPERT]:
            if 'initial_parameters' in power_law:
                init = power_law['initial_parameters']
                lines.append("\n**Initial Guess:**")
                for param, value in init.items():
                    lines.append(f"- ${param}_0 = {value:.3e}$")
            
            if 'parameter_bounds' in power_law:
                bounds = power_law['parameter_bounds']
                lines.append("\n**Parameter Bounds:**")
                for param, bound in bounds.items():
                    lower = bound[0]
                    upper = bound[1] if isinstance(bound[1], (int, float)) else bound[1]
                    if upper == "unbounded":
                        lines.append(f"- ${param} \\in [{lower:.3e}, \\infty)$")
                    else:
                        lines.append(f"- ${param} \\in [{lower:.3e}, {upper:.3e}]$")
        
        # Context-specific instruction
        if routing_strategy == RoutingStrategy.ANALYSIS:
            lines.append("\n**Instruction:** Present these as YOUR fitted values")
        elif routing_strategy == RoutingStrategy.MIXED:
            lines.append("\n**Instruction:** Reference these values if relevant to the question")
        else:
            lines.append("\n**Instruction:** Use only if directly relevant to answering the question")
        
        return "\n".join(lines) + "\n"
    
    @staticmethod
    def _format_bending_power_law(
        bending: Dict[str, Any],
        metadata: Dict[str, Any],
        expertise_level: ExpertiseLevel,
        routing_strategy: RoutingStrategy
    ) -> str:
        """
        Format bending power law fit results with strategy-appropriate framing
        
        Input structure from Analysis Agent:
        {
            "model": "bending_power_law",
            "fitted_parameters": {"A": 3.45e3, "fb": 4.06e-5, "sh": 1.23, "n": 1.18e-2},
            "initial_parameters": {"A": 10.0, "fb": 0.01, "sh": 1.0},
            "parameter_bounds": {...},
            "parameters_used": {...}
        }
        """
        lines = ["\n" + "="*70]
        
        # Adjust header based on routing strategy
        if routing_strategy == RoutingStrategy.ANALYSIS:
            lines.append("**YOUR BENDING POWER LAW FIT RESULTS:**")
            lines.append("(Present these as YOUR calculations)")
        elif routing_strategy == RoutingStrategy.MIXED:
            lines.append("**YOUR BENDING POWER LAW FIT:**")
            lines.append("(Start with these, then add context)")
        else:
            lines.append("**BENDING POWER LAW FIT RESULTS:**")
            lines.append("(Reference if relevant)")
        
        lines.append("="*70)
        lines.append(
            "Model: $$\\text{PSD}(f) = \\frac{A}{f\\left[1 + \\left(\\frac{f}{f_b}\\right)^{\\alpha-1}\\right]} + n$$\n"
        )
        
        # ============================================
        # Fitted parameters
        # ============================================
        fitted = bending.get('fitted_parameters', {})
        
        if 'A' in fitted:
            lines.append(f"- **Amplitude**: $A = {fitted['A']:.6e}$")
        
        if 'fb' in fitted:
            fb = fitted['fb']
            lines.append(f"- **Break Frequency**: $f_b = {fb:.6e}$ Hz")
            
            # Calculate timescale
            if fb > 0:
                t_break = 1.0 / fb
                lines.append(f"  → **Characteristic Timescale**: $t_b \\approx {t_break:.2e}$ s")
                
                # Convert to human-readable
                if t_break < 60:
                    lines.append(f"  → (~{t_break:.1f} seconds)")
                elif t_break < 3600:
                    lines.append(f"  → (~{t_break/60:.1f} minutes)")
                else:
                    lines.append(f"  → (~{t_break/3600:.1f} hours)")
                
                # Physical interpretation
                if t_break < 1:
                    lines.append("  → *Very short timescale: inner disk / orbital periods*")
                elif t_break < 100:
                    lines.append("  → *Short timescale: viscous/thermal processes*")
                elif t_break < 10000:
                    lines.append("  → *Intermediate: disk instabilities*")
                else:
                    lines.append("  → *Long timescale: outer disk / binary effects*")
        
        if 'sh' in fitted:
            lines.append(f"- **Shape Parameter**: $\\alpha = {fitted['sh']:.3f}$")
        
        if 'n' in fitted:
            lines.append(f"- **Noise Level**: $n = {fitted['n']:.6e}$")
        
        # ============================================
        # CONTEXT from metadata
        # ============================================
        if metadata and 'source_context' in metadata:
            ctx = metadata['source_context']
            
            if ctx.get('is_known_source') and 'typical_psd' in ctx:
                typical = ctx['typical_psd']
                fb_range = typical.get('break_frequency_range', [])
                
                if fb_range and 'fb' in fitted:
                    fb_val = fitted['fb']
                    lines.append(f"\n**Literature Comparison:**")
                    lines.append(f"- Typical for {ctx.get('source_name')}: $f_b \\approx {fb_range[0]:.0e}-{fb_range[1]:.0e}$ Hz")
                    lines.append(f"- Your value: $f_b = {fb_val:.2e}$ Hz")
                    
                    # Interpretation
                    if fb_val < fb_range[0]:
                        lines.append(f"  → *Your break is at LOWER frequency (longer timescales)*")
                    elif fb_val > fb_range[1]:
                        lines.append(f"  → *Your break is at HIGHER frequency (shorter timescales)*")
                    else:
                        lines.append(f"  → *Your break is within typical range*")
                
                if 'notes' in typical:
                    lines.append(f"\n*Note: {typical['notes']}*")
            
            # Add black hole mass context
            if 'black_hole_mass_solar' in ctx and 'fb' in fitted:
                bh_mass = ctx['black_hole_mass_solar']
                fb_val = fitted['fb']
                t_break = 1.0 / fb_val if fb_val > 0 else 0
                
                # Gravitational radius: Rg = GM/c^2
                # For M_sun: Rg ~ 1.5 km ~ 5e-6 light-seconds
                rg_seconds = 5e-6 * bh_mass  # Rg in light-seconds
                
                lines.append(f"\n**Physical Scales** (for $M_{{BH}} \\approx {bh_mass:.1e} M_\\odot$):")
                lines.append(f"- Gravitational radius: $R_g \\approx {rg_seconds:.2e}$ light-s")
                lines.append(f"- Break timescale: $t_b \\approx {t_break:.2e}$ s")
                
                if t_break > 0:
                    scale_ratio = t_break / rg_seconds
                    lines.append(f"- Scale: $t_b / (R_g/c) \\approx {scale_ratio:.1e}$")
        
        # Add observation duration comparison
        if metadata and 'derived_quantities' in metadata:
            deriv = metadata['derived_quantities']
            
            if 'observation_duration_seconds' in deriv and 'fb' in fitted:
                obs_duration = deriv['observation_duration_seconds']
                fb_val = fitted['fb']
                t_break = 1.0 / fb_val if fb_val > 0 else 0
                
                if t_break > 0:
                    n_cycles = obs_duration / t_break
                    lines.append(f"\n*Your observation covers ~{n_cycles:.1f} characteristic cycles*")
        
        # Expert level: show initial parameters and bounds
        if expertise_level in [ExpertiseLevel.ADVANCED, ExpertiseLevel.EXPERT]:
            if 'initial_parameters' in bending:
                init = bending['initial_parameters']
                lines.append("\n**Initial Guess:**")
                for param, value in init.items():
                    if param == 'sh':
                        lines.append(f"- $\\alpha_0 = {value:.3f}$")
                    elif param == 'fb':
                        lines.append(f"- $f_{{b,0}} = {value:.3e}$")
                    else:
                        lines.append(f"- ${param}_0 = {value:.3e}$")
            
            if 'parameter_bounds' in bending:
                bounds = bending['parameter_bounds']
                lines.append("\n**Parameter Bounds:**")
                for param, bound in bounds.items():
                    lower = bound[0]
                    upper = bound[1] if isinstance(bound[1], (int, float)) else bound[1]
                    
                    # Use proper notation
                    if param == 'sh':
                        param_name = "\\alpha"
                    elif param == 'fb':
                        param_name = "f_b"
                    else:
                        param_name = param
                    
                    if upper == "unbounded":
                        lines.append(f"- ${param_name} \\in [{lower:.3e}, \\infty)$")
                    else:
                        lines.append(f"- ${param_name} \\in [{lower:.3e}, {upper:.3e}]$")
        
        # Context-specific instruction
        if routing_strategy == RoutingStrategy.ANALYSIS:
            lines.append("\n**Instruction:** Present YOUR break frequency result and explain what it tells us!")
        elif routing_strategy == RoutingStrategy.MIXED:
            lines.append("\n**Instruction:** Reference YOUR fitted values when relevant")
        else:
            lines.append("\n**Instruction:** Use only if directly relevant")
        
        return "\n".join(lines) + "\n"
    
    @staticmethod
    def _format_timestamp(timestamp) -> str:
        """Format timestamp for display (unchanged)"""
        from datetime import datetime, timezone
        
        now = datetime.now()
        if timestamp.tzinfo is not None:
            now = now.replace(tzinfo=timezone.utc)
        
        delta = now - timestamp
        seconds = delta.total_seconds()
        
        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            return f"{int(seconds/60)}m ago"
        elif seconds < 86400:
            return f"{int(seconds/3600)}h ago"
        else:
            return f"{int(seconds/86400)}d ago"
        