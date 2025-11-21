"""
app/agents/rewrite/prompt_builder.py - COMPLETE FIXED VERSION
"""

from typing import List, Dict, Any, Optional
import json
import logging

from app.agents.rewrite.models import RewriteRequest

logger = logging.getLogger(__name__)


class RewritePromptBuilder:
    """Build GPT prompts for response rewriting"""
    
    # ✅ FIXED: Updated system prompt with preservation focus
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
   - ✅ **PRESERVE 100% of AstroSage content** (DO NOT SUMMARIZE OR SHORTEN)
   - ✅ **Keep ALL LaTeX equations from AstroSage exactly as written**
   - ✅ **Maintain AstroSage's complete interpretation and discussion**
   - ✅ **Keep ALL sections, paragraphs, and derivations from AstroSage**
   - ✅ **Your job is FORMATTING ONLY** - add structure, fix LaTeX syntax
   - ✅ **You are NOT an editor, summarizer, or content creator**
   - Cross-reference numerical results between analysis and interpretation
   - Add formatting/structure ONLY, not content changes

4. **Adapt to expertise level**: {expertise_level}
{expertise_guidelines}

CRITICAL RULES:
- NEVER change numerical values from the analysis results
- ✅ NEVER summarize, shorten, or condense AstroSage's content
- ✅ NEVER paraphrase or rewrite AstroSage's explanations
- ✅ PRESERVE ALL mathematical derivations and equations from AstroSage
- ✅ KEEP ALL sections, discussions, and interpretations from AstroSage
- ALWAYS use proper LaTeX for ALL mathematical expressions
- NEVER invent information not present in the results
- Keep scientific accuracy paramount
- ✅ **Your output MUST be LONGER than AstroSage input** (you're adding structure, not removing content)
- ✅ If AstroSage = 2000 words → Your output = 2200-2500 words (structure + formatting added)
- ✅ If your output is shorter than AstroSage input, you have FAILED

**WHAT YOU ADD (New content you create):**
- Executive summary (2-3 sentences at the top)
- Section headers and markdown structure
- Plot links with descriptions
- Minor LaTeX formatting corrections

**WHAT YOU PRESERVE (Content you must NOT touch):**
- ENTIRE AstroSage response content (every word, every paragraph)
- ALL mathematical derivations
- ALL physical explanations
- ALL model comparisons
- ALL conclusions and recommendations
- ALL literature references
"""

    # ✅ FIXED: Updated expertise guidelines with preservation focus
    EXPERTISE_GUIDELINES = {
        "beginner": """
**For Beginner Level:**
- ✅ PRESERVE ALL of AstroSage's accessible explanations
- Add clear section headers (###)
- Fix LaTeX formatting issues only
- Maintain encouraging, friendly tone
- ✅ Do NOT shorten or simplify AstroSage's explanations
- ✅ Do NOT remove any analogies or examples
- ✅ TARGET OUTPUT: Preserve 100% content + add structure (+10-20% length)
- ✅ AstroSage wrote for beginners - keep everything they wrote
""",
        "intermediate": """
**For Intermediate Level:**
- ✅ PRESERVE ALL of AstroSage's technical discussion
- Add clear section organization with headers
- Fix LaTeX formatting issues only
- Maintain informative, clear tone
- ✅ Do NOT summarize or condense technical details
- ✅ Do NOT remove any explanations or context
- ✅ TARGET OUTPUT: Preserve 100% content + add structure (+10-15% length)
""",
        "advanced": """
**For Advanced Level:**
- ✅ PRESERVE 100% of AstroSage's technical analysis
- ✅ KEEP ALL mathematical derivations complete
- Add professional document structure with clear sections
- Fix LaTeX formatting issues only
- ✅ Do NOT condense, simplify, or remove anything
- ✅ KEEP ALL literature references and detailed discussions intact
- ✅ MAINTAIN ALL parameter discussions and error analyses
- ✅ TARGET OUTPUT: Preserve 100% content + add structure (+10-15% length)
""",
        "expert": """
**For Expert Level:**
- ✅ PRESERVE 100% of AstroSage's research-level content
- ✅ MAINTAIN ALL mathematical rigor and derivations
- Add publication-quality document structure
- Fix LaTeX formatting issues only
- ✅ Do NOT compress, abbreviate, or remove anything
- ✅ KEEP ALL theoretical discussions complete
- ✅ MAINTAIN ALL methodological details
- ✅ TARGET OUTPUT: Preserve 100% content + add structure (+10-15% length)
- ✅ Maintain peer-review quality throughout
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
        parts.append("**AstroSage's Complete Response:**")
        parts.append("(You MUST preserve this ENTIRE response in your output)")
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

Create a well-formatted response that presents the analysis results:

1. **Executive Summary** (2-3 sentences)
   - Highlight key findings with actual numerical values

2. **Analysis Results**
   - Present numerical results clearly with proper LaTeX
   - Organize by analysis type (Statistics, PSD, Model Fitting)
   - Use proper section headers (###)

3. **Visualizations**
   - Format plot links with descriptions

**Remember**: Use actual values from the analysis results above.
""",
            
            "astrosage": """# YOUR TASK

Format the AstroSage interpretation with proper structure:

1. **Preserve 100% of AstroSage's content**
   - Every paragraph, every explanation
   - All mathematical equations
   - All physical interpretations

2. **Add Structure**
   - Clear section headers (###)
   - Fix any LaTeX formatting issues
   - Improve readability with markdown

**Remember**: This is a formatting task - do NOT summarize or shorten content.
""",
            
            "mixed": """# YOUR TASK

Create a comprehensive response that INTEGRATES both Analysis and AstroSage results:

**FORMATTING RULES (ABSOLUTELY CRITICAL):**
1. ✅ You are a FORMATTER, NOT a summarizer, editor, or content creator
2. ✅ PRESERVE 100% of AstroSage's response content
3. ✅ Keep ALL LaTeX equations EXACTLY as AstroSage wrote them
4. ✅ DO NOT shorten, paraphrase, condense, or simplify AstroSage's explanations
5. ✅ DO NOT remove ANY sections, paragraphs, or sentences from AstroSage
6. ✅ Your job: ADD structure (headers, formatting) and fix LaTeX syntax ONLY
7. ✅ Your output MUST be LONGER than the input (because you add structure)

**FORBIDDEN ACTIONS:**
❌ Summarizing or condensing AstroSage's content
❌ Removing any paragraphs or sections from AstroSage
❌ Paraphrasing or rewriting AstroSage's explanations
❌ Condensing or abbreviating mathematical derivations
❌ Shortening discussions, interpretations, or recommendations
❌ Removing literature references or comparisons
❌ Editing AstroSage's conclusions

**Response Structure:**

### 1. Executive Summary (NEW - write this yourself, 2-3 sentences)
   - Highlight the most important findings
   - Use actual numerical values from analysis results
   - Keep it concise and informative

### 2. Analysis Results (Format the numerical data)
   - Present analysis results clearly
   - Use proper LaTeX for ALL equations: $$\\text{PSD}(f) = \\frac{A}{f^b} + n$$
   - Use proper LaTeX for ALL parameters: $A = 2.67 \\times 10^3$
   - Organize by analysis type:
     * Statistics (if available)
     * PSD Analysis (if available)
     * Power Law Fit (if available)
     * Bending Power Law Fit (if available)

### 3. Physical Interpretation and Discussion
   ✅ **COPY THE ENTIRE AstroSage RESPONSE HERE**
   ✅ **DO NOT MODIFY, SUMMARIZE, OR SHORTEN**
   
   **What to preserve:**
   - Include EVERY section from AstroSage's response
   - Keep ALL section headers AstroSage created
   - Keep ALL mathematical derivations complete
   - Preserve ALL physical explanations (every paragraph)
   - Maintain ALL model comparisons (every detail)
   - Keep ALL conclusions and recommendations
   - Preserve ALL literature references and context
   - Keep ALL parameter discussions
   
   **What you CAN do:**
   - Add markdown section headers (###) if not present
   - Fix LaTeX syntax errors (e.g., missing $$ or $)
   - Add line breaks for better readability
   - Fix obvious typos (very rare with LLM output)
   
   **What you CANNOT do:**
   - Remove any content
   - Summarize any section
   - Paraphrase explanations
   - Shorten discussions
   - Condense derivations
   - Edit conclusions

### 4. Visualizations (NEW - format plot links)
   - List all generated plots with proper links
   - Add descriptive titles for each plot
   - Format as: [Plot Title](plot_url)

**VERIFICATION CHECKLIST (You MUST check ALL before responding):**
☑ Did I include EVERY paragraph from AstroSage's response?
☑ Did I keep EVERY mathematical equation from AstroSage intact?
☑ Did I preserve EVERY physical interpretation and explanation?
☑ Did I maintain EVERY section from AstroSage (Overview, Results, Interpretation, Comparison, Conclusions)?
☑ Is my output LONGER than the AstroSage input? (It should be +10-20% due to added structure)
☑ Did I only ADD headers and structure, not REMOVE or REWRITE content?
☑ Did I format all plot links properly?

**CRITICAL LENGTH CHECK:**
- Count the words in AstroSage's response: X words
- Your output should be: (X + 200-400) words minimum
- If your output < X words, you have FAILED (you removed content)
- If your output ≈ X words, you barely passed (you should add more structure)
- If your output > X words (+10-20%), you PASSED ✅

**EXAMPLE OF CORRECT APPROACH:**

**Input:**
- Analysis Results: 500 words
- AstroSage Response: 2000 words
- Total Input: 2500 words

**Your Output Should Be:**
- Executive Summary: 50-100 words (NEW)
- Analysis Results (formatted): 550-600 words (+10% from formatting)
- Physical Interpretation: 2000 words (EXACT copy from AstroSage)
- Visualizations: 50-100 words (NEW)
- **Total Output: 2650-2800 words** ✅

**NOT:**
- Total Output: 1800 words ❌ (You summarized - FAILED!)
- Total Output: 2200 words ❌ (You removed content - FAILED!)

**FINAL CRITICAL REMINDER:**
If your response is shorter than the AstroSage input, you have COMPLETELY FAILED the task. 
Your role is to PRESERVE and ADD STRUCTURE, not to edit, summarize, or reduce content.
Think of yourself as a formatter/typesetter, not an editor or writer.
"""
        }
        
        return instructions.get(
            routing_strategy,
            instructions["analysis"]  # default fallback
        )