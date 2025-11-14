#!/usr/bin/env python3
"""
multi-agent-fits-dev-02/app/agent/classification_parameter/unified_FITS_classification_parameter_agent.py

"""

import asyncio
import json
import logging
import re
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib

# LangChain imports
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI

from langchain.schema import HumanMessage, SystemMessage
from langchain_community.callbacks.manager import get_openai_callback

from app.services.conversation_history_service import (
    ConversationHistoryService,
    ConversationMessageDTO
)

from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class UnifiedFITSResult:
    """Unified result containing both classification and parameters"""
    # Classification fields
    primary_intent: str
    analysis_types: List[str] = field(default_factory=list)
    routing_strategy: str = "analysis"
    confidence: float = 0.0
    reasoning: str = ""
    
    # Mixed request handling
    is_mixed_request: bool = False
    question_context: Optional[str] = None
    astrosage_required: bool = False
    
    # Question categorization
    question_category: str = "unknown"
    complexity_level: str = "intermediate"
    
    # Parameter fields
    parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    parameter_confidence: Dict[str, float] = field(default_factory=dict)
    parameter_source: Dict[str, str] = field(default_factory=dict)
    
    # Workflow and guidance
    suggested_workflow: List[str] = field(default_factory=list)
    parameter_explanations: Dict[str, str] = field(default_factory=dict)
    potential_issues: List[str] = field(default_factory=list)
    
    # Metadata
    processing_time: float = 0.0
    tokens_used: int = 0
    cost_estimate: float = 0.0
    model_used: str = ""
    
    # Context enhancement
    requires_file: bool = False
    file_available: bool = False


class UnifiedFITSClassificationAgent:
    """
    Unified FITS Classification + Parameter Agent v2.2
    FIXED: Enhanced mixed request detection for astrophysics interpretation
    """
    
    def __init__(self, 
                 model_name: str = "gpt-3.5-turbo", 
                 temperature: float = 0.1,
                 max_tokens: int = 1500):
        self.name = "UnifiedFITSAgent_v2.2"
        self.logger = logging.getLogger(f"agent.{self.name}")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=30
        )
        
        # Parameter schemas
        self.parameter_schemas = self._build_parameter_schemas()
        
        # Build comprehensive system prompt with FIXED mixed request detection
        self.system_prompt = self._build_comprehensive_system_prompt()
        
        # Simple cache
        self.cache: Dict[str, UnifiedFITSResult] = {}
        self.cache_ttl = 1800  # 30 minutes
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "avg_processing_time": 0.0,
            "intent_distribution": {"analysis": 0, "general": 0, "mixed": 0},
            "question_categories": {"astronomy": 0, "physics": 0, "data_analysis": 0, "methods": 0, "unknown": 0},
            "analysis_type_distribution": {"statistics": 0, "psd": 0, "power_law": 0, "bending_power_law": 0, "metadata": 0},
            "parameter_extractions": 0,
            "auto_corrected_mixed": 0  # ✅ NEW: Track auto-corrections
        }
        
        self.logger.info(f"Unified FITS Agent v2.2 (FIXED) initialized with {model_name}")
    
    def _build_parameter_schemas(self) -> Dict[str, Dict]:
        """Build comprehensive parameter schemas"""
        return {
            "statistics": {
                "parameters": {
                    "metrics": {
                        "type": "list",
                        "default": ["mean", "median", "std", "min", "max", "count"],
                        "options": ["mean", "median", "std", "min", "max", "count", "percentile_25", "percentile_75", "percentile_90"],
                        "description": "Statistical metrics to calculate"
                    }
                },
                "defaults": {"metrics": ["mean", "median", "std", "min", "max", "count"]},
                "validation": {"metrics": "Must be subset of available metrics"}
            },
            
            "psd": {
                "parameters": {
                    "low_freq": {"type": "float", "default": 1e-5, "range": [1e-6, 0.1], "description": "Minimum frequency (Hz)"},
                    "high_freq": {"type": "float", "default": 0.05, "range": [0.001, 0.5], "description": "Maximum frequency (Hz)"},
                    "bins": {"type": "int", "default": 3500, "range": [100, 10000], "description": "Number of frequency bins"}
                },
                "defaults": {"low_freq": 1e-5, "high_freq": 0.05, "bins": 3500},
                "validation": {
                    "low_freq": "Must be > 0 and < high_freq",
                    "high_freq": "Must be > low_freq and <= 0.5",
                    "bins": "Must be integer between 100-10000"
                }
            },
            
            "power_law": {
                "parameters": {
                    "low_freq": {"type": "float", "default": 1e-5, "range": [1e-6, 0.1]},
                    "high_freq": {"type": "float", "default": 0.05, "range": [0.001, 0.5]},
                    "bins": {"type": "int", "default": 3500, "range": [100, 10000]},
                    "noise_bound_percent": {"type": "float", "default": 0.7, "range": [0.1, 1.0]},
                    "A0": {"type": "float", "default": 1.0, "range": [0.1, 100.0]},
                    "b0": {"type": "float", "default": 1.0, "range": [0.1, 3.0]},
                    "A_min": {"type": "float", "default": 0.0, "range": [0.0, 1e10]},
                    "A_max": {"type": "float", "default": 1e38, "range": [1.0, 1e38]},
                    "b_min": {"type": "float", "default": 0.1, "range": [0.01, 2.0]},
                    "b_max": {"type": "float", "default": 3.0, "range": [1.0, 5.0]},
                    "maxfev": {"type": "int", "default": 1000000, "range": [1000, 10000000]}
                },
                "defaults": {
                    "low_freq": 1e-5, "high_freq": 0.05, "bins": 3500,
                    "noise_bound_percent": 0.7, "A0": 1.0, "b0": 1.0,
                    "A_min": 0.0, "A_max": 1e38, "b_min": 0.1, "b_max": 3.0,
                    "maxfev": 1000000
                }
            },
            
            "bending_power_law": {
                "parameters": {
                    "low_freq": {"type": "float", "default": 1e-5, "range": [1e-6, 0.1]},
                    "high_freq": {"type": "float", "default": 0.05, "range": [0.001, 0.5]},
                    "bins": {"type": "int", "default": 3500, "range": [100, 10000]},
                    "noise_bound_percent": {"type": "float", "default": 0.7, "range": [0.1, 1.0]},
                    "A0": {"type": "float", "default": 10.0, "range": [0.1, 1000.0]},
                    "fb0": {"type": "float", "default": 0.01, "range": [1e-5, 0.1]},
                    "sh0": {"type": "float", "default": 1.0, "range": [0.1, 5.0]},
                    "A_min": {"type": "float", "default": 0.0, "range": [0.0, 1e10]},
                    "A_max": {"type": "float", "default": 1e38, "range": [1.0, 1e38]},
                    "fb_min": {"type": "float", "default": 2e-5, "range": [1e-6, 0.01]},
                    "fb_max": {"type": "float", "default": 0.05, "range": [0.001, 0.1]},
                    "sh_min": {"type": "float", "default": 0.3, "range": [0.1, 2.0]},
                    "sh_max": {"type": "float", "default": 3.0, "range": [1.0, 5.0]},
                    "maxfev": {"type": "int", "default": 1000000, "range": [1000, 10000000]}
                },
                "defaults": {
                    "low_freq": 1e-5, "high_freq": 0.05, "bins": 3500,
                    "noise_bound_percent": 0.7, "A0": 10.0, "fb0": 0.01, "sh0": 1.0,
                    "A_min": 0.0, "A_max": 1e38, "fb_min": 2e-5, "fb_max": 0.05,
                    "sh_min": 0.3, "sh_max": 3.0, "maxfev": 1000000
                }
            },

            "metadata": {
                "parameters": {},
                "defaults": {},
                "validation": {}
            }
        }
    
    def _build_comprehensive_system_prompt(self) -> str:
        """
        Build system prompt with FIXED mixed request detection for astrophysics interpretation
        ✅ CRITICAL FIX: Enhanced patterns for domain-specific interpretation requests
        """
        
        # Build parameter documentation
        param_docs = ""
        for analysis_type, schema in self.parameter_schemas.items():
            param_docs += f"\n{analysis_type.upper().replace('_', ' ')} PARAMETERS:\n"
            
            parameters = schema.get("parameters", {})
            defaults = schema.get("defaults", {})

            if analysis_type == "metadata":
                param_docs += " - No complex parameters needed\n"
            else:
                for param_name, param_info in parameters.items():
                    description = param_info.get('description', f'{param_name} parameter')
                    default_value = defaults.get(param_name, param_info.get('default', 'N/A'))
                    param_docs += f"  - {param_name}: {description} (default: {default_value})\n"
        
        return f"""You are an expert AI agent specializing in FITS file analysis for astrophysical data.

            CORE MISSION:
            Provide unified classification, question categorization, and parameter extraction for astronomical time-series analysis requests.

            🔬 DOMAIN EXPERTISE - XMM-NEWTON X-RAY ASTRONOMY:
            - FITS file analysis and astronomical data processing
            - XMM-Newton telescope observations of IRAS 13224-3809 (AGN)
            - Power Spectral Density (PSD) analysis for timing studies
            - Power law and bending power law model fitting
            - Accretion disk physics and magneto-rotational instability (MRI)
            - Black hole accretion regimes and spectral state transitions
            - Statistical analysis of lightcurve data

            TASK DESCRIPTION:
            Provide BOTH classification and parameter extraction in a single response:

            1. **INTENT CLASSIFICATION**: "analysis", "general", or "mixed"
            2. **QUESTION CATEGORIZATION**: "astronomy", "physics", "data_analysis", "methods", "unknown"
            3. **PARAMETER EXTRACTION**: Extract specific parameters or provide defaults

            CLASSIFICATION RULES:

            **ANALYSIS REQUESTS** (routing_strategy: "analysis"):
            - Direct data processing requests WITHOUT explanatory questions
            - Keywords: "calculate", "compute", "fit", "analyze", "get statistics", "show metadata"
            - Examples: "Calculate mean", "Fit power law", "Compute PSD"
            - Pure data processing

            **GENERAL QUESTIONS** (routing_strategy: "astrosage"):
            - Pure questions without data analysis requests
            - Keywords: "what", "why", "how", "explain", "tell me about"
            - Examples: "What is a neutron star?", "Explain accretion physics"
            - 🚨 CRITICAL: If NO analysis is requested (no compute/fit/calculate), it's "general" NOT "mixed"
            - Pure conceptual questions: "What is IRAS 13224-3809?", "Explain MRI turbulence"
            - No data processing involved

            🔴 CRITICAL RULE: "Explain [concept/process]" = ALWAYS general (astrosage)
            - "Explain magneto-rotational instability" → general
            - "Explain how MRI causes turbulence" → general  
            - "Explain accretion disk physics" → general
            - Even if mentions astrophysical objects/context → still general
            - ONLY mixed if user asks to "compute/fit/analyze data THEN explain"

            **MIXED REQUESTS** (routing_strategy: "mixed"):
            - 🚨 CRITICAL: Any request combining data analysis AND explanation/interpretation
            - 🚨 ASTROPHYSICS INTERPRETATION = ALWAYS MIXED

            **PATTERNS TO DETECT AS MIXED:**
            - "Do X and explain Y" → mixed
            - "Calculate X and tell me what Y means" → mixed
            - "Do X, then explain what results indicate" → mixed
            - "Do X and discuss how..." → mixed
            - "Fit X, then compare how..." → mixed
            - "Compute X and interpret in terms of..." → mixed
            - "Analyze X then relate to [physics]" → mixed

            **🔬 ASTROPHYSICS INTERPRETATION INDICATORS (ALWAYS MIXED):**
            When user request includes analysis + ANY of these phrases, ALWAYS classify as mixed:

            1. **Accretion Physics:**
            - "accretion regime/disk/flow/state"
            - "inner disk/outer disk dynamics"
            - "disk truncation/geometry"
            
            2. **Turbulence & Instabilities:**
            - "magneto-rotational/MRI turbulence"
            - "stochastic variability"
            - "turbulent fluctuations"
            
            3. **Spectral Analysis:**
            - "spectral state/transition/evolution"
            - "spectral indices/slopes"
            - "low-frequency/high-frequency behavior"
            
            4. **Physical Interpretation:**
            - "physical meaning/interpretation/significance"
            - "astrophysical implications/context"
            - "what does [result] tell us about [physics]"
            - "how does [result] relate to/reflect/correspond to [astrophysics]"
            - "discuss the physics behind..."
            - "compare in terms of [physical process]"

            → ALL ABOVE = routing_strategy: "mixed", astrosage_required: true

            **KEY MIXED REQUEST INDICATORS:**
            - "explain what [results] mean/indicate"
            - "interpret the [analysis results]"
            - "discuss how [results] relate to [physics]"
            - "compare how [results] correspond to [astrophysics]"
            - Analysis + interpretation in same request

            ANALYSIS TYPES:
            - **statistics**: Basic statistical analysis (mean, median, std, etc.)
            - **psd**: Power Spectral Density computation
            - **power_law**: Simple power law model fitting (PSD(f) = A/f^b + n)
            - **bending_power_law**: Bending power law model fitting (PSD(f) = A/[f(1+(f/fb)^(sh-1))] + n)
            - **metadata**: FITS file metadata extraction

            CRITICAL ANALYSIS TYPE DETECTION RULES:
            - "fit power law" → ["power_law"]
            - "fit bending power law" → ["bending_power_law"]
            - "fit both models" → ["power_law", "bending_power_law"]
            - "fit both power law and bending" → ["power_law", "bending_power_law"]
            - "fit PSD using both models" → ["power_law", "bending_power_law"] (NOT just "psd")
            - "timing analysis" → ["psd", "power_law"] (standard timing workflow)
            - "compute PSD" alone → ["psd"]
            - "show metadata/header/observation details" → ["metadata"]

            PARAMETER SCHEMAS:
            {param_docs}

            RESPONSE FORMAT:
            Always respond with this exact JSON structure:
            {{
                "classification": {{
                    "primary_intent": "analysis|general|mixed",
                    "analysis_types": ["statistics", "psd", "power_law", "bending_power_law", "metadata"],
                    "routing_strategy": "analysis|astrosage|mixed",
                    "confidence": 0.0-1.0,
                    "reasoning": "Clear explanation",
                    "is_mixed_request": true|false,
                    "question_context": "before_analysis|after_analysis|parallel|standalone",
                    "astrosage_required": true|false,
                    "question_category": "astronomy|physics|data_analysis|methods|unknown",
                    "complexity_level": "beginner|intermediate|advanced"
                }},

            **QUESTION CATEGORY DECISION RULES:**
            - "What is IRAS/AGN/black hole/neutron star" → astronomy
            - "Why is [object] important for studying [astronomy]" → astronomy  
            - "Explain MRI/turbulence/thermal-viscous instability" → physics
            - "How does [X] cause [Y]" (physics mechanism) → physics
            - "What is PSD/power law/statistics/fitting" → data_analysis
            - "Why is PSD useful for timing" → data_analysis
            - "Best approach for/How to choose [parameters]" → methods
            - "What do filters/corrections/quality mean" → methods
            - Interpretation of results in physics terms → physics
            - Questions about objects/phenomena → astronomy
                "parameters": {{
                    "psd": {{"low_freq": 1e-5, "high_freq": 0.05, "bins": 3500}},
                    "power_law": {{"A0": 1.0, "b0": 1.0}}
                }},
                "parameter_confidence": {{"psd": 0.9}},
                "parameter_source": {{"psd": "user_specified|defaults_used"}},
                "workflow": {{
                    "suggested_steps": ["compute_psd", "fit_models", "interpret_physics"],
                    "execution_pattern": "sequential",
                    "estimated_time": "2 minutes"
                }},
                "guidance": {{
                    "parameter_explanations": {{}},
                    "potential_issues": [],
                    "optimization_suggestions": []
                }}
            }}

            CRITICAL EXAMPLES - ASTROPHYSICS MIXED REQUESTS:

            Example 1 - YOUR EXACT USE CASE:
            Input: "Compute both Power Law and Bending Power Law fits with amplitude 2, 5000 bins, break frequency 0.02. Then, discuss how the spectral indices reflect the transition between accretion regimes."
            Classification: "mixed" (analysis + accretion physics interpretation)
            Routing: "mixed"
            Question Category: "astronomy"
            AstroSage Required: true
            Analysis Types: ["power_law", "bending_power_law"]
            Reasoning: "Request combines model fitting with interpretation of spectral behavior in terms of accretion disk physics"

            Example 2 - MRI TURBULENCE INTERPRETATION:
            Input: "Fit PSD using both models with 5000 bins and break at 0.02. Compare how spectral slopes correspond to magneto-rotational turbulence."
            Classification: "mixed" (fit + turbulence physics interpretation)
            Routing: "mixed"
            Question Category: "physics"
            AstroSage Required: true
            Reasoning: "Request combines PSD fitting with MRI turbulence interpretation"

            Example 3 - PURE ANALYSIS (NOT MIXED):
            Input: "Fit both power law and bending power law with 5000 bins and break frequency 0.02"
            Classification: "analysis" (pure data processing, NO interpretation requested)
            Routing: "analysis"
            Question Category: "unknown" (no question asked)
            AstroSage Required: false
            Reasoning: "Pure fitting request without interpretation"

            Example 4 - PURE QUESTION (NOT MIXED):
            Input: "What is IRAS 13224-3809 and why is it important for studying AGN?"
            Classification: "general" (pure question, NO analysis requested)
            Routing: "astrosage"
            Question Category: "astronomy"
            AstroSage Required: true
            Reasoning: "Pure astronomy question without data analysis"

            Example 5 - PURE QUESTION (NOT MIXED):
            Input: "Explain magneto-rotational instability and how it causes turbulence"
            Classification: "general" (pure question, NO analysis requested)
            Routing: "astrosage"
            Question Category: "physics"
            AstroSage Required: true
            Reasoning: "Pure physics concept question without data analysis - even though mentions accretion context"

            🔴 CRITICAL: Example 5 shows "Explain [physics]" = general, NOT mixed
            - Even if mentions astrophysical context (accretion disks, turbulence)
            - "Explain" without "compute/fit/analyze" = ALWAYS general
            - Do NOT classify as mixed just because of astrophysics terminology

            Example 6 - PURE QUESTION WITH CONTEXT (NOT MIXED):
            Input: "Explain how accretion disk physics works in AGN systems"
            Classification: "general" (pure explanation, NO analysis)
            Routing: "astrosage"
            Question Category: "astronomy"
            AstroSage Required: true
            Reasoning: "Conceptual explanation request - 'explain' without compute/fit/analyze = general"

            Example 7 - MIXED WITH DISK PHYSICS:
            Input: "Compute PSD then explain how frequency peaks relate to inner disk dynamics"
            Classification: "mixed" (PSD + disk physics interpretation)
            Routing: "mixed"
            Question Category: "astronomy"
            AstroSage Required: true

            QUALITY STANDARDS:
            - Precisely detect mixed requests with astrophysics interpretation
            - Correctly categorize questions for optimal routing
            - Extract parameters explicitly mentioned
            - Provide intelligent defaults
            - 🚨 NEVER miss mixed requests that include physical interpretation
            """
    
    async def process_request(self, 
                            user_input: str, 
                            context: Dict[str, Any] = None,
                            # history support
                            session = None,  # AsyncSession
                            session_id: Optional[str] = None) -> UnifiedFITSResult:
        """
        Main processing method with FIXED mixed request detection

        Main processing method with history support
    
        Args:
            user_input: User query
            context: Additional context
            session: Database session (for history loading)
            session_id: Session ID (for history loading)
        """

        if context is None:
            context = {}
        
        start_time = datetime.now()
        self.stats["total_requests"] += 1
        
        # Clean cache
        self._cleanup_cache()

        # Load conversation history if available
        history_context = {}
        if session and session_id:
            try:
                history_context = await self._load_history_context(
                    session=session,
                    session_id=session_id,
                    file_id=context.get("file_id")
                )
                
                # Enhance context with history
                context["has_history"] = history_context.get("has_history", False)
                context["last_parameters"] = history_context.get("last_parameters")
                
            except Exception as e:
                self.logger.warning(f"Failed to load history: {e}")
                history_context = {"has_history": False}
        
        # Check cache (include history in cache key)
        cache_key = self._generate_cache_key(user_input, context)
        if cache_key in self.cache:
            self.stats["cache_hits"] += 1
            cached_result = self.cache[cache_key]
            cached_result.processing_time = 0.001
            self.logger.info(f"Cache hit for: {user_input[:50]}...")
            return cached_result
        
        try:
            # Build prompt
            full_prompt = self._build_unified_prompt(user_input, context, history_context)
            
            # LLM call
            with get_openai_callback() as cb:
                messages = [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=full_prompt)
                ]
                
                response = await self.llm.agenerate([messages])
                raw_output = response.generations[0][0].text.strip()
                
                self.logger.debug(f"RAW OUTPUT:\n{raw_output[:500]}...")
                
                tokens_used = cb.total_tokens if hasattr(cb, 'total_tokens') else 0
                cost = cb.total_cost if hasattr(cb, 'total_cost') else tokens_used * 0.000002
                
                self.stats["total_tokens"] += tokens_used
                self.stats["total_cost"] += cost
            
            # Parse response
            result = self._parse_unified_response(raw_output)

            # Apply parameter inheritance from history
            if history_context.get("has_history"):
                result = await self._apply_parameter_inheritance(
                    result=result,
                    history_context=history_context,
                    user_input=user_input
                )
            
            # Post-processing detection of missed mixed requests
            result = self._detect_missed_mixed_requests(result, user_input)
            
            # Enhance with metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = processing_time
            result.tokens_used = tokens_used
            result.cost_estimate = cost
            result.model_used = self.llm.model_name
            
            # Apply context
            result = self._enhance_with_context(result, context)
            
            # Validate parameters
            result = self._validate_and_optimize_parameters(result)
            
            # Update stats
            self.stats["intent_distribution"][result.primary_intent] += 1
            self.stats["question_categories"][result.question_category] += 1
            for analysis_type in result.analysis_types:
                if analysis_type in self.stats["analysis_type_distribution"]:
                    self.stats["analysis_type_distribution"][analysis_type] += 1
            if result.parameters:
                self.stats["parameter_extractions"] += 1
            
            # Update average processing time
            total = self.stats["total_requests"]
            self.stats["avg_processing_time"] = (
                (self.stats["avg_processing_time"] * (total - 1) + processing_time) / total
            )
            
            # Cache result
            self.cache[cache_key] = result
            self.cache_timestamps[cache_key] = datetime.now()
            
            self.logger.info(f"Processing complete: {result.primary_intent} → {result.routing_strategy} "
                           f"(confidence: {result.confidence:.2f}, time: {processing_time:.3f}s)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}")
            return self._create_fallback_result(user_input, str(e))
    
    def _detect_missed_mixed_requests(self, result: UnifiedFITSResult, user_input: str) -> UnifiedFITSResult:
        """
        CRITICAL FIX: Post-processing to catch mixed requests that LLM misclassified
        
        This method validates LLM classification and corrects false negatives where:
        - User requests analysis + astrophysics interpretation
        - LLM incorrectly classified as pure "analysis"
        
        IMPORTANT: Only correct "analysis" to "mixed"
        - Do NOT change "general" to "mixed" (pure questions stay pure)
        """
        
        # Only check if LLM classified as pure analysis
        if result.primary_intent != "analysis" or result.is_mixed_request:
            return result  # Already mixed or general, no correction needed
        
        # CRITICAL: Must have analysis types to be mixed
        # If no analysis types, it's a pure question, not mixed
        if not result.analysis_types:
            return result  # No analysis = pure question, don't touch
        
        user_input_lower = user_input.lower()
        
        # Astrophysics interpretation keywords
        astro_interpretation_keywords = [
            # Accretion physics
            "accretion regime", "accretion disk", "accretion flow", "accretion state",
            "inner disk", "outer disk", "disk dynamics", "disk truncation",
            
            # Turbulence & instabilities  
            "magneto-rotational", "mri turbulence", "turbulence", "turbulent",
            "stochastic variability", "fluctuations",
            
            # Spectral analysis
            "spectral state", "spectral transition", "spectral evolution",
            "spectral indices", "spectral slopes", "spectral behavior",
            "low-frequency behavior", "high-frequency behavior",
            
            # Physical interpretation
            "physical interpretation", "physical meaning", "physical significance",
            "astrophysical implications", "astrophysical significance",
            "astrophysical context", "physics behind"
        ]
        
        # Interpretation verbs indicating mixed requests
        interpretation_verbs = [
            "discuss", "discuss how", "explain", "explain how", "explain what",
            "compare how", "compare in terms", "interpret", "interpret in terms",
            "tell us about", "tell us how", "relate to", "reflect", "correspond to",
            "indicate about", "show how", "demonstrate how"
        ]
        
        # Check for interpretation keywords
        has_astro_keywords = any(keyword in user_input_lower for keyword in astro_interpretation_keywords)
        
        # Check for interpretation verbs
        has_interpretation_verbs = any(verb in user_input_lower for verb in interpretation_verbs)
        
        # Check for "then" pattern (sequential interpretation)
        has_then_pattern = "then" in user_input_lower and has_interpretation_verbs
        
        # If we have analysis types AND (astrophysics keywords OR interpretation pattern)
        if result.analysis_types and (has_astro_keywords or has_then_pattern):
            self.logger.warning(f"🔧 AUTO-CORRECTING misclassified mixed request: {user_input[:80]}...")
            
            # Correct classification
            result.primary_intent = "mixed"
            result.routing_strategy = "mixed"
            result.is_mixed_request = True
            result.astrosage_required = True
            
            # Determine question category
            if any(kw in user_input_lower for kw in ["accretion", "black hole", "agn", "disk"]):
                result.question_category = "astronomy"
            elif any(kw in user_input_lower for kw in ["turbulence", "mri", "instability", "physics"]):
                result.question_category = "physics"
            elif any(kw in user_input_lower for kw in ["spectral", "frequency", "psd"]):
                result.question_category = "data_analysis"
            else:
                result.question_category = "astronomy"  # Default for astrophysics
            
            # Update reasoning
            detected_keywords = [kw for kw in astro_interpretation_keywords if kw in user_input_lower]
            detected_verbs = [verb for verb in interpretation_verbs if verb in user_input_lower]
            
            result.reasoning += (
                f" [✅ AUTO-CORRECTED: Detected astrophysics interpretation request. "
                f"Keywords: {detected_keywords[:3]}, Verbs: {detected_verbs[:2]}]"
            )
            
            # Track correction
            self.stats["auto_corrected_mixed"] += 1
            
            self.logger.info(f"✅ Corrected to MIXED: question_category={result.question_category}")
        
        return result
    
    def _build_unified_prompt(
        self, 
        user_input: str, 
        context: Dict[str, Any],
        history_context: Dict[str, Any] = None
    ) -> str:
        """Build comprehensive prompt with history"""
        
        # Build context section
        context_info = []
        if context.get("has_uploaded_files"):
            context_info.append("✅ User has uploaded FITS files")
        else:
            context_info.append("❌ No FITS files uploaded yet")
        
        if context.get("user_expertise"):
            expertise = context["user_expertise"]
            context_info.append(f"👤 User expertise: {expertise}")
        
        if context.get("previous_analyses"):
            prev = ", ".join(context["previous_analyses"])
            context_info.append(f"📊 Previous analyses: {prev}")

        # ✅ NEW: Add history information
        if history_context.get("has_history"):
            context_info.append("📝 Conversation history available")
            
            # Add last parameters info
            last_params = history_context.get("last_parameters")
            if last_params:
                param_types = [k for k in last_params.keys() if not k.startswith("_")]
                if param_types:
                    context_info.append(f"🔧 Last used parameters: {', '.join(param_types)}")
                    
                    # Show last parameter values (brief)
                    for ptype in param_types:
                        params = last_params[ptype]
                        key_params = {k: v for k, v in params.items() 
                                    if k in ["bins", "A0", "b0", "fb0", "low_freq", "high_freq"]}
                        if key_params:
                            context_info.append(f"   {ptype}: {key_params}")
        
        # return f"""
        #     UNIFIED FITS ANALYSIS REQUEST

        #     USER INPUT: "{user_input}"

        #     CONTEXT INFORMATION:
        #     {chr(10).join(context_info) if context_info else "No additional context"}

        #     TASK: Provide comprehensive classification, question categorization, and parameter extraction.

        #     🚨 CRITICAL: Pay special attention to astrophysics interpretation requests!
        #     - If user asks to "discuss", "explain", "compare" physical processes → MIXED
        #     - If user mentions accretion/turbulence/spectral physics → MIXED
        #     - Analysis + interpretation = ALWAYS MIXED

        #     RESPOND WITH COMPLETE JSON:
        #     """
        prompt = f"""
            UNIFIED FITS ANALYSIS REQUEST

            USER INPUT: "{user_input}"

            CONTEXT INFORMATION:
            {chr(10).join(context_info) if context_info else "No additional context"}

            TASK: Provide comprehensive classification, question categorization, and parameter extraction.

            🚨 HISTORY CONTEXT:
            """
        
        # ✅ NEW: Add formatted history if available
        if history_context.get("has_history"):
            prompt += """
                ✅ User has previous analyses in this session.
                - If user says "again", "same", "repeat", "last settings" → they want to reuse parameters
                - Only extract explicitly mentioned NEW parameter values
                - Mark inherited parameters appropriately
                """
        else:
            prompt += """
                ❌ No previous history available.
                - Extract all parameters from query or use defaults
                """
        
        prompt += """

                🚨 CRITICAL: Pay special attention to astrophysics interpretation requests!
                - If user asks to "discuss", "explain", "compare" physical processes → MIXED
                - If user mentions accretion/turbulence/spectral physics → MIXED
                - Analysis + interpretation = ALWAYS MIXED

                RESPOND WITH COMPLETE JSON:
                """
        
        return prompt
    
    def _parse_unified_response(self, raw_output: str) -> UnifiedFITSResult:
        """Parse LLM JSON response"""
        try:
            cleaned_output = self._extract_json_from_response(raw_output)
            response_data = json.loads(cleaned_output)
            
            classification = response_data.get("classification", {})
            parameters = response_data.get("parameters", {})
            param_confidence = response_data.get("parameter_confidence", {})
            param_source = response_data.get("parameter_source", {})
            workflow = response_data.get("workflow", {})
            guidance = response_data.get("guidance", {})
            context_data = response_data.get("context", {})
            
            # Validate question_category
            question_category = classification.get("question_category", "unknown")
            valid_categories = ["astronomy", "physics", "data_analysis", "methods", "unknown"]
            if question_category not in valid_categories:
                question_category = "unknown"
            
            # Validate complexity_level
            complexity_level = classification.get("complexity_level", "intermediate")
            valid_levels = ["beginner", "intermediate", "advanced"]
            if complexity_level not in valid_levels:
                complexity_level = "intermediate"
            
            result = UnifiedFITSResult(
                primary_intent=classification.get("primary_intent", "analysis"),
                analysis_types=classification.get("analysis_types", []),
                routing_strategy=classification.get("routing_strategy", "analysis"),
                confidence=float(classification.get("confidence", 0.7)),
                reasoning=classification.get("reasoning", "Unified classification"),
                
                is_mixed_request=classification.get("is_mixed_request", False),
                question_context=classification.get("question_context"),
                # Fix: astrosage_required should be true for general/mixed with questions
                astrosage_required=classification.get("astrosage_required", False),
                
                question_category=question_category,
                complexity_level=complexity_level,
                
                parameters=parameters,
                parameter_confidence=param_confidence,
                parameter_source=param_source,
                
                suggested_workflow=workflow.get("suggested_steps", []),
                parameter_explanations=guidance.get("parameter_explanations", {}),
                potential_issues=guidance.get("potential_issues", []),
                
                requires_file=context_data.get("requires_file", False)
            )
            
            # ✅ Post-processing: Ensure astrosage_required consistency
            if result.routing_strategy in ["astrosage", "mixed"] and not result.astrosage_required:
                result.astrosage_required = True
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to parse response: {str(e)}")
            self.logger.error(f"Raw output: {raw_output[:500]}...")
            return self._create_fallback_parsing_result(raw_output)
    
    def _extract_json_from_response(self, text: str) -> str:
        """Extract JSON from various formats"""
        text = text.strip()
        
        # JSON code blocks
        json_pattern = r'```json\s*(.*?)\s*```'
        json_match = re.search(json_pattern, text, re.DOTALL | re.IGNORECASE)
        if json_match:
            json_text = json_match.group(1).strip()
        else:
            # Find JSON object
            brace_pattern = r'\{.*\}'
            brace_match = re.search(brace_pattern, text, re.DOTALL)
            if brace_match:
                return brace_match.group(0).strip()
            else:
                json_text = text

        # Remove comments
        json_text = re.sub(r'//.*?(?=\n|$)', '', json_text)
        json_text = re.sub(r'/\*.*?\*/', '', json_text, flags=re.DOTALL)
        json_text = re.sub(r',\s*([\]}])', r'\1', json_text)

        return json_text
    
    def _enhance_with_context(self, result: UnifiedFITSResult, context: Dict[str, Any]) -> UnifiedFITSResult:
        """Enhance result with context"""
        
        if context.get("has_uploaded_files"):
            result.file_available = True
            if result.analysis_types:
                result.requires_file = True
        
        user_expertise = context.get("user_expertise", "intermediate")
        if user_expertise in ["beginner", "intermediate", "advanced"]:
            result.complexity_level = user_expertise
        
        if user_expertise == "beginner" and result.primary_intent == "analysis":
            if not result.is_mixed_request:
                result.astrosage_required = True
                result.suggested_workflow.append("provide_beginner_explanation")
        
        return result
    
    def _validate_and_optimize_parameters(self, result: UnifiedFITSResult) -> UnifiedFITSResult:
        """Validate and optimize parameters"""
        
        # Fill defaults for each analysis type
        for analysis_type in result.analysis_types:
            if analysis_type == "metadata":
                if analysis_type not in result.parameters:
                    result.parameters[analysis_type] = {}
                result.parameter_source[analysis_type] = "extract_all"
                result.parameter_confidence[analysis_type] = 1.0
            elif analysis_type in result.parameters:
                params = result.parameters[analysis_type]
                schema = self.parameter_schemas.get(analysis_type, {})
                defaults = schema.get("defaults", {})
                
                for param_name, default_value in defaults.items():
                    if param_name not in params:
                        params[param_name] = default_value
                        
                        if analysis_type not in result.parameter_source:
                            result.parameter_source[analysis_type] = "defaults_used"
                        elif result.parameter_source[analysis_type] == "user_specified":
                            result.parameter_source[analysis_type] = "mixed"
            else:
                schema = self.parameter_schemas.get(analysis_type, {})
                defaults = schema.get("defaults", {})
                if defaults:
                    result.parameters[analysis_type] = defaults.copy()
                    result.parameter_source[analysis_type] = "defaults_used"
                    result.parameter_confidence[analysis_type] = 0.8
        
        return result
    
    def _create_fallback_result(self, user_input: str, error: str) -> UnifiedFITSResult:
        """Create fallback result"""
        return UnifiedFITSResult(
            primary_intent="analysis",
            analysis_types=["statistics"],
            routing_strategy="analysis",
            confidence=0.3,
            reasoning=f"Fallback: {error}",
            question_category="unknown",
            complexity_level="intermediate",
            parameters={"statistics": self.parameter_schemas["statistics"]["defaults"]},
            parameter_confidence={"statistics": 0.3},
            parameter_source={"statistics": "defaults_used"},
            suggested_workflow=["check_input", "retry_request"],
            potential_issues=[f"processing_error: {error}"],
            requires_file=True
        )
    
    def _create_fallback_parsing_result(self, raw_output: str) -> UnifiedFITSResult:
        """Create result when parsing fails"""
        text_lower = raw_output.lower()
        
        if "mixed" in text_lower:
            primary_intent = "mixed"
            routing_strategy = "mixed"
            astrosage_required = True
        elif "general" in text_lower:
            primary_intent = "general"
            routing_strategy = "astrosage"
            astrosage_required = True
        else:
            primary_intent = "analysis"
            routing_strategy = "analysis"
            astrosage_required = False
        
        analysis_types = []
        if "statistics" in text_lower:
            analysis_types.append("statistics")
        if "psd" in text_lower:
            analysis_types.append("psd")
        if "power law" in text_lower:
            analysis_types.append("power_law")
        
        if not analysis_types and primary_intent == "analysis":
            analysis_types = ["statistics"]
        
        question_category = "unknown"
        if "accretion" in text_lower or "disk" in text_lower:
            question_category = "astronomy"
        elif "turbulence" in text_lower:
            question_category = "physics"
        
        parameters = {}
        parameter_confidence = {}
        parameter_source = {}
        for analysis_type in analysis_types:
            if analysis_type in self.parameter_schemas:
                parameters[analysis_type] = self.parameter_schemas[analysis_type]["defaults"]
                parameter_confidence[analysis_type] = 0.5
                parameter_source[analysis_type] = "defaults_used"
        
        return UnifiedFITSResult(
            primary_intent=primary_intent,
            analysis_types=analysis_types,
            routing_strategy=routing_strategy,
            confidence=0.6,
            reasoning="Fallback text analysis",
            question_category=question_category,
            complexity_level="intermediate",
            parameters=parameters,
            parameter_confidence=parameter_confidence,
            parameter_source=parameter_source,
            astrosage_required=astrosage_required,
            suggested_workflow=["verify_request"],
            potential_issues=["json_parsing_failed"],
            requires_file=(primary_intent == "analysis")
        )
    
    def _cleanup_cache(self):
        """Clean expired cache"""
        current_time = datetime.now()
        expired_keys = []
        
        for cache_key, timestamp in self.cache_timestamps.items():
            if current_time - timestamp > timedelta(seconds=self.cache_ttl):
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            self.cache.pop(key, None)
            self.cache_timestamps.pop(key, None)
    
    def _generate_cache_key(self, user_input: str, context: Dict[str, Any]) -> str:
        """Generate cache key"""
        normalized_input = re.sub(r'\s+', ' ', user_input.lower().strip())
        context_key = {
            "has_files": context.get("has_uploaded_files", False),
            "expertise": context.get("user_expertise", "intermediate"),
            "prev_analyses": sorted(context.get("previous_analyses", [])),
            # Include history in cache key
            "has_history": context.get("has_history", False),
            "last_params": bool(context.get("last_parameters"))
        }
        
        combined = f"{normalized_input}:{json.dumps(context_key, sort_keys=True)}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        total = max(self.stats["total_requests"], 1)
        
        return {
            "usage": {
                "total_requests": self.stats["total_requests"],
                "cache_hits": self.stats["cache_hits"],
                "cache_hit_rate": round(self.stats["cache_hits"] / total, 3),
                "auto_corrected_mixed": self.stats["auto_corrected_mixed"],
                "auto_correction_rate": round(self.stats["auto_corrected_mixed"] / total, 3),
                "total_tokens": self.stats["total_tokens"],
                "total_cost": round(self.stats["total_cost"], 4),
                "avg_processing_time": round(self.stats["avg_processing_time"], 3)
            },
            "classification_distribution": {
                "analysis": self.stats["intent_distribution"]["analysis"],
                "general": self.stats["intent_distribution"]["general"],
                "mixed": self.stats["intent_distribution"]["mixed"],
                "analysis_pct": round(self.stats["intent_distribution"]["analysis"] / total * 100, 1),
                "mixed_pct": round(self.stats["intent_distribution"]["mixed"] / total * 100, 1)
            },
            "question_categories": self.stats["question_categories"],
            "analysis_types": self.stats["analysis_type_distribution"]
        }
    
    def clear_cache(self):
        """Clear cache"""
        self.cache.clear()
        self.cache_timestamps.clear()
        self.logger.info("Cache cleared")

    async def _load_history_context(
        self,
        session,
        session_id: str,
        file_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load conversation history context.
        
        Returns:
            Dictionary with messages and last_parameters
        """
        
        try:
            # Load recent messages
            messages = await ConversationHistoryService.get_recent_messages(
                session=session,
                session_id=session_id,
                limit=5,  # Last 5 messages for context
                include_system=False,
                max_tokens=1000
            )
            
            # Load last parameters
            last_parameters = None
            
            if file_id:
                # Try file-specific first
                last_parameters = await ConversationHistoryService.get_last_parameters(
                    session=session,
                    session_id=session_id,
                    file_id=file_id,
                    scope="file",
                    search_depth=5
                )
            
            # Fallback to session-wide
            if not last_parameters:
                last_parameters = await ConversationHistoryService.get_last_parameters(
                    session=session,
                    session_id=session_id,
                    scope="session",
                    search_depth=5
                )
            
            return {
                "messages": messages,
                "last_parameters": last_parameters,
                "has_history": len(messages) > 0 or last_parameters is not None
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to load history context: {e}")
            return {
                "messages": [],
                "last_parameters": None,
                "has_history": False
            }
        
    async def _apply_parameter_inheritance(
        self,
        result: UnifiedFITSResult,
        history_context: Dict[str, Any],
        user_input: str
    ) -> UnifiedFITSResult:
        """
        Apply parameter inheritance from history when appropriate.
        
        Detects phrases like:
        - "again"
        - "same parameters"
        - "use last settings"
        - "same but bins=4000"
        """
        
        user_input_lower = user_input.lower()
        last_parameters = history_context.get("last_parameters")
        
        # Check for inheritance intent
        inheritance_keywords = [
            "again", "same", "last", "previous", "repeat",
            "use same", "same parameters", "last settings"
        ]
        
        should_inherit = any(keyword in user_input_lower for keyword in inheritance_keywords)
        
        if not should_inherit or not last_parameters:
            return result  # No inheritance needed
        
        self.logger.info(f"Applying parameter inheritance from history")
        
        # For each analysis type, inherit parameters
        for analysis_type in result.analysis_types:
            if analysis_type in last_parameters:
                # Get inherited parameters
                inherited_params = last_parameters[analysis_type].copy()
                
                # Remove metadata fields
                inherited_params.pop("_inherited", None)
                inherited_params.pop("_overridden_fields", None)
                
                # Get current parameters (explicit overrides)
                current_params = result.parameters.get(analysis_type, {})
                
                # Merge: start with inherited, apply overrides
                merged_params = inherited_params.copy()
                overridden_fields = []
                
                for key, value in current_params.items():
                    if key in merged_params and merged_params[key] != value:
                        overridden_fields.append(key)
                    merged_params[key] = value
                
                # Update result
                result.parameters[analysis_type] = merged_params
                result.parameter_source[analysis_type] = "inherited"
                
                # Track overrides
                if overridden_fields:
                    result.parameters[analysis_type]["_overridden_fields"] = overridden_fields
                    result.parameter_source[analysis_type] = "inherited_with_overrides"
                
                # Add metadata
                if "_metadata" in last_parameters:
                    result.parameters[analysis_type]["_inherited_from"] = {
                        "analysis_id": last_parameters["_metadata"].get("analysis_id"),
                        "timestamp": last_parameters["_metadata"].get("timestamp"),
                        "position": last_parameters["_metadata"].get("search_position")
                    }
                
                self.logger.info(
                    f"Inherited {analysis_type} parameters "
                    f"(overridden: {overridden_fields})"
                )
        
        # Update reasoning
        if result.reasoning:
            result.reasoning += " [Parameters inherited from previous analysis]"
        
        return result


# ============================================
# TEST CODE
# ============================================

async def test_agent():
    """Test the FIXED agent with your exact use cases"""
    
    agent = UnifiedFITSClassificationAgent(model_name="gpt-3.5-turbo")
    
    test_cases = [
        {
            "input": "Compute both Power Law and Bending Power Law fits for the PSD using an initial amplitude of 2, 5000 bins, and a break frequency of 0.02. Then, discuss how the low- and high-frequency spectral indices reflect the transition between different accretion regimes within the disk.",
            "expected": "mixed",
            "description": "Accretion regimes interpretation"
        },
        {
            "input": "Fit the PSD using both Power Law and Bending Power Law models with an initial amplitude of 2, 5000 bins, and break frequency at 0.02. Compare how the spectral slopes before and after the break correspond to stochastic variability or magneto-rotational turbulence in the accretion disk.",
            "expected": "mixed",
            "description": "MRI turbulence interpretation"
        },
        {
            "input": "Fit both power law and bending power law with 5000 bins and break frequency 0.02",
            "expected": "analysis",
            "description": "Pure analysis (no interpretation)"
        }
    ]
    
    print("\n" + "="*80)
    print("TESTING FIXED AGENT v2.2 - Enhanced Mixed Request Detection")
    print("="*80)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST CASE {i}: {test['description']}")
        print(f"{'='*80}")
        print(f"Input: {test['input'][:100]}...")
        print(f"Expected: {test['expected']}")
        
        result = await agent.process_request(test["input"])
        
        print(f"\n📊 RESULTS:")
        print(f"  Primary Intent: {result.primary_intent}")
        print(f"  Routing Strategy: {result.routing_strategy}")
        print(f"  Is Mixed: {result.is_mixed_request}")
        print(f"  AstroSage Required: {result.astrosage_required}")
        print(f"  Question Category: {result.question_category}")
        print(f"  Analysis Types: {result.analysis_types}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"\n  Reasoning: {result.reasoning}")
        
        # Check if result matches expected
        success = result.routing_strategy == test["expected"]
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"\n{status}: Got '{result.routing_strategy}', expected '{test['expected']}'")
        
        if result.parameters:
            print(f"\n  Parameters:")
            for analysis_type, params in result.parameters.items():
                print(f"    {analysis_type}: {params}")
    
    # Print statistics
    print(f"\n{'='*80}")
    print("AGENT STATISTICS")
    print("="*80)
    stats = agent.get_comprehensive_stats()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    asyncio.run(test_agent())