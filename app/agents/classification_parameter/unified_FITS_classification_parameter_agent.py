#!/usr/bin/env python3
"""
multi-agent-fits-dev-02/app/agent/classification_parameter/unified_FITS_classification_parameter_agent.py
=======================================
Unified FITS Classification + Parameter Agent - COMPLETE FIXED VERSION v2.1
* ORCHESTRATOR COMPATIBLE - Fixed missing question_category attribute
* Enhanced error handling and validation
* Improved LLM prompts for better classification
* Complete compatibility with FITSAnalysisOrchestrator

Single agent that handles both intent classification and parameter extraction
Perfect for MVP and single developer maintenance
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

from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class UnifiedFITSResult:
    """
    Unified result containing both classification and parameters
    ✅ ORCHESTRATOR COMPATIBLE - includes all required attributes
    """
    # Classification fields
    primary_intent: str                                           # "analysis", "general", "mixed"
    analysis_types: List[str] = field(default_factory=list)      # ["statistics", "psd", "fitting"]
    routing_strategy: str = "analysis"                           # "analysis", "astrosage", "mixed"
    confidence: float = 0.0
    reasoning: str = ""
    
    # Mixed request handling
    is_mixed_request: bool = False
    question_context: Optional[str] = None                       # "before_analysis", "after_analysis", "parallel", "standalone"
    astrosage_required: bool = False
    
    # ========================================
    # ✅ ORCHESTRATOR COMPATIBILITY FIX
    # ========================================
    question_category: str = "unknown"                           # "astronomy", "physics", "data_analysis", "methods", "unknown"
    complexity_level: str = "intermediate"                       # "beginner", "intermediate", "advanced"
    
    # Parameter fields for each analysis type
    parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # {"psd": {"low_freq": 1e-5}}
    parameter_confidence: Dict[str, float] = field(default_factory=dict)  # {"psd": 0.9}
    parameter_source: Dict[str, str] = field(default_factory=dict)       # {"psd": "user_specified"}
    
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
    Unified FITS Classification + Parameter Agent v2.1
    ORCHESTRATOR COMPATIBLE VERSION
    
    Single agent that handles both:
    1. Intent classification (analysis/general/mixed)
    2. Parameter extraction for FITS analysis
    3. Question categorization for AstroSage routing
    
    Perfect for MVP and single developer projects.
    """
    
    def __init__(self, 
                 model_name: str = "gpt-3.5-turbo", 
                 temperature: float = 0.1,
                 max_tokens: int = 1500):
        self.name = "UnifiedFITSAgent"
        self.logger = logging.getLogger(f"agent.{self.name}")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=30
        )
        
        # Parameter schemas for validation
        self.parameter_schemas = self._build_parameter_schemas()
        
        # Build comprehensive system prompt with question categorization
        self.system_prompt = self._build_comprehensive_system_prompt()
        
        # Simple cache for frequently used requests
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
            
            "intent_distribution": {
                "analysis": 0, 
                "general": 0, 
                "mixed": 0
            },
            
            "question_categories": {
                "astronomy": 0, 
                "physics": 0, 
                "data_analysis": 0, 
                "methods": 0, 
                "unknown": 0
            },
            
            "analysis_type_distribution": {
                "statistics": 0,
                "psd": 0,
                "power_law": 0,
                "bending_power_law": 0,
                "metadata": 0
            },
            
            "parameter_extractions": 0
        }
        
        self.logger.info(f"Unified FITS Agent v2.1 initialized with {model_name}")
    
    def _build_parameter_schemas(self) -> Dict[str, Dict]:
        """Build comprehensive parameter schemas for all analysis types"""
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
                "validation": {
                    "metrics": "Must be subset of available metrics"
                }
            },
            
            "psd": {
                "parameters": {
                    "low_freq": {
                        "type": "float",
                        "default": 1e-5,
                        "range": [1e-6, 0.1],
                        "description": "Minimum frequency for PSD computation (Hz)"
                    },
                    "high_freq": {
                        "type": "float", 
                        "default": 0.05,
                        "range": [0.001, 0.5],
                        "description": "Maximum frequency for PSD computation (Hz)"
                    },
                    "bins": {
                        "type": "int",
                        "default": 3500,
                        "range": [100, 10000],
                        "description": "Number of frequency bins for PSD"
                    }
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
                "parameters": {},  # ไม่มี parameters ที่ซับซ้อน
                "defaults": {},    # ไม่มี defaults
                "validation": {}   # ไม่มี validation rules
            }
        }
    
    def _build_comprehensive_system_prompt(self) -> str:
        """Build comprehensive system prompt for unified classification + parameters + question categorization"""
        """Build system prompt with enhanced shared parameter handling"""
        """Build system prompt enhanced with real XMM-Newton FITS data patterns"""

        # Build parameter documentation
        param_docs = ""
        for analysis_type, schema in self.parameter_schemas.items():
            param_docs += f"\n{analysis_type.upper().replace('_', ' ')} PARAMETERS:\n"
            
            parameters = schema.get("parameters", {})
            defaults = schema.get("defaults", {})

            if analysis_type == "metadata":
                param_docs += " - No complex parameters, needed\n"
                param_docs += " - Simple extraction approach: extract all available metadata\n"
            else:
                for param_name, param_info in parameters.items():
                    description = param_info.get('description', f'{param_name} parameter')
                    default_value = defaults.get(param_name, param_info.get('default', 'N/A'))
                    param_docs += f"  - {param_name}: {description} (default: {default_value})\n"

        # Enhanced shared parameter documentation
        shared_param_docs = """
            🔄 CRITICAL: COMPREHENSIVE SHARED PARAMETER HANDLING

            PARAMETER OVERLAP MATRIX:
            ┌─────────────────────┬─────────┬─────┬───────────┬─────────────┐
            │ Parameter           │ Stats   │ PSD │ Power Law │ Bending Law │
            ├─────────────────────┼─────────┼─────┼───────────┼─────────────┤
            │ low_freq            │   ❌    │ ✅  │    ✅     │     ✅      │  TRIPLE OVERLAP
            │ high_freq           │   ❌    │ ✅  │    ✅     │     ✅      │  TRIPLE OVERLAP  
            │ bins                │   ❌    │ ✅  │    ✅     │     ✅      │  TRIPLE OVERLAP
            │ A0                  │   ❌    │ ❌  │    ✅     │     ✅      │ 🔴 SPECIAL SHARED
            │ noise_bound_percent │   ❌    │ ❌  │    ✅     │     ✅      │ 🔄 FITTING SHARED
            │ A_min, A_max        │   ❌    │ ❌  │    ✅     │     ✅      │ 🔄 FITTING SHARED
            │ maxfev              │   ❌    │ ❌  │    ✅     │     ✅      │ 🔄 FITTING SHARED
            └─────────────────────┴─────────┴─────┴───────────┴─────────────┘

            🎯 SHARED PARAMETER RESOLUTION RULES:

            1. **TRIPLE OVERLAP PARAMETERS** (PSD + Both Fitting):
                - "frequency range X to Y" → low_freq: X, high_freq: Y in ALL THREE
                - "X bins" → bins: X in ALL THREE  
                - "bins from X to Y" → bins: X in ALL THREE (use first value)

            2. **FITTING-ONLY SHARED PARAMETERS** (Both Fitting Models):
                - "noise bound X%" → noise_bound_percent: X in BOTH fitting models
                - "amplitude bounds X to Y" → A_min: X, A_max: Y in BOTH fitting models
                - "max iterations X" → maxfev: X in BOTH fitting models

            3. **SPECIAL SHARED PARAMETER - A0**:
                - Different defaults: power_law=1.0, bending=10.0
                - "amplitude X" without model specification → apply to BOTH but explain
                - "power law amplitude X" → apply only to power law
                - "bending amplitude X" → apply only to bending power law
                - No amplitude specified → use model-specific defaults

            4. **CONFLICT RESOLUTION**:
                - Same parameter with different values → USE LAST MENTIONED
                - Ambiguous specification → USE CONTEXT or EXPLAIN
                - Model-specific override → APPLY TO SPECIFIC MODEL ONLY

            CRITICAL EXAMPLES:

            Example 1 - TRIPLE OVERLAP:
            Input: "Compute PSD and fit both power law models with frequency range 1e-4 to 0.1 Hz and 5000 bins"
            Response: Apply shared values to ALL THREE analysis types

            Example 2 - A0 SPECIAL HANDLING:
            Input: "Fit both models with amplitude 5.0"
            Response: Both models get A0=5.0, explain potential suboptimality

            Example 3 - CONFLICT RESOLUTION:
            Input: "Compute PSD with 4000 bins, then fit power law with 5000 bins"
            Response: Use 5000 bins for both (last mentioned), explain resolution

            GOLDEN RULE: "One specification, universal application" unless explicitly overridden
            """

        # Real metadata examples from actual FITS files
        real_metadata_examples = """
            REAL XMM-NEWTON FITS METADATA EXAMPLES (from actual data):
            
            🔭 OBSERVATORY CONTEXT (question_category: "astronomy"):
            "Show telescope and target" → TELESCOP="XMM", OBJECT="IRAS 13224-3809"
            "Who observed this?" → OBSERVER="Prof Andy Fabian"/"Prof Andrew Fabian"
            "What object is being studied?" → OBJECT="IRAS 13224-3809" (Active Galactic Nucleus)
            "Show observation ID" → OBS_ID="0673580401", "0792180101", etc.
            
            📊 DATA ANALYSIS CONTEXT (question_category: "data_analysis"):  
            "Show exposure time" → EXPOSURE=50261-121209 seconds (varies by observation)
            "What's the time resolution?" → TIMEDEL=1.0 seconds (1-second bins)
            "How many data points?" → NAXIS2=125008-137682 rows
            "Show data structure" → HDU info, EXTNAME="RATE", columns
            
            🔬 METHODS CONTEXT (question_category: "methods"):
            "What corrections were applied?" → BACKAPP=T, DEADAPP=T, VIGNAPP=T
            "What energy range?" → CHANMIN=300/1200/5000 eV (different energy bands)
            "What filter was used?" → FILTER="Thin1"
            "Show data processing" → CREATOR="epiclccorr", selection expressions
            
            ⚛️ PHYSICS CONTEXT (question_category: "physics"):
            "What instrument?" → INSTRUME="EPN" (European Photon Imaging Camera)
            "Energy band details?" → PI ranges, energy filters, detector specifics
            "Background ratio?" → BKGRATIO=6.2-6.3 (background to source area ratio)
        """
        
        # Enhanced metadata detection patterns
        metadata_detection_patterns = """
            METADATA REQUEST DETECTION PATTERNS:
            
            DEFINITIVE METADATA REQUESTS (analysis_types: ["metadata"]):
            - "show/display file info/header/metadata/structure"
            - "what's in this file/observation details/file details"
            - "show telescope/instrument/target/object/observer"
            - "display exposure/energy range/filter/corrections"
            - "file information/observation information"
            - "HDU structure/data structure/column information"
            
            XMM-NEWTON SPECIFIC PATTERNS:
            - "show observation ID/OBS_ID" → extract OBS_ID
            - "display exposure time/EXPOSURE" → extract EXPOSURE
            - "what corrections applied" → extract BACKAPP, DEADAPP, VIGNAPP
            - "energy range/band/CHANMIN" → extract CHANMIN, energy info
            - "who observed/OBSERVER" → extract OBSERVER
            - "target object/OBJECT" → extract OBJECT
        """
        
        # Real data-driven parameter suggestions
        parameter_optimization = """
            REAL DATA-DRIVEN PARAMETER OPTIMIZATION:
            
            Based on actual XMM-Newton observations:
            - Exposure times: 50,000-120,000 seconds (14-33 hours)
            - Time resolution: 1.0 second bins (TIMEDEL=1.0)
            - Data points: 125,000-140,000 per observation
            - Nyquist frequency: 0.5 Hz (from 1-second sampling)
            - Minimum useful frequency: ~1e-5 Hz (from long exposures)
            
            OPTIMAL PSD PARAMETERS:
            - low_freq: 1e-5 to 2e-5 Hz (at least 2-3 cycles in exposure)
            - high_freq: 0.05-0.1 Hz (well below Nyquist frequency)
            - bins: 3500-5000 (good frequency resolution)
        """

        
        return f"""You are an expert AI agent specializing in FITS (Flexible Image Transport System) file analysis for astrophysical data.

            CORE MISSION:
            Provide unified classification, question categorization, and parameter extraction for astronomical time-series analysis requests.

            🔬 DOMAIN EXPERTISE - XMM-NEWTON X-RAY ASTRONOMY:
            - FITS file analysis and astronomical data processing
            - XMM-Newton telescope observations of IRAS 13224-3809 (AGN)
            - Power Spectral Density (PSD) analysis for timing studies
            - Power law and bending power law model fitting
            - Statistical analysis of lightcurve data
            - X-ray astronomy, pulsar observations, neutron star studies
            - Prof Andy Fabian's black hole accretion research
            - EPN instrument light curve data analysis

            {real_metadata_examples}

            {metadata_detection_patterns}

            TASK DESCRIPTION:
            You must provide BOTH classification and parameter extraction in a single response:

            1. **INTENT CLASSIFICATION**: Determine if the request is:
                - "analysis": Direct data analysis request
                - "general": Pure astronomy/physics questions
                - "mixed": Combination of questions and analysis

            2. **QUESTION CATEGORIZATION**: For general/mixed requests, categorize the question:
                - "astronomy": Celestial objects, stellar evolution, cosmic phenomena, IRAS 13224-3809, AGN, black holes
                - "physics": Physical processes, laws, fundamental concepts
                - "data_analysis": Analysis methods, statistical concepts, PSD, fitting
                - "methods": Techniques, procedures, best practices
                - "unknown": Cannot categorize or mixed categories

            3. **PARAMETER EXTRACTION**: For analysis requests, extract specific parameters or provide intelligent defaults

            CLASSIFICATION RULES:

            **ANALYSIS REQUESTS** (routing_strategy: "analysis"):
            - Direct requests to analyze user's data WITHOUT explanatory questions
            - Keywords: "calculate", "compute", "fit", "analyze", "get statistics", "show metadata"
            - Examples: "Calculate mean and std", "Fit power law", "Compute PSD", "Show file info"
            - Pure data processing requests

            **GENERAL QUESTIONS** (routing_strategy: "astrosage"):
            - Pure astronomy/physics questions without data analysis
            - Keywords: "what", "why", "how", "explain", "tell me about"
            - Examples: "What is a neutron star?", "Explain stellar evolution"
            - Examples: "What is IRAS 13224-3809?", "Explain AGN variability"
            - No data processing involved

            **MIXED REQUESTS** (routing_strategy: "mixed"):
            - CRITICAL: Any request that combines data analysis AND explanation/interpretation
            - Patterns to detect as MIXED:
            * "Do X and explain Y" → mixed
            * "Show metadata and explain what TELESCOP means" → mixed
            * "Calculate X and tell me what Y means" → mixed  
            * "What is X? Then do Y" → mixed
            * "I'm new to... do X and explain..." → mixed
            * "Do X, then explain what the results indicate" → mixed
            * Any analysis request that asks for interpretation or explanation
            - Examples: 
            * "What is PSD? Then compute it" → mixed, question_category: "data_analysis"
            * "Calculate statistics and explain what they mean" → mixed, question_category: "data_analysis"
            * "Compute PSD and explain what frequency peaks indicate" → mixed, question_category: "astronomy"
            * "I'm new to astronomy. Calculate statistics and explain them" → mixed, question_category: "astronomy"
            * "Show observation details and explain what IRAS 13224-3809 is" → mixed, question_category: "astronomy"
            * "Display metadata and explain XMM-Newton mission" → mixed, question_category: "astronomy"
            * "Calculate statistics and explain what they mean" → mixed, question_category: "data_analysis"

            **KEY MIXED REQUEST INDICATORS:**
            - "explain what [results] mean/indicate"
            - "tell me what [results] show"
            - "interpret the [analysis results]"
            - "what do [analysis results] indicate about..."
            - "I'm new to..." followed by analysis request
            - Analysis + interpretation in same request

            ANALYSIS TYPES:
            - **statistics**: Basic statistical analysis (mean, median, std, etc.)
            - **psd**: Power Spectral Density computation
            - **fitting_power_law**: Simple power law model fitting
            - **fitting_bending_power_law**: Bending power law model fitting
            - **metadata**: FITS file metadata extraction and examination
    
            METADATA ANALYSIS - ENHANCED DETECTION:
            - "show header/metadata/file info" → metadata
            - "display observation details" → metadata
            - "what's in the FITS file" → metadata
            - "examine structure/HDU info" → metadata
            - "telescope/instrument/target info" → metadata
            - "observation ID/exposure time" → metadata
            
            METADATA QUESTION CLASSIFICATION:
            - Questions about IRAS 13224-3809, observer, telescope → question_category: "astronomy"
            - Questions about exposure, data points, structure → question_category: "data_analysis"
            - Questions about corrections, filters, processing → question_category: "methods"
            - Questions about EPN instrument, energy bands → question_category: "physics"

            PARAMETER SCHEMAS:
            {param_docs}

            {shared_param_docs}

            RESPONSE FORMAT:
            [JSON format same as before...]

            CRITICAL EXAMPLES - REAL XMM-NEWTON DATA CONTEXT:
            [Examples same as before...]

            QUESTION CATEGORIZATION EXAMPLES:
            - "What is a neutron star?" → question_category: "astronomy"
            - "Explain stellar evolution" → question_category: "astronomy" 
            - "How do black holes form?" → question_category: "astronomy"
            - "What causes X-ray emission?" → question_category: "physics"
            - "Explain quantum mechanics" → question_category: "physics"
            - "What is power spectral density?" → question_category: "data_analysis"
            - "How do I interpret PSD results?" → question_category: "data_analysis"
            - "What's the best way to fit models?" → question_category: "methods"
            - "How should I analyze timing data?" → question_category: "methods"

            RESPONSE FORMAT:
            CRITICAL: Respond with VALID JSON ONLY. Do NOT include:
            - Javascript-style comments (// or /* */)
            - Trailing commas in lists or objects
            - Any text outside the JSON structure

            RESPONSE FORMAT:
            Always respond with this exact JSON structure:
            {{
                "classification": {{
                    "primary_intent": "analysis|general|mixed",
                    "analysis_types": ["statistics", "psd", "power_law", "bending_power_law", "metadata"],
                    "routing_strategy": "analysis|astrosage|mixed",
                    "confidence": 0.0-1.0,
                    "reasoning": "Clear explanation of classification decision",
                    "is_mixed_request": true|false,
                    "question_context": "before_analysis|after_analysis|parallel|standalone",
                    "astrosage_required": true|false,
                    "question_category": "astronomy|physics|data_analysis|methods|unknown",
                    "complexity_level": "beginner|intermediate|advanced"
                }},
                "parameters": {{
                    "metadata": {{}},
                    "statistics": {{"metrics": ["mean", "std"]}},
                    "psd": {{"low_freq": 1e-5, "high_freq": 0.05, "bins": 3500}},
                    "power_law": {{"A0": 1.0, "b0": 1.0, "noise_bound_percent": 0.7}},
                    "bending_power_law": {{"A0": 10.0, "fb0": 0.01, "sh0": 1.0}}
                }},
                "parameter_confidence": {{
                    "metadata": 1.0,
                    "statistics": 0.9,
                    "psd": 0.8,
                    "power_law": 0.7
                }},
                "parameter_source": {{
                    "metadata": "extract_all",
                    "statistics": "user_specified|defaults_used|inferred",
                    "psd": "user_specified|defaults_used|inferred"
                }},
                "workflow": {{
                    "suggested_steps": ["load_fits", "extract_metadata", "compute_psd", "fit_model"],
                    "execution_pattern": "single|sequential|parallel",
                    "estimated_time": "30 seconds"
                }},
                "guidance": {{
                    "parameter_explanations": {{"bins": "Using 3500 bins for good frequency resolution"}},
                    "potential_issues": ["check_data_quality", "verify_frequency_range"],
                    "optimization_suggestions": ["consider_higher_bins_for_better_resolution"]
                }},
                "context": {{
                    "requires_file": true|false,
                    "complexity_level": "basic|intermediate|advanced",
                    "expertise_adaptation": "beginner|intermediate|expert"
                }}
            }}

            CRITICAL EXAMPLES:

            Example 1 - MIXED REQUEST:
            Input: "I'm new to X-ray astronomy. Calculate basic statistics for my neutron star data and explain what they mean"
            Classification: "mixed" (has both analysis AND explanation request)
            Routing: "mixed"
            Question Category: "astronomy"
            AstroSage Required: true

            Example 2 - MIXED REQUEST:
            Input: "Calculate statistics and compute PSD, then explain what the frequency peaks might indicate about the neutron star's rotation"
            Classification: "mixed" (analysis + interpretation request)
            Routing: "mixed"
            Question Category: "astronomy"
            AstroSage Required: true

            Example 3 - PURE ANALYSIS:
            Input: "Perform complete timing analysis: compute PSD with 4000 bins, then fit both power law and bending power law models"
            Classification: "analysis" (pure data processing, no explanation requested)
            Routing: "analysis"
            Question Category: "unknown" (no question asked)
            AstroSage Required: false

            Example 4 - PURE GENERAL:
            Input: "What causes neutron stars to emit X-rays?"
            Classification: "general" (pure question, no analysis)
            Routing: "astrosage"
            Question Category: "astronomy"
            AstroSage Required: true

            Example 5 - DATA ANALYSIS QUESTION:
            Input: "What is power spectral density and how do I interpret the results?"
            Classification: "general" (pure question)
            Routing: "astrosage"
            Question Category: "data_analysis"
            AstroSage Required: true

            Example 6 - METADATA REQUEST:
            Input: "Show me the telescope, target object, and exposure time"
            Classification: "analysis" (pure metadata extraction)
            Analysis Types: ["metadata"]
            Routing: "analysis"
            Question Category: "astronomy" (about observation context)
            AstroSage Required: false (pure technical extraction)

            Example 7 - MIXED REQUEST:
            Input: "Display observation details and explain what IRAS 13224-3809 is"
            Classification: "mixed" (metadata + explanation)
            Analysis Types: ["metadata"]
            Routing: "mixed"
            Question Category: "astronomy" (about AGN object)
            AstroSage Required: true

            Example 8 - PURE ANALYSIS:
            Input: "Compute PSD with 4000 bins for this XMM observation"
            Classification: "analysis" (pure data processing)
            Analysis Types: ["psd"]
            Routing: "analysis"
            Question Category: "unknown" (no question asked)
            AstroSage Required: false

            PARAMETER EXTRACTION RULES:
            - "shape parameter X" → sh0: X (for bending power law)
            - "amplitude bounds from X to Y" → A_min: X, A_max: Y
            - "power index X" → b0: X (for power law)
            - "break frequency X" → fb0: X (for bending power law)
            - "frequency range X to Y" → low_freq: X, high_freq: Y
            - "X bins" → bins: X
            - Pay careful attention to parameter names and their correct mapping

            QUALITY STANDARDS:
            - Be precise with analysis type detection
            - Correctly categorize questions for optimal AstroSage routing
            - Only extract parameters explicitly mentioned or strongly implied
            - Provide intelligent defaults for missing parameters
            - Consider user expertise level for parameter suggestions
            - Include workflow guidance for complex analyses
            - Identify potential issues proactively
            - CORRECTLY identify mixed requests that need both analysis AND explanation
            - Correctly identify metadata requests vs other analysis types
            - Use real XMM-Newton context for question categorization
            - Apply proper parameter source ("extract_all" for metadata)
            - Consider IRAS 13224-3809 AGN context for astronomy questions
            - Provide data-driven parameter recommendations
            
            METADATA ANALYSIS - SIMPLE APPROACH:
            - metadata requests always extract complete file information
            - No complex parameter filtering needed
            - AstroSage will handle specific information filtering
            - Focus on intent detection rather than parameter extraction

            METADATA EXAMPLES:
            - "Show file header" → analysis, metadata, parameters: {{}}
            - "Display observation info" → analysis, metadata, parameters: {{}}
            - "What's in this FITS file?" → analysis, metadata, parameters: {{}}"""

    def _build_parameter_docs(self) -> str:
        """Build parameter documentation with real data context"""
        param_docs = ""
        for analysis_type, schema in self.parameter_schemas.items():
            param_docs += f"\n{analysis_type.upper().replace('_', ' ')} PARAMETERS:\n"
            
            parameters = schema.get("parameters", {})
            defaults = schema.get("defaults", {})
            
            if analysis_type == "metadata":
                param_docs += "  - No complex parameters needed\n"
                param_docs += "  - Simple extraction approach: extract all available metadata\n"
                param_docs += "  - AstroSage handles specific filtering and interpretation\n"
            else:
                for param_name, param_info in parameters.items():
                    description = param_info.get('description', f'{param_name} parameter')
                    default_value = defaults.get(param_name, param_info.get('default', 'N/A'))
                    param_docs += f"  - {param_name}: {description} (default: {default_value})\n"
        
        return param_docs
                
    async def process_request(self, 
                            user_input: str, 
                            context: Dict[str, Any] = None) -> UnifiedFITSResult:
        """
        Main method: Unified classification and parameter extraction with question categorization
        ✅ ORCHESTRATOR COMPATIBLE
        
        Args:
            user_input: User's natural language request
            context: Additional context (files, expertise, etc.)
            
        Returns:
            UnifiedFITSResult with classification, question categorization, and parameters
        """
        if context is None:
            context = {}
        
        start_time = datetime.now()
        self.stats["total_requests"] += 1
        
        # Clean up expired cache entries
        self._cleanup_cache()
        
        # Check cache first
        cache_key = self._generate_cache_key(user_input, context)
        if cache_key in self.cache:
            self.stats["cache_hits"] += 1
            cached_result = self.cache[cache_key]
            cached_result.processing_time = 0.001  # Almost instant
            self.logger.info(f"Cache hit for request: {user_input[:50]}...")
            return cached_result
        
        try:
            # Build comprehensive prompt
            full_prompt = self._build_unified_prompt(user_input, context)
            
            # Single LLM call for everything
            with get_openai_callback() as cb:
                messages = [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=full_prompt)
                ]
                
                response = await self.llm.agenerate([messages])
                raw_output = response.generations[0][0].text.strip()
                print(f"\n=== RAW API OUTPUT ===")
                print(raw_output)
                print(f"=== END RAW OUTPUT ===\n")
                
                # Track usage
                tokens_used = cb.total_tokens if hasattr(cb, 'total_tokens') else 0
                cost = cb.total_cost if hasattr(cb, 'total_cost') else tokens_used * 0.000002
                
                self.stats["total_tokens"] += tokens_used
                self.stats["total_cost"] += cost
            
            # Parse unified response with question categorization
            result = self._parse_unified_response(raw_output)
            
            # Enhance with metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = processing_time
            result.tokens_used = tokens_used
            result.cost_estimate = cost
            result.model_used = self.llm.model_name
            
            # Apply context enhancements
            result = self._enhance_with_context(result, context)
            
            # Validate parameters
            result = self._validate_and_optimize_parameters(result)
            
            # Update statistics
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
            
            # Cache the result
            self.cache[cache_key] = result
            self.cache_timestamps[cache_key] = datetime.now()
            
            self.logger.info(f"Unified processing completed: {result.primary_intent} → {result.analysis_types} "
                           f"(confidence: {result.confidence:.2f}, time: {processing_time:.3f}s)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Unified processing failed: {str(e)}")
            return self._create_fallback_result(user_input, str(e))
    
    def _build_unified_prompt(self, user_input: str, context: Dict[str, Any]) -> str:
        """Build comprehensive prompt for unified processing with question categorization"""
        """
        Method นี้เป็นตัวสร้าง prompt สำหรับส่งไปยัง LLM โดยรวม user input และ context เข้าด้วยกันให้เป็น structured prompt

        Args:
            user_input: ข้อความที่ผู้ใช้ป้อนเข้ามา
            context: ข้อมูลบริบทเพิ่มเติม เช่น การอัปโหลดไฟล์ (has_uploaded_files), ระดับความเชี่ยวชาญของผู้ใช้ (user_expertise), การวิเคราะห์ก่อนหน้า (previous_analyses) เป็นต้น 
        Returns:
            full_prompt: ข้อความ prompt ที่รวม user input และ context แล้ว
        1. รวม user input เข้ากับ context ที่มีอยู่
        2. สร้างส่วนของ context ที่แสดงสถานะการอัปโหลดไฟล์และข้อมูลผู้ใช้
        3. รวมคำแนะนำสำหรับการวิเคราะห์, การจัดหมวดหมู่คำถาม, และการดึงพารามิเตอร์
        """

        """
        Context Information Mapping
        1. File Status Mapping:
            # Different file states:
            {"has_uploaded_files": True}   → "✅ User has uploaded FITS files"
            {"has_uploaded_files": False}  → "❌ No FITS files uploaded yet"
            {}                             → "❌ No FITS files uploaded yet"

        2. User Expertise Mapping:
        {"user_expertise": "beginner"}     → "👤 User expertise: beginner"
        {"user_expertise": "intermediate"} → "👤 User expertise: intermediate"
        {"user_expertise": "advanced"}     → "👤 User expertise: advanced"
        {"user_expertise": "expert"}       → "👤 User expertise: expert"
        {}                                 → (no expertise line added)

        3. Previous Analyses Mapping:
        {"previous_analyses": ["psd"]}                    → "📊 Previous analyses: psd"
        {"previous_analyses": ["stats", "psd", "fit"]}   → "📊 Previous analyses: stats, psd, fit"
        {"previous_analyses": []}                        → (no history line added)
        {}                                               → (no history line added)

        =============================================================================

        Example Full Prompt Structure:

        Example 1: Beginner with no files

            # Input:
            user_input = "What is power spectral density?"
            context = {"has_uploaded_files": False, "user_expertise": "beginner"}

            # Output prompt:
            
            UNIFIED FITS ANALYSIS REQUEST

            USER INPUT: "What is power spectral density?"

            CONTEXT INFORMATION:
            ❌ No FITS files uploaded yet
            👤 User expertise: beginner

            TASK: Provide comprehensive classification...
            [rest of template]
            
        Example 2: Expert with complex request

            # Input:
            user_input = "Fit both power law and bending power law with custom parameters"
            context = {
                "has_uploaded_files": True,
                "user_expertise": "expert",
                "previous_analyses": ["statistics", "psd"]
            }

            # Output prompt:
            
            UNIFIED FITS ANALYSIS REQUEST

            USER INPUT: "Fit both power law and bending power law with custom parameters"

            CONTEXT INFORMATION:
            ✅ User has uploaded FITS files
            👤 User expertise: expert
            📊 Previous analyses: statistics, psd

            TASK: Provide comprehensive classification...
            [rest of template]
            
        Example 3: No context provided

            # Input:
            user_input = "Calculate mean and std"
            context = {}

            # Output prompt:
            
            UNIFIED FITS ANALYSIS REQUEST

            USER INPUT: "Calculate mean and std"

            CONTEXT INFORMATION:
            No additional context

            TASK: Provide comprehensive classification...
            [rest of template]

        """
        
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
        
        return f"""
            UNIFIED FITS ANALYSIS REQUEST

            USER INPUT: "{user_input}"

            CONTEXT INFORMATION:
            {chr(10).join(context_info) if context_info else "No additional context"}

            TASK: Provide comprehensive classification, question categorization, and parameter extraction in single response.

            ANALYSIS GUIDELINES:
            1. **Classification**: Determine primary intent and required analysis types
            2. **Question Categorization**: Classify questions for optimal AstroSage routing
            3. **Parameters**: Extract specific values mentioned or provide intelligent defaults
            4. **Workflow**: Suggest optimal analysis sequence
            5. **Guidance**: Provide helpful explanations and potential issue warnings

            PARAMETER EXTRACTION RULES:
            - Only extract parameters explicitly mentioned or strongly implied
            - Use domain expertise to suggest optimal defaults
            - Consider user expertise level for complexity
            - Validate parameter ranges and relationships
            - Provide explanations for non-obvious choices

            RESPOND WITH COMPLETE JSON (no truncation):
            """
    
    def _parse_unified_response(self, raw_output: str) -> UnifiedFITSResult:
        """Parse LLM JSON response into UnifiedFITSResult with question categorization"""

        """
        Method นี้รับ raw JSON string จาก LLM และแปลงเป็น UnifiedFITSResult object พร้อมจัดการ errors และ validation
        
        inputs:
            raw_output = "
                "classification": {
                    "primary_intent": "analysis",
                    "analysis_types": ["statistics", "psd"],
                    "routing_strategy": "analysis",
                    "confidence": 0.9,
                    "question_category": "unknown"
                },
                "parameters": {
                    "statistics": {"metrics": ["mean", "std"]},
                    "psd": {"low_freq": 1e-5, "bins": 3500}
                }
                }"   

        outputs:
            UnifiedFITSResult object with all fields populated, including defaults and validations
        1. Clean and extract JSON from raw output
        2. Parse JSON into dictionary
        3. Extract classification, question categorization, parameters, workflow, guidance, and context
        4. Validate and normalize fields (e.g., question_category, complexity_level)
        5. Populate UnifiedFITSResult dataclass
        6. Handle parsing errors gracefully with fallback

        EXAMPLE INPUTS/OUTPUTS:
        
                UnifiedFITSResult(
            primary_intent="analysis",
            analysis_types=["statistics", "psd"],
            question_category="unknown",
            parameters={"statistics": {...}, "psd": {...}},
            # ... other fields
)
        
        """
        try:
            # Clean and extract JSON
            cleaned_output = self._extract_json_from_response(raw_output)
            response_data = json.loads(cleaned_output)
            
            # Extract classification data
            classification = response_data.get("classification", {})
            parameters = response_data.get("parameters", {})
            param_confidence = response_data.get("parameter_confidence", {})
            param_source = response_data.get("parameter_source", {})
            workflow = response_data.get("workflow", {})
            guidance = response_data.get("guidance", {})
            context_data = response_data.get("context", {})
            
            # ===== ORCHESTRATOR COMPATIBILITY =====
            # Extract and validate question_category
            question_category = classification.get("question_category", "unknown")
            valid_categories = ["astronomy", "physics", "data_analysis", "methods", "unknown"]
            if question_category not in valid_categories:
                question_category = "unknown"
            
            # Extract and validate complexity_level
            complexity_level = classification.get("complexity_level", "intermediate")
            valid_levels = ["beginner", "intermediate", "advanced"]
            if complexity_level not in valid_levels:
                complexity_level = "intermediate"
            
            # Create unified result with all required attributes
            result = UnifiedFITSResult(
                # Classification fields
                primary_intent=classification.get("primary_intent", "analysis"),
                analysis_types=classification.get("analysis_types", []),
                routing_strategy=classification.get("routing_strategy", "analysis"),
                confidence=float(classification.get("confidence", 0.7)),
                reasoning=classification.get("reasoning", "Unified classification"),
                
                # Mixed request handling
                is_mixed_request=classification.get("is_mixed_request", False),
                question_context=classification.get("question_context"),
                astrosage_required=classification.get("astrosage_required", False),
                
                # ===== ORCHESTRATOR REQUIRED ATTRIBUTES =====
                question_category=question_category,
                complexity_level=complexity_level,
                
                # Parameter fields
                parameters=parameters,
                parameter_confidence=param_confidence,
                parameter_source=param_source,
                
                # Workflow and guidance
                suggested_workflow=workflow.get("suggested_steps", []),
                parameter_explanations=guidance.get("parameter_explanations", {}),
                potential_issues=guidance.get("potential_issues", []),
                
                # Context
                requires_file=context_data.get("requires_file", False)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to parse unified response: {str(e)}")
            self.logger.error(f"Raw output: {raw_output[:500]}...")
            return self._create_fallback_parsing_result(raw_output)
    
    def _extract_json_from_response(self, text: str) -> str:
        """Extract JSON from various response formats"""
        text = text.strip()
        
        # Method 1: JSON code blocks
        json_pattern = r'```json\s*(.*?)\s*```'
        json_match = re.search(json_pattern, text, re.DOTALL | re.IGNORECASE)
        if json_match:
            json_text = json_match.group(1).strip()
        else:
            # Method 2: Find JSON object
            brace_pattern = r'\{.*\}'
            brace_match = re.search(brace_pattern, text, re.DOTALL)
            if brace_match:
                return brace_match.group(0).strip()
            else:
                # Method 3: Assume entire text is JSON
                json_text = text

        # Remove JavaScript-style comments
        # Remove single-line comments: // comment
        json_text = re.sub(r'//.*?(?=\n|$)', '', json_text)

        # Remove multi-line comments: /* comment */
        json_text = re.sub(r'/\*.*?\*/', '', json_text, flags=re.DOTALL)

        # Remove trailing sommas (also common JSON error from LLMs)
        json_text = re.sub(r',\s*([\]}])', r'\1', json_text)

        return json_text
    
    def _enhance_with_context(self, result: UnifiedFITSResult, context: Dict[str, Any]) -> UnifiedFITSResult:
        """Enhance result with context information"""
        
        # File availability
        if context.get("has_uploaded_files"):
            result.file_available = True
            if result.analysis_types:
                result.requires_file = True
        
        # User expertise adaptation
        user_expertise = context.get("user_expertise", "intermediate")
        if user_expertise in ["beginner", "intermediate", "advanced"]:
            result.complexity_level = user_expertise
        
        if user_expertise == "beginner" and result.primary_intent == "analysis":
            # Add educational guidance for beginners
            if not result.is_mixed_request:
                result.astrosage_required = True
                result.suggested_workflow.append("provide_beginner_explanation")

        # Special workflow for metadata
        if "metadata" in result.analysis_types:
            # if result.is_mixed_request:
            #     # Metadata + interpretation
            #     if "extract_metadata" not in result.suggested_workflow:
            #         result.suggested_workflow.insert(0, "extract_metadata")
            #     result.suggested_workflow.append("interpret_metadata_with_astrosage")
            # else:
            #     # Pure metadata or metadayta + analysis
            #     if "extract_metadata" not in result.suggested_workflow:
            #         result.suggested_workflow.insert(0, "extract_metadata")

            if result.is_mixed_request:
                # Mixed requests always need AstroSage
                result.astrosage_required = True
            elif context.get("user_expertise") == "beginner":
                # Beginners need explanation
                result.astrosage_required = True
            elif result.question_category == "astronomy":
                # Questions about IRAS 13224-3809, telescopes need explanation
                result.astrosage_required = True
            elif result.primary_intent == "general":
                # Pure questions need AstroSage
                result.astrosage_required = True
            else:
                # Pure technical metadata extraction
                result.astrosage_required = False
        
        return result
    
    def _validate_and_optimize_parameters(self, result: UnifiedFITSResult) -> UnifiedFITSResult:
        """Enhanced parameter validation with comprehensive shared parameter handling"""
        
        # Define shared parameter groups
        TRIPLE_OVERLAP_PARAMS = {"low_freq", "high_freq", "bins"}  # PSD + Both Fitting
        FITTING_SHARED_PARAMS = {"noise_bound_percent", "A_min", "A_max", "maxfev"}  # Both Fitting only
        SPECIAL_SHARED_PARAMS = {"A0"}  # Same name, different defaults
        
        # Step 1: Identify analysis types by category
        psd_types = [t for t in result.analysis_types if t == "psd"]
        fitting_types = [t for t in result.analysis_types if t.startswith("fitting_")]
        frequency_analysis_types = psd_types + fitting_types  # Types that use frequency parameters
        
        # Step 2: Handle TRIPLE OVERLAP parameters (PSD + Both Fitting)
        # if len(frequency_analysis_types) > 1:
        #     result = self._resolve_triple_overlap_parameters(
        #         result, frequency_analysis_types, TRIPLE_OVERLAP_PARAMS
        #     )

        if len(frequency_analysis_types) > 1:
            result = self._resolve_triple_overlap_parameters_simple(
                result, frequency_analysis_types, TRIPLE_OVERLAP_PARAMS
            )
        
        # Step 3: Handle FITTING SHARED parameters (Both Fitting Models)
        if len(fitting_types) > 1:
            result = self._resolve_fitting_shared_parameters(
                result, fitting_types, FITTING_SHARED_PARAMS
            )
        
        # Step 4: Handle SPECIAL SHARED parameters (A0 with different defaults)
        if len(fitting_types) > 1:
            result = self._resolve_special_shared_parameters(
                result, fitting_types, SPECIAL_SHARED_PARAMS
            )
        
        # Step 5: Fill defaults for each analysis type
        for analysis_type in result.analysis_types:
            if analysis_type == "metadata":
                # Metadata ไม่ต้องมี parameters ซับซ้อน
                if analysis_type not in result.parameters:
                    result.parameters[analysis_type] = {}  # Empty parameters
                result.parameter_source[analysis_type] = "extract_all" # Not "user_specified"
                result.parameter_confidence[analysis_type] = 1.0  # High confidence for metadata

            elif analysis_type in result.parameters:
                params = result.parameters[analysis_type]
                schema = self.parameter_schemas.get(analysis_type, {})
                defaults = schema.get("defaults", {})
                
                # Fill missing parameters with defaults
                for param_name, default_value in defaults.items():
                    if param_name not in params:
                        params[param_name] = default_value
                        
                        # Update parameter source
                        if analysis_type not in result.parameter_source:
                            result.parameter_source[analysis_type] = "defaults_used"
                        elif result.parameter_source[analysis_type] == "user_specified":
                            result.parameter_source[analysis_type] = "mixed"
                
                # Validate parameter ranges
                validation_errors = self._validate_parameter_ranges(analysis_type, params)
                if validation_errors:
                    result.potential_issues.extend(validation_errors)
            else:
                # Use defaults for unspecified analysis types
                schema = self.parameter_schemas.get(analysis_type, {})
                defaults = schema.get("defaults", {})
                if defaults:
                    result.parameters[analysis_type] = defaults.copy()
                    result.parameter_source[analysis_type] = "defaults_used"
                    result.parameter_confidence[analysis_type] = 0.8
        
        # Step 6: Final consistency validation
        consistency_errors = self._validate_comprehensive_parameter_consistency(
            result.parameters, frequency_analysis_types, fitting_types
        )
        if consistency_errors:
            result.potential_issues.extend(consistency_errors)
        
        # Step 7: Remove extra parameters
        requested_types = set(result.analysis_types)
        current_param_types = set(result.parameters.keys())
        extra_types = current_param_types - requested_types
        
        for extra_type in extra_types:
            result.parameters.pop(extra_type, None)
            result.parameter_source.pop(extra_type, None)
            result.parameter_confidence.pop(extra_type, None)
        
        return result

    def _resolve_triple_overlap_parameters_simple(self, 
                                                result: UnifiedFITSResult, 
                                                frequency_analysis_types: List[str],
                                                triple_params: set) -> UnifiedFITSResult:
        """
        SIMPLER APPROACH: Only resolve when there's actual ambiguity
        If each analysis type has different explicit values, keep them as-is
        """
        
        for param_name in triple_params:
            # Collect values from all frequency analysis types
            param_values = {}
            for analysis_type in frequency_analysis_types:
                if (analysis_type in result.parameters and 
                    param_name in result.parameters[analysis_type]):
                    param_values[analysis_type] = result.parameters[analysis_type][param_name]
            
            if len(param_values) == 0:
                continue
            elif len(set(param_values.values())) == 1:
                # All values are the same, propagate to all types that don't have it
                shared_value = list(param_values.values())[0]
                for analysis_type in frequency_analysis_types:
                    if analysis_type not in result.parameters:
                        result.parameters[analysis_type] = {}
                    if param_name not in result.parameters[analysis_type]:
                        result.parameters[analysis_type][param_name] = shared_value
            else:
                # Different values exist
                # NEW LOGIC: Only resolve if we have partial specification
                
                total_types = len(frequency_analysis_types)
                specified_types = len(param_values)
                
                if specified_types == total_types:
                    # Each analysis type has its own value - keep as explicit specification
                    explanation = f"Using explicit {param_name} values per analysis type: {param_values}"
                    result.parameter_explanations[f"{param_name}_explicit_values"] = explanation
                    
                    # No changes needed - values are already correctly assigned
                    
                else:
                    # Partial specification - apply conflict resolution
                    last_mentioned_value = list(param_values.values())[-1]
                    
                    # Apply to all types
                    for analysis_type in frequency_analysis_types:
                        if analysis_type not in result.parameters:
                            result.parameters[analysis_type] = {}
                        result.parameters[analysis_type][param_name] = last_mentioned_value
                    
                    explanation = f"Resolved {param_name} conflict: used last mentioned value ({last_mentioned_value}) across {total_types} analysis types"
                    result.parameter_explanations[f"{param_name}_conflict_resolution"] = explanation
        
        return result

    def _resolve_fitting_shared_parameters(self, 
                                        result: UnifiedFITSResult,
                                        fitting_types: List[str], 
                                        fitting_params: set) -> UnifiedFITSResult:
        """Resolve parameters shared between fitting models only"""
        
        for param_name in fitting_params:
            # Collect values from fitting types
            param_values = {}
            for fitting_type in fitting_types:
                if (fitting_type in result.parameters and 
                    param_name in result.parameters[fitting_type]):
                    param_values[fitting_type] = result.parameters[fitting_type][param_name]
            
            if len(param_values) == 0:
                continue
            elif len(set(param_values.values())) == 1:
                # All values are the same, propagate to all fitting types
                shared_value = list(param_values.values())[0]
                for fitting_type in fitting_types:
                    if fitting_type not in result.parameters:
                        result.parameters[fitting_type] = {}
                    result.parameters[fitting_type][param_name] = shared_value
            else:
                # Conflict detected
                resolved_value = list(param_values.values())[-1]  # Last mentioned
                
                # Apply resolved value to all fitting types
                for fitting_type in fitting_types:
                    if fitting_type not in result.parameters:
                        result.parameters[fitting_type] = {}
                    result.parameters[fitting_type][param_name] = resolved_value
                
                # Add explanation
                explanation = f"Resolved {param_name} across {len(fitting_types)} fitting models: used last mentioned value ({resolved_value})"
                result.parameter_explanations[f"{param_name}_fitting_shared"] = explanation
        
        return result

    def _resolve_special_shared_parameters(self, 
                                        result: UnifiedFITSResult,
                                        fitting_types: List[str], 
                                        special_params: set) -> UnifiedFITSResult:
        """Handle A0 parameter with special logic for different defaults"""
        
        if "A0" not in special_params or len(fitting_types) <= 1:
            return result
        
        # Check A0 specifications
        a0_values = {}
        for fitting_type in fitting_types:
            if (fitting_type in result.parameters and 
                "A0" in result.parameters[fitting_type]):
                a0_values[fitting_type] = result.parameters[fitting_type]["A0"]
        
        if len(a0_values) == 0:
            # No A0 specified - use model-specific defaults
            for fitting_type in fitting_types:
                if fitting_type not in result.parameters:
                    result.parameters[fitting_type] = {}
                
                if fitting_type == "power_law":
                    result.parameters[fitting_type]["A0"] = 1.0
                elif fitting_type == "bending_power_law":
                    result.parameters[fitting_type]["A0"] = 10.0
            
            explanation = "Used model-specific A0 defaults: power_law=1.0, bending=10.0 (optimized for each model)"
            result.parameter_explanations["A0_model_defaults"] = explanation
        
        elif len(a0_values) == 1:
            # One A0 specified - could be intentional for one model or general
            specified_type = list(a0_values.keys())[0]
            specified_value = a0_values[specified_type]
            
            # Apply to the specified model, use defaults for others
            for fitting_type in fitting_types:
                if fitting_type not in result.parameters:
                    result.parameters[fitting_type] = {}
                
                if fitting_type == specified_type:
                    result.parameters[fitting_type]["A0"] = specified_value
                else:
                    # Use model-specific default for unspecified
                    if fitting_type == "power_law":
                        result.parameters[fitting_type]["A0"] = 1.0
                    elif fitting_type == "bending_power_law":
                        result.parameters[fitting_type]["A0"] = 10.0
            
            explanation = f"A0={specified_value} applied to {specified_type}, others use model-specific defaults"
            result.parameter_explanations["A0_partial_specification"] = explanation
        
        else:
            # Multiple A0 specified - use as specified
            for fitting_type in fitting_types:
                if fitting_type not in result.parameters:
                    result.parameters[fitting_type] = {}
                
                if fitting_type in a0_values:
                    result.parameters[fitting_type]["A0"] = a0_values[fitting_type]
                else:
                    # Use default for unspecified
                    if fitting_type == "power_law":
                        result.parameters[fitting_type]["A0"] = 1.0
                    elif fitting_type == "bending_power_law":
                        result.parameters[fitting_type]["A0"] = 10.0
            
            explanation = f"Model-specific A0 values applied: {a0_values}"
            result.parameter_explanations["A0_model_specific"] = explanation
        
        return result

    def _validate_comprehensive_parameter_consistency(self, 
                                                    parameters: Dict[str, Dict[str, Any]],
                                                    frequency_analysis_types: List[str],
                                                    fitting_types: List[str]) -> List[str]:
        """Validate consistency across all shared parameter groups"""
        
        errors = []
        
        # Check triple overlap consistency (PSD + Fitting)
        triple_params = ["low_freq", "high_freq", "bins"]
        if len(frequency_analysis_types) > 1:
            for param_name in triple_params:
                values = []
                types_with_param = []
                
                for analysis_type in frequency_analysis_types:
                    if analysis_type in parameters and param_name in parameters[analysis_type]:
                        values.append(parameters[analysis_type][param_name])
                        types_with_param.append(analysis_type)
                
                if len(set(values)) > 1:
                    errors.append(f"Inconsistent triple overlap {param_name}: {dict(zip(types_with_param, values))}")
        
        # Check fitting-only shared consistency
        fitting_params = ["noise_bound_percent", "A_min", "A_max", "maxfev"]
        if len(fitting_types) > 1:
            for param_name in fitting_params:
                values = []
                types_with_param = []
                
                for fitting_type in fitting_types:
                    if fitting_type in parameters and param_name in parameters[fitting_type]:
                        values.append(parameters[fitting_type][param_name])
                        types_with_param.append(fitting_type)
                
                if len(set(values)) > 1:
                    errors.append(f"Inconsistent fitting shared {param_name}: {dict(zip(types_with_param, values))}")
        
        # Note: A0 is intentionally NOT validated for consistency since it can have different values
        
        return errors
    
    def _validate_parameter_ranges(self, analysis_type: str, params: Dict[str, Any]) -> List[str]:
        """Validate parameter ranges for analysis type"""
        errors = []
        schema = self.parameter_schemas.get(analysis_type, {})
        param_defs = schema.get("parameters", {})
        
        for param_name, param_value in params.items():
            if param_name in param_defs:
                param_def = param_defs[param_name]
                param_range = param_def.get("range")
                
                if param_range and isinstance(param_value, (int, float)):
                    min_val, max_val = param_range
                    if param_value < min_val or param_value > max_val:
                        errors.append(f"{analysis_type}.{param_name} ({param_value}) outside valid range [{min_val}, {max_val}]")
        
        # Cross-parameter validation
        if analysis_type in ["psd", "power_law", "bending_power_law"]:
            if "low_freq" in params and "high_freq" in params:
                if params["low_freq"] >= params["high_freq"]:
                    errors.append(f"{analysis_type}: low_freq must be < high_freq")
        
        return errors
    
    def _create_fallback_result(self, user_input: str, error: str) -> UnifiedFITSResult:
        """Create fallback result when processing fails - ORCHESTRATOR COMPATIBLE"""
        return UnifiedFITSResult(
            primary_intent="analysis",
            analysis_types=["statistics"],  # Safe default
            routing_strategy="analysis",
            confidence=0.3,
            reasoning=f"Fallback result due to error: {error}",
            question_category="unknown",  # ✅ ORCHESTRATOR REQUIRED
            complexity_level="intermediate",  # ✅ ORCHESTRATOR REQUIRED
            parameters={"statistics": self.parameter_schemas["statistics"]["defaults"]},
            parameter_confidence={"statistics": 0.3},
            parameter_source={"statistics": "defaults_used"},
            suggested_workflow=["check_input", "retry_request"],
            potential_issues=[f"processing_error: {error}"],
            requires_file=True
        )
    
    def _create_fallback_parsing_result(self, raw_output: str) -> UnifiedFITSResult:
        """Create result when JSON parsing fails but we have text - ORCHESTRATOR COMPATIBLE"""
        
        # Try to extract basic intent from text
        text_lower = raw_output.lower()
        
        if "mixed" in text_lower:
            primary_intent = "mixed"
            routing_strategy = "mixed"
            astrosage_required = True
        elif "general" in text_lower or "question" in text_lower:
            primary_intent = "general"
            routing_strategy = "astrosage"
            astrosage_required = True
        else:
            primary_intent = "analysis"
            routing_strategy = "analysis"
            astrosage_required = False
        
        # Extract analysis types mentioned
        analysis_types = []
        if "statistics" in text_lower or "mean" in text_lower:
            analysis_types.append("statistics")
        if "psd" in text_lower or "power spectral" in text_lower:
            analysis_types.append("psd")
        if "fitting" in text_lower or "power law" in text_lower:
            if "bending" in text_lower:
                analysis_types.append("bending_power_law")
            else:
                analysis_types.append("power_law")
        
        if not analysis_types and primary_intent == "analysis":
            analysis_types = ["statistics"]  # Safe default
        
        # Determine question category from text
        question_category = "unknown"
        if "neutron star" in text_lower or "black hole" in text_lower or "stellar" in text_lower:
            question_category = "astronomy"
        elif "psd" in text_lower or "statistics" in text_lower or "fitting" in text_lower:
            question_category = "data_analysis"
        elif "physics" in text_lower or "quantum" in text_lower or "emission" in text_lower:
            question_category = "physics"
        elif "how" in text_lower and ("analyze" in text_lower or "method" in text_lower):
            question_category = "methods"
        
        # Build parameters with defaults
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
            reasoning="Fallback text analysis due to JSON parsing failure",
            question_category=question_category,  # ✅ ORCHESTRATOR REQUIRED
            complexity_level="intermediate",  # ✅ ORCHESTRATOR REQUIRED
            parameters=parameters,
            parameter_confidence=parameter_confidence,
            parameter_source=parameter_source,
            astrosage_required=astrosage_required,
            suggested_workflow=["verify_request", "reprocess_if_needed"],
            potential_issues=["json_parsing_failed"],
            requires_file=(primary_intent == "analysis")
        )
    
    def _cleanup_cache(self):
        """Clean up expired cache entries"""

        """
        1. Check timestamps of cached entries

        2. If older than TTL (e.g., 1 hour), remove from cache

        3. Run periodically before adding new entries
        
        """
        current_time = datetime.now()
        expired_keys = []
        
        for cache_key, timestamp in self.cache_timestamps.items():
            if current_time - timestamp > timedelta(seconds=self.cache_ttl):
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            self.cache.pop(key, None)
            self.cache_timestamps.pop(key, None)
    
    def _generate_cache_key(self, user_input: str, context: Dict[str, Any]) -> str:
        """Generate cache key from input and context"""
        """
        Processing Steps:
        1. Normalize user input: lowercase, trim whitespace, collapse spaces
        # Input: "  Calculate   STATISTICS   for my data  "
        normalized_input = "calculate statistics for my data"

        2. Build context key with sorted previous analyses
        context_key = {
            "has_files": True,
            "expertise": "advanced", 
            "prev_analyses": ["psd", "statistics"]  # sorted
            }
        
        3. Combine normalized input and context key into single string
        combined = "calculate statistics for my data:{\"has_files\": true, \"expertise\": \"advanced\", \"prev_analyses\": [\"psd\", \"statistics\"]}"

        4. Generate MD5 Hash: Hash combined string with MD5 for fixed-length key 
        cache_key = "a1b2c3d4e5f6789..."  # 32-character hex string

        5. Use cache_key for caching
        6. Cache TTL: Entries expire after defined time (e.g., 1 hour)
        
        """
        normalized_input = re.sub(r'\s+', ' ', user_input.lower().strip())
        context_key = {
            "has_files": context.get("has_uploaded_files", False),
            "expertise": context.get("user_expertise", "intermediate"),
            "prev_analyses": sorted(context.get("previous_analyses", []))
        }
        
        combined = f"{normalized_input}:{json.dumps(context_key, sort_keys=True)}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics with question categorization"""
        total = max(self.stats["total_requests"], 1)
        
        return {
            "usage": {
                "total_requests": self.stats["total_requests"],
                "cache_hits": self.stats["cache_hits"],
                "cache_hit_rate": round(self.stats["cache_hits"] / total, 3),
                "total_tokens_used": self.stats["total_tokens"],
                "total_cost": round(self.stats["total_cost"], 4),
                "avg_tokens_per_request": round(self.stats["total_tokens"] / total, 1),
                "avg_cost_per_request": round(self.stats["total_cost"] / total, 6),
                "avg_processing_time": round(self.stats["avg_processing_time"], 3)
            },
            "classification_distribution": {
                "analysis_requests": self.stats["intent_distribution"]["analysis"],
                "general_questions": self.stats["intent_distribution"]["general"],
                "mixed_requests": self.stats["intent_distribution"]["mixed"],
                "analysis_percentage": round(self.stats["intent_distribution"]["analysis"] / total * 100, 1),
                "general_percentage": round(self.stats["intent_distribution"]["general"] / total * 100, 1),
                "mixed_percentage": round(self.stats["intent_distribution"]["mixed"] / total * 100, 1)
            },
            "question_categories": {
                "astronomy": self.stats["question_categories"]["astronomy"],
                "physics": self.stats["question_categories"]["physics"],
                "data_analysis": self.stats["question_categories"]["data_analysis"],
                "methods": self.stats["question_categories"]["methods"],
                "unknown": self.stats["question_categories"]["unknown"],
                "astronomy_percentage": round(self.stats["question_categories"]["astronomy"] / total * 100, 1),
                "physics_percentage": round(self.stats["question_categories"]["physics"] / total * 100, 1),
                "data_analysis_percentage": round(self.stats["question_categories"]["data_analysis"] / total * 100, 1),
                "methods_percentage": round(self.stats["question_categories"]["methods"] / total * 100, 1)
            },
            "analysis_types": {
                "statistics": self.stats["analysis_types_distribution"]["statistics"],
                "psd": self.stats["analysis_types_distribution"]["psd"],
                "power_law": self.stats["analysis_types_distribution"]["power_law"],
                "bending_power_law": self.stats["analysis_types_distribution"]["bending_power_law"],
                "metadata": self.stats["analysis_types_distribution"]["metadata"],
                "most_used": max(self.stats["analysis_types_distribution"].items(), key=lambda x: x[1])[0],
            },
            "parameter_stats": {
                "parameter_extractions": self.stats["parameter_extractions"],
                "parameter_extraction_rate": round(self.stats["parameter_extractions"] / total, 3)
            },
            "cost_projections": {
                "cost_per_1000_requests": round(self.stats["total_cost"] / total * 1000, 2),
                "monthly_cost_10k_requests": round(self.stats["total_cost"] / total * 10000, 2),
                "monthly_cost_100k_requests": round(self.stats["total_cost"] / total * 100000, 2)
            },
            "cache_info": {
                "cache_size": len(self.cache),
                "cache_efficiency": round(self.stats["cache_hits"] / total, 3) if total > 0 else 0
            }
        }
    
    def clear_cache(self):
        """Clear the cache"""
        self.cache.clear()
        self.cache_timestamps.clear()
        self.logger.info("Cache cleared")
    
    def get_parameter_schema(self, analysis_type: str) -> Dict[str, Any]:
        """Get parameter schema for specific analysis type"""
        return self.parameter_schemas.get(analysis_type, {})
    
    def validate_parameters(self, analysis_type: str, parameters: Dict[str, Any]) -> List[str]:
        """Validate parameters for specific analysis type"""
        return self._validate_parameter_ranges(analysis_type, parameters)

