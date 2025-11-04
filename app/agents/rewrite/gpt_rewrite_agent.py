"""
app/agents/rewrite/gpt_rewrite_agent.py - HYBRID VERSION
"""

from openai import AsyncOpenAI
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from app.core.config import settings
from app.agents.rewrite.models import RewriteRequest, RewriteResponse
from app.agents.rewrite.prompt_builder import RewritePromptBuilder

logger = logging.getLogger(__name__)


class GPTRewriteAgent:
    """
    Hybrid GPT Rewrite Agent
    
    Strategy:
    1. Use GPT-4o-mini by default (fast + cheap)
    2. Fallback to GPT-3.5-turbo if 4o-mini unavailable
    3. Optional: Use GPT-4o for complex/expert queries
    """
    
    # Model configurations
    MODELS = {
        "mini": {
            "name": "gpt-4o-mini",
            "temperature": 0.3,
            "max_tokens": 3000,
            "cost_per_1k_input": 0.00015,
            "cost_per_1k_output": 0.0006
        },
        "turbo": {
            "name": "gpt-3.5-turbo",
            "temperature": 0.3,
            "max_tokens": 3000,
            "cost_per_1k_input": 0.0005,
            "cost_per_1k_output": 0.0015
        },
        "standard": {
            "name": "gpt-4o",
            "temperature": 0.3,
            "max_tokens": 3000,
            "cost_per_1k_input": 0.005,
            "cost_per_1k_output": 0.015
        }
    }
    
    def __init__(
        self,
        default_model: str = "turbo",  # "mini", "turbo", or "standard"
        auto_upgrade: bool = False     # Auto-upgrade for expert queries
    ):
        self.default_model = default_model
        self.auto_upgrade = auto_upgrade
        
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.prompt_builder = RewritePromptBuilder() 
        
        logger.info(
            f"GPT Rewrite Agent initialized: "
            f"default={default_model}, auto_upgrade={auto_upgrade}"
        )
    
    async def rewrite_response(
        self,
        user_input: str,
        context: Dict[str, Any],
        intermediate_results: List[Dict[str, Any]]
    ) -> str:
        """
        Main entry point with automatic model selection
        """
        
        # Determine which model to use
        model_tier = self._select_model(context, intermediate_results)
        
        logger.info(f"Using model tier: {model_tier}")
        
        try:
            # Build request
            request = self._build_request(user_input, context, intermediate_results)
            
            # Build prompt
            messages = self.prompt_builder.build_prompt(request)

            # Select model
            model_tier = self._select_model(context, intermediate_results)
            
            # Call GPT with selected model
            response = await self._call_gpt(messages, model_tier)
            
            return response
            
        except Exception as e:
            logger.error(f"Rewrite failed: {e}", exc_info=True)
            
            # Fallback strategy
            if model_tier != "turbo":
                logger.info("Retrying with turbo model...")
                try:
                    response = await self._call_gpt(messages, "turbo")
                    return response
                except:
                    pass
            
            # Final fallback: basic formatting
            return self._create_fallback_response(intermediate_results)
    
    def _select_model(
        self,
        context: Dict[str, Any],
        intermediate_results: List[Dict[str, Any]]
    ) -> str:
        """
        Select appropriate model based on query complexity
        
        Logic:
        1. Default: use mini (fast + cheap)
        2. Expert level + complex query: upgrade to standard
        3. Very long results: use mini (works well with long context)
        """
        
        # Check expertise level
        expertise = context.get('user_expertise', 'intermediate')
        
        # Check if auto-upgrade enabled
        if not self.auto_upgrade:
            return self.default_model
        
        # Check complexity indicators
        complexity_score = 0
        
        # Factor 1: Expertise level
        if expertise == "expert":
            complexity_score += 2
        elif expertise == "advanced":
            complexity_score += 1
        
        # Factor 2: Number of analysis types
        analysis_step = None
        for step in intermediate_results:
            if step.get('step') == 'analysis':
                analysis_step = step
                break
        
        if analysis_step:
            results = analysis_step.get('analysis_result', {}).get('results', {})
            num_analyses = len(results)
            
            if num_analyses >= 4:
                complexity_score += 2
            elif num_analyses >= 2:
                complexity_score += 1
        
        # Factor 3: AstroSage involved (means interpretation needed)
        has_astrosage = any(
            step.get('step') == 'astrosage' 
            for step in intermediate_results
        )
        if has_astrosage:
            complexity_score += 1
        
        # Decision logic
        if complexity_score >= 4 and expertise in ["expert", "advanced"]:
            return "standard"  # Use GPT-4o for high complexity
        else:
            return "mini"      # Use GPT-4o-mini for most cases
    
    async def _call_gpt(
        self,
        messages: List[Dict[str, str]],
        model_tier: str
    ) -> str:
        """Call GPT with specified model"""
        
        config = self.MODELS[model_tier]
        
        start_time = datetime.now()
        
        response = await self.client.chat.completions.create(
            model=config["name"],
            messages=messages,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"]
        )
        
        content = response.choices[0].message.content
        
        # Calculate cost
        usage = response.usage
        cost = (
            usage.prompt_tokens / 1000 * config["cost_per_1k_input"] +
            usage.completion_tokens / 1000 * config["cost_per_1k_output"]
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        
        logger.info(
            f"GPT response: model={config['name']}, "
            f"tokens={usage.total_tokens}, "
            f"cost=${cost:.4f}, "
            f"time={duration:.2f}s"
        )
        
        return content
    
    def _build_request(self, user_input, context, intermediate_results):
        """Build request (same as before)"""
        # from app.agents.rewrite.models import RewriteRequest
        
        routing_strategy = "unknown"
        for step in intermediate_results:
            if step.get('step') == 'classification':
                result = step.get('classification_result', {})
                routing_strategy = result.get('routing_strategy', 'unknown')
                break
        
        expertise_level = context.get('user_expertise', 'intermediate')
        
        return RewriteRequest(
            user_query=user_input,
            routing_strategy=routing_strategy,
            completed_steps=intermediate_results,
            expertise_level=expertise_level
        )
    
    def _create_fallback_response(self, intermediate_results):
        """Fallback response (same as before)"""
        
        analysis_results = None
        for step in intermediate_results:
            if step.get('step') == 'analysis':
                analysis_results = step.get('analysis_result', {}).get('results', {})
                break
        
        if not analysis_results:
            return "I apologize, but I encountered an error processing your request."
        
        response = "# Analysis Results\n\n"
        
        for analysis_type, result in analysis_results.items():
            response += f"## {analysis_type.replace('_', ' ').title()}\n\n"
            
            # Format key parameters
            if isinstance(result, dict):
                if 'fitted_parameters' in result:
                    params = result['fitted_parameters']
                    for key, value in params.items():
                        if isinstance(value, (int, float)):
                            response += f"- **{key}**: {value:.6e}\n"
                        else:
                            response += f"- **{key}**: {value}\n"
            
            response += "\n"
        
        return response