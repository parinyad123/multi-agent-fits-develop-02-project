"""
app/core/prompt_logger.py - FIXED VERSION

Comprehensive prompt logging system for debugging LLM interactions
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


class PromptLogger:
    """
    Centralized prompt logging for all LLM agents
    
    Features:
    - Logs complete prompts (system + user messages)
    - Logs responses
    - Tracks token usage and timing
    - Saves to structured JSON files
    - Provides formatted text output for human review
    """
    
    def __init__(self, log_base_dir: Optional[Path] = None):
        """
        Initialize PromptLogger
        
        Args:
            log_base_dir: Base directory for logs. If None, uses project root / "logs" / "prompts"
        """
        # ✅ FIX: Don't use settings.base_dir, compute it dynamically
        if log_base_dir is None:
            # Get project root (multi-agent-fits-dev-02/)
            # This file is at: app/core/prompt_logger.py
            # So project root is 2 levels up
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
            log_base_dir = project_root / "logs" / "prompts"
        
        self.base_log_dir = Path(log_base_dir)
        self.base_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories for each agent
        self.astrosage_dir = self.base_log_dir / "astrosage"
        self.rewrite_dir = self.base_log_dir / "rewrite"
        self.classification_dir = self.base_log_dir / "classification"
        
        for directory in [self.astrosage_dir, self.rewrite_dir, self.classification_dir]:
            directory.mkdir(exist_ok=True)
        
        logger.info(f"PromptLogger initialized: {self.base_log_dir}")
    
    def log_astrosage_prompt(
        self,
        session_id: str,
        user_id: UUID,
        user_query: str,
        messages: List[Dict[str, str]],
        routing_strategy: str,
        expertise_level: str,
        analysis_results: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log AstroSage prompt
        
        Returns:
            log_id: Unique identifier for this log
        """
        log_id = str(uuid4())
        timestamp = datetime.now()
        
        # Build log data
        log_data = {
            "log_id": log_id,
            "agent": "astrosage",
            "timestamp": timestamp.isoformat(),
            "session_id": session_id,
            "user_id": str(user_id),
            "routing_strategy": routing_strategy,
            "expertise_level": expertise_level,
            
            # User query
            "user_query": user_query,
            
            # Complete prompt
            "messages": messages,
            
            # Analysis results (if available)
            "analysis_results": self._extract_key_parameters(analysis_results) if analysis_results else None,
            
            # Additional metadata
            "metadata": metadata or {},
            
            # Statistics
            "statistics": {
                "system_prompt_length": len(messages[0]["content"]) if messages else 0,
                "user_message_length": len(messages[1]["content"]) if len(messages) > 1 else 0,
                "total_prompt_length": sum(len(m["content"]) for m in messages),
                "estimated_tokens": self._estimate_tokens(messages)
            }
        }
        
        # Save JSON
        json_file = self.astrosage_dir / f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{log_id[:8]}.json"
        
        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save JSON log: {e}")
        
        # Save human-readable text
        txt_file = self.astrosage_dir / f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{log_id[:8]}.txt"
        
        try:
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(self._format_astrosage_prompt(log_data))
        except Exception as e:
            logger.error(f"Failed to save text log: {e}")
        
        logger.info(
            f"AstroSage prompt logged: {log_id[:8]} "
            f"(~{log_data['statistics']['estimated_tokens']} tokens)"
        )
        
        return log_id
    
    def log_astrosage_response(
        self,
        log_id: str,
        response: str,
        tokens_used: int,
        response_time: float,
        model_used: str,
        success: bool,
        error: Optional[str] = None
    ):
        """
        Log AstroSage response (appends to existing log)
        """
        timestamp = datetime.now()
        
        # Find log files
        json_files = list(self.astrosage_dir.glob(f"*_{log_id[:8]}.json"))
        
        if not json_files:
            logger.warning(f"Cannot find log file for log_id: {log_id}")
            return
        
        json_file = json_files[0]
        
        try:
            # Load existing log
            with open(json_file, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            # Add response data
            log_data["response"] = {
                "content": response,
                "timestamp": timestamp.isoformat(),
                "model_used": model_used,
                "tokens_used": tokens_used,
                "response_time": response_time,
                "success": success,
                "error": error,
                "response_length": len(response),
                "response_word_count": len(response.split())
            }
            
            # Save updated JSON
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            
            # Append to text file
            txt_file = json_file.with_suffix('.txt')
            with open(txt_file, 'a', encoding='utf-8') as f:
                f.write("\n\n" + "="*80 + "\n")
                f.write("RESPONSE\n")
                f.write("="*80 + "\n\n")
                f.write(f"Model: {model_used}\n")
                f.write(f"Tokens: {tokens_used}\n")
                f.write(f"Time: {response_time:.2f}s\n")
                f.write(f"Success: {success}\n")
                if error:
                    f.write(f"Error: {error}\n")
                f.write(f"\n{response}\n")
            
            logger.info(
                f"AstroSage response logged: {log_id[:8]} "
                f"({tokens_used} tokens, {response_time:.2f}s)"
            )
            
        except Exception as e:
            logger.error(f"Failed to log AstroSage response: {e}", exc_info=True)
    
    def log_rewrite_prompt(
        self,
        session_id: str,
        user_id: UUID,
        user_query: str,
        messages: List[Dict[str, str]],
        routing_strategy: str,
        expertise_level: str,
        intermediate_results: List[Dict[str, Any]],
        model_tier: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log Rewrite Agent prompt
        
        Returns:
            log_id: Unique identifier for this log
        """
        log_id = str(uuid4())
        timestamp = datetime.now()
        
        # Extract key info from intermediate results
        analysis_summary = self._extract_analysis_summary(intermediate_results)
        astrosage_summary = self._extract_astrosage_summary(intermediate_results)
        
        # Build log data
        log_data = {
            "log_id": log_id,
            "agent": "rewrite",
            "timestamp": timestamp.isoformat(),
            "session_id": session_id,
            "user_id": str(user_id),
            "routing_strategy": routing_strategy,
            "expertise_level": expertise_level,
            "model_tier": model_tier,
            
            # User query
            "user_query": user_query,
            
            # Complete prompt
            "messages": messages,
            
            # Intermediate results summary
            "intermediate_results_summary": {
                "analysis": analysis_summary,
                "astrosage": astrosage_summary,
                "num_steps": len(intermediate_results)
            },
            
            # Additional metadata
            "metadata": metadata or {},
            
            # Statistics
            "statistics": {
                "system_prompt_length": len(messages[0]["content"]) if messages else 0,
                "user_message_length": len(messages[1]["content"]) if len(messages) > 1 else 0,
                "total_prompt_length": sum(len(m["content"]) for m in messages),
                "estimated_tokens": self._estimate_tokens(messages)
            }
        }
        
        # Save JSON
        json_file = self.rewrite_dir / f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{log_id[:8]}.json"
        
        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save JSON log: {e}")
        
        # Save human-readable text
        txt_file = self.rewrite_dir / f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{log_id[:8]}.txt"
        
        try:
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(self._format_rewrite_prompt(log_data))
        except Exception as e:
            logger.error(f"Failed to save text log: {e}")
        
        logger.info(
            f"Rewrite prompt logged: {log_id[:8]} "
            f"(~{log_data['statistics']['estimated_tokens']} tokens)"
        )
        
        return log_id
    
    def log_rewrite_response(
        self,
        log_id: str,
        response: str,
        tokens_used: int,
        response_time: float,
        model_used: str,
        validation_passed: bool,
        validation_errors: Optional[List[str]] = None,
        retry_count: int = 0
    ):
        """
        Log Rewrite Agent response (appends to existing log)
        """
        timestamp = datetime.now()
        
        # Find log files
        json_files = list(self.rewrite_dir.glob(f"*_{log_id[:8]}.json"))
        
        if not json_files:
            logger.warning(f"Cannot find log file for log_id: {log_id}")
            return
        
        json_file = json_files[0]
        
        try:
            # Load existing log
            with open(json_file, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            # Add response data
            log_data["response"] = {
                "content": response,
                "timestamp": timestamp.isoformat(),
                "model_used": model_used,
                "tokens_used": tokens_used,
                "response_time": response_time,
                "validation_passed": validation_passed,
                "validation_errors": validation_errors,
                "retry_count": retry_count,
                "response_length": len(response),
                "response_word_count": len(response.split())
            }
            
            # Save updated JSON
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            
            # Append to text file
            txt_file = json_file.with_suffix('.txt')
            with open(txt_file, 'a', encoding='utf-8') as f:
                f.write("\n\n" + "="*80 + "\n")
                f.write("RESPONSE\n")
                f.write("="*80 + "\n\n")
                f.write(f"Model: {model_used}\n")
                f.write(f"Tokens: {tokens_used}\n")
                f.write(f"Time: {response_time:.2f}s\n")
                f.write(f"Validation: {'✅ PASSED' if validation_passed else '❌ FAILED'}\n")
                if validation_errors:
                    f.write(f"\nValidation Errors:\n")
                    for err in validation_errors:
                        f.write(f"  - {err}\n")
                if retry_count > 0:
                    f.write(f"Retry Count: {retry_count}\n")
                f.write(f"\n{response}\n")
            
            logger.info(
                f"Rewrite response logged: {log_id[:8]} "
                f"({tokens_used} tokens, {response_time:.2f}s, "
                f"validation: {'pass' if validation_passed else 'fail'})"
            )
            
        except Exception as e:
            logger.error(f"Failed to log Rewrite response: {e}", exc_info=True)
    
    # ============================================
    # Helper Methods
    # ============================================
    
    def _estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Rough token estimation (1 token ≈ 4 chars)"""
        total_chars = sum(len(m["content"]) for m in messages)
        return total_chars // 4
    
    def _extract_key_parameters(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key parameters from analysis results for logging"""
        summary = {}
        
        if 'power_law' in analysis_results:
            pl = analysis_results['power_law']
            if 'fitted_parameters' in pl:
                summary['power_law'] = pl['fitted_parameters']
        
        if 'bending_power_law' in analysis_results:
            bpl = analysis_results['bending_power_law']
            if 'fitted_parameters' in bpl:
                summary['bending_power_law'] = bpl['fitted_parameters']
        
        return summary
    
    def _extract_analysis_summary(self, intermediate_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract analysis summary from intermediate results"""
        for step in intermediate_results:
            if step.get('step') == 'analysis':
                result = step.get('analysis_result', {})
                return {
                    'status': result.get('status'),
                    'completed_analyses': result.get('completed_analyses', []),
                    'failed_analyses': result.get('failed_analyses', []),
                    'has_power_law': 'power_law' in result.get('results', {}),
                    'has_bending_power_law': 'bending_power_law' in result.get('results', {})
                }
        return None
    
    def _extract_astrosage_summary(self, intermediate_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract AstroSage summary from intermediate results"""
        for step in intermediate_results:
            if step.get('step') == 'astrosage':
                return {
                    'success': step.get('success'),
                    'model_used': step.get('model_used'),
                    'tokens_used': step.get('tokens_used'),
                    'response_time': step.get('response_time'),
                    'response_length': len(step.get('response', ''))
                }
        return None
    
    def _format_astrosage_prompt(self, log_data: Dict[str, Any]) -> str:
        """Format AstroSage prompt for human-readable text file"""
        lines = []
        
        # Header
        lines.append("="*80)
        lines.append("ASTROSAGE PROMPT LOG")
        lines.append("="*80)
        lines.append(f"\nLog ID: {log_data['log_id']}")
        lines.append(f"Timestamp: {log_data['timestamp']}")
        lines.append(f"Session ID: {log_data['session_id']}")
        lines.append(f"User ID: {log_data['user_id']}")
        lines.append(f"Routing Strategy: {log_data['routing_strategy']}")
        lines.append(f"Expertise Level: {log_data['expertise_level']}")
        
        # Statistics
        stats = log_data['statistics']
        lines.append(f"\nPrompt Statistics:")
        lines.append(f"  - System prompt: {stats['system_prompt_length']:,} chars")
        lines.append(f"  - User message: {stats['user_message_length']:,} chars")
        lines.append(f"  - Total: {stats['total_prompt_length']:,} chars")
        lines.append(f"  - Estimated tokens: ~{stats['estimated_tokens']:,}")
        
        # Analysis results (if available)
        if log_data.get('analysis_results'):
            lines.append(f"\nAnalysis Results:")
            for key, params in log_data['analysis_results'].items():
                lines.append(f"  {key}:")
                for param_name, param_value in params.items():
                    if isinstance(param_value, float):
                        lines.append(f"    - {param_name}: {param_value:.6e}")
                    else:
                        lines.append(f"    - {param_name}: {param_value}")
        
        # User query
        lines.append(f"\n" + "-"*80)
        lines.append("USER QUERY")
        lines.append("-"*80)
        lines.append(f"\n{log_data['user_query']}\n")
        
        # Messages
        for i, message in enumerate(log_data['messages']):
            lines.append("\n" + "="*80)
            lines.append(f"MESSAGE {i+1}: {message['role'].upper()}")
            lines.append("="*80)
            lines.append(f"\nLength: {len(message['content']):,} chars")
            lines.append(f"\n{message['content']}\n")
        
        return "\n".join(lines)
    
    def _format_rewrite_prompt(self, log_data: Dict[str, Any]) -> str:
        """Format Rewrite prompt for human-readable text file"""
        lines = []
        
        # Header
        lines.append("="*80)
        lines.append("REWRITE AGENT PROMPT LOG")
        lines.append("="*80)
        lines.append(f"\nLog ID: {log_data['log_id']}")
        lines.append(f"Timestamp: {log_data['timestamp']}")
        lines.append(f"Session ID: {log_data['session_id']}")
        lines.append(f"User ID: {log_data['user_id']}")
        lines.append(f"Routing Strategy: {log_data['routing_strategy']}")
        lines.append(f"Expertise Level: {log_data['expertise_level']}")
        lines.append(f"Model Tier: {log_data['model_tier']}")
        
        # Statistics
        stats = log_data['statistics']
        lines.append(f"\nPrompt Statistics:")
        lines.append(f"  - System prompt: {stats['system_prompt_length']:,} chars")
        lines.append(f"  - User message: {stats['user_message_length']:,} chars")
        lines.append(f"  - Total: {stats['total_prompt_length']:,} chars")
        lines.append(f"  - Estimated tokens: ~{stats['estimated_tokens']:,}")
        
        # Intermediate results summary
        summary = log_data.get('intermediate_results_summary', {})
        lines.append(f"\nIntermediate Results Summary:")
        lines.append(f"  - Total steps: {summary.get('num_steps', 0)}")
        
        if summary.get('analysis'):
            anal = summary['analysis']
            lines.append(f"  - Analysis status: {anal.get('status')}")
            lines.append(f"    Completed: {', '.join(anal.get('completed_analyses', []))}")
            if anal.get('failed_analyses'):
                lines.append(f"    Failed: {', '.join(anal['failed_analyses'])}")
        
        if summary.get('astrosage'):
            sage = summary['astrosage']
            lines.append(f"  - AstroSage: {'✅ Success' if sage.get('success') else '❌ Failed'}")
            lines.append(f"    Model: {sage.get('model_used')}")
            lines.append(f"    Tokens: {sage.get('tokens_used')}")
            lines.append(f"    Response: {sage.get('response_length'):,} chars")
        
        # User query
        lines.append(f"\n" + "-"*80)
        lines.append("USER QUERY")
        lines.append("-"*80)
        lines.append(f"\n{log_data['user_query']}\n")
        
        # Messages
        for i, message in enumerate(log_data['messages']):
            lines.append("\n" + "="*80)
            lines.append(f"MESSAGE {i+1}: {message['role'].upper()}")
            lines.append("="*80)
            lines.append(f"\nLength: {len(message['content']):,} chars")
            lines.append(f"\n{message['content']}\n")
        
        return "\n".join(lines)


# ✅ Create global instance (without requiring settings.base_dir)
prompt_logger = PromptLogger()