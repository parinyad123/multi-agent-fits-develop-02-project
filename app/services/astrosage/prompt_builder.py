"""
multi-agent-fits-dev-02/app/services/astrosage/prompt_builder.py

Build prompts for AstroSage LLM
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
    """
    
    # Base system prompt (always included)
    BASE_SYSTEM_PROMPT = """You are AstroSage, an expert astrophysicist AI assistant specializing in stellar physics, cosmology, and observational astronomy. You provide accurate, detailed scientific explanations while maintaining conversation continuity. Always reference previous discussion points when relevant."""
    
    @classmethod
    def build_system_prompt(cls, expertise_level: ExpertiseLevel) -> str:
        """
        Build system prompt based on expertise level
        
        Args:
            expertise_level: User's expertise level
        
        Returns:
            Complete system prompt
        """
        # Start with base prompt
        prompt = cls.BASE_SYSTEM_PROMPT
        
        # Add expertise-specific modifier
        modifier = ExpertiseAdapter.get_system_prompt_modifier(expertise_level)
        prompt += "\n" + modifier
        
        return prompt

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
            Formatted conversation context (empty string if none)
        """
        if not conversations:
            return ""
        
        lines = ["\n\nPREVIOUS CONVERSATION (Last 10 exchanges):\n"]
        
        for pair in conversations:
            # Format time
            time_str = cls._format_timestamp(pair.timestamp)
            
            # User message
            lines.append(f"[User - {time_str}]: {pair.user_message}")
            
            # Assistant message
            lines.append(f"[AstroSage]: {pair.assistant_message}\n")
        
        lines.append("\nPlease maintain continuity with the above conversation when answering.")
        
        return "\n".join(lines)
    
    @classmethod
    def build_analysis_context(
        cls,
        analysis_results: Optional[Dict[str, Any]]
    ) -> str:
        """
        Build analysis results context section
        
        Args:
            analysis_results: Results from FITS analysis
        
        Returns:
            Formatted analysis context (empty string if none)
        """
        if not analysis_results:
            return ""
        
        lines = ["\n\nANALYSIS RESULTS FROM USER'S FITS FILE:\n"]
        
        # Statistics
        if 'statistics' in analysis_results:
            lines.append(cls._format_statistics(analysis_results['statistics']))
        
        # PSD
        if 'psd' in analysis_results:
            lines.append(cls._format_psd(analysis_results['psd']))
        
        # Power Law
        if 'power_law' in analysis_results:
            lines.append(cls._format_power_law(analysis_results['power_law']))
        
        # Bending Power Law
        if 'bending_power_law' in analysis_results:
            lines.append(cls._format_bending_power_law(analysis_results['bending_power_law']))
        
        # Metadata
        if 'metadata' in analysis_results:
            lines.append(cls._format_metadata(analysis_results['metadata']))
        
        lines.append("\nPlease reference these results when answering the user's question.")
        
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
                request.analysis_results
            )
            system_prompt += analysis_context
        
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
        
        logger.debug(f"Built prompt with {len(system_prompt)} characters in system message")
        
        return messages

    # ==========================================
    # Helper Methods for Formatting
    # ==========================================
    
    @staticmethod
    def _format_statistics(stats: Dict[str, Any]) -> str:
        """Format statistics results"""
        lines = ["Statistics:"]
        
        if 'statistics' in stats:
            stats = stats['statistics']
        
        if 'mean' in stats:
            lines.append(f"  - Mean: {stats['mean']:.6e}")
        if 'median' in stats:
            lines.append(f"  - Median: {stats['median']:.6e}")
        if 'std' in stats:
            lines.append(f"  - Standard Deviation: {stats['std']:.6e}")
        if 'min' in stats:
            lines.append(f"  - Min: {stats['min']:.6e}")
        if 'max' in stats:
            lines.append(f"  - Max: {stats['max']:.6e}")
        if 'count' in stats:
            lines.append(f"  - Data Points: {stats['count']}")
        
        return "\n".join(lines) + "\n"
    
    @staticmethod
    def _format_psd(psd: Dict[str, Any]) -> str:
        """Format PSD results"""
        lines = ["Power Spectral Density Analysis:"]
        
        if 'freq_range' in psd:
            freq_range = psd['freq_range']
            lines.append(f"  - Frequency Range: {freq_range[0]:.6e} to {freq_range[1]:.6e} Hz")
        
        if 'n_points' in psd:
            lines.append(f"  - Number of Frequency Bins: {psd['n_points']}")
        
        if 'psd_range' in psd:
            psd_range = psd['psd_range']
            lines.append(f"  - PSD Range: {psd_range[0]:.6e} to {psd_range[1]:.6e}")
        
        return "\n".join(lines) + "\n"
    
    @staticmethod
    def _format_power_law(power_law: Dict[str, Any]) -> str:
        """Format power law fit results"""
        lines = ["Power Law Fit (PSD = A/f^b + n):"]
        
        if 'fitted_parameters' in power_law:
            params = power_law['fitted_parameters']
            lines.append(f"  - Amplitude (A): {params['A']:.6e}")
            lines.append(f"  - Power Law Index (b): {params['b']:.3f}")
            lines.append(f"  - Noise Level (n): {params['n']:.6e}")
        
        return "\n".join(lines) + "\n"
    
    @staticmethod
    def _format_bending_power_law(bending: Dict[str, Any]) -> str:
        """Format bending power law fit results"""
        lines = ["Bending Power Law Fit:"]
        
        if 'fitted_parameters' in bending:
            params = bending['fitted_parameters']
            lines.append(f"  - Amplitude (A): {params['A']:.6e}")
            lines.append(f"  - Break Frequency (fb): {params['fb']:.6e} Hz")
            lines.append(f"  - Shape Parameter (sh): {params['sh']:.3f}")
            lines.append(f"  - Noise Level (n): {params['n']:.6e}")
        
        return "\n".join(lines) + "\n"
    
    @staticmethod
    def _format_metadata(metadata: Dict[str, Any]) -> str:
        """Format file metadata"""
        lines = ["File Information:"]
        
        if 'original_filename' in metadata:
            lines.append(f"  - Filename: {metadata['original_filename']}")
        if 'file_size' in metadata:
            size_mb = metadata['file_size'] / (1024 * 1024)
            lines.append(f"  - File Size: {size_mb:.2f} MB")
        if 'uploaded_at' in metadata:
            lines.append(f"  - Uploaded: {metadata['uploaded_at']}")
        
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