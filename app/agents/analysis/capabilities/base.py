"""
multi-agent-fits-dev-02/app/agents/analysis/capabilities/base.py

Base capability interface for all analysis capabilities
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

class AnalysisCapability(ABC):
    """
    Abstract base class for analysis capabilities

    Each capability implements:
    - execute(): Main analysis logic
    - validate_parameters(): Parameter validation
    - get_dependencues(): 
    """

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"capability.{name}")

    @abstractmethod
    async def execute(
        self,
        rate_data: np.ndarray,
        parameters: Dict[str, Any],
        **kwargs
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Execute the analysis
        
        Args:
            rate_data: FITS time series data
            parameters: Analysis parameters (with defaults already applied)
            **kwargs: Additional context (file_record, etc.)
        
        Returns:
            Tuple of (result_dict, plot_url)
            - result_dict: Analysis results
            - plot_url: URL to plot image (None if no plot)
        
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If analysis fails
        """
        pass
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Validate parameters before execution

        Args:
            parameters: Parameters to validate

        Returns:
            True if valid, False otherwise
        """

        return True

    def get_dependencies(self) -> list:
        """
        Get list of analysis types this capability depends on

        Returns:
            List of analysis type name (e.g., ["psd"])
        """
        return []

    def get_default_parameters(self) -> Dict[str, Any]:
        """
        Get default parameters for this capability

        Returns:
            Dictionary of default parameters
        """
        return {}