"""
multi-agent-fits-dev-02/app/agents/analysis/models.py

Models for Analysis Agent - Input and Output structures
"""

from pydantic import BaseModel, Field
from uuid import UUID
from typing import Dict, List, Any, Optional
from datetime import datetime

# ==========================================
# Input Model
# ==========================================

class AnalysisRequest(BaseModel):
    """
    Request model for Analysis Agent
    Receives data from Classification Agent via Orchestrator
    """
    file_id: UUID
    user_id: UUID
    session_id: str
    analysis_types: List[str]  # ["statistics", "psd", "power_law", "bending_power_law", "metadata"]
    parameters: Dict[str, Dict[str, Any]] = Field(default_factory=dict)  # Already has defaults from Classification Agent
    context: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "file_id": "123e4567-e89b-12d3-a456-426614174000",
                "user_id": "789e0123-e89b-12d3-a456-426614174000",
                "session_id": "session_abc123",
                "analysis_types": ["statistics", "psd", "power_law"],
                "parameters": {
                    "statistics": {"metrics": ["mean", "std", "median"]},
                    "psd": {"low_freq": 1e-5, "high_freq": 0.05, "bins": 3500},
                    "power_law": {"A0": 1.0, "b0": 1.0, "noise_bound_percent": 0.7}
                },
                "context": {}
            }
        }

# ==========================================
# Output Models
# ==========================================

class PlotInfo(BaseModel):
    """Information about a generated plot"""
    plot_id: UUID
    plot_type: str  # "psd", "power_law", "bending_power_law"
    plot_url: str  # "/storage/plots/psd/psd_xxx.png"
    created_at: datetime

class AnalysisResult(BaseModel):
    """
    Result model for Analysis Agent
    Returns to Orchestrator with partial results support
    """
    analysis_id: UUID
    file_id: UUID
    status: str  # "completed", "partial", "failed"
    
    # Successful results
    results: Dict[str, Any] = Field(default_factory=dict)
    
    # Failed analyses
    errors: Dict[str, str] = Field(default_factory=dict)
    
    # Plot information
    plots: List[PlotInfo] = Field(default_factory=list)
    
    # Execution metadata
    execution_time: float
    completed_analyses: List[str] = Field(default_factory=list)
    failed_analyses: List[str] = Field(default_factory=list)
    skipped_analyses: List[str] = Field(default_factory=list)
    
    # Timestamps
    started_at: datetime
    completed_at: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "analysis_id": "456e7890-e89b-12d3-a456-426614174000",
                "file_id": "123e4567-e89b-12d3-a456-426614174000",
                "status": "partial",
                "results": {
                    "statistics": {
                        "mean": 123.45,
                        "std": 15.6,
                        "median": 120.0
                    },
                    "metadata": {
                        "file_size": 1024000,
                        "original_filename": "data.fits"
                    }
                },
                "errors": {
                    "psd": "Insufficient data points for binning"
                },
                "plots": [],
                "execution_time": 2.34,
                "completed_analyses": ["statistics", "metadata"],
                "failed_analyses": ["psd"],
                "skipped_analyses": ["power_law"]
            }
        }