"""
app/agents/rewrite/models.py

Data models for Rewrite Agent
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime


class RewriteRequest(BaseModel):
    """
    Input to Rewrite Agent
    
    Contains all information needed to rewrite the response:
    - User's original query
    - Routing strategy (analysis/astrosage/mixed)
    - Results from all previous agents
    - User expertise level
    """
    user_query: str = Field(..., description="Original user query")
    routing_strategy: str = Field(..., description="Workflow routing strategy")
    completed_steps: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Results from Classification, Analysis, AstroSage agents"
    )
    expertise_level: str = Field(
        default="intermediate",
        description="User expertise level: beginner, intermediate, advanced, expert"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_query": "Fit power law and bending power law with initial amplitude 2",
                "routing_strategy": "mixed",
                "completed_steps": [
                    {
                        "step": "classification",
                        "classification_result": {
                            "primary_intent": "mixed",
                            "analysis_types": ["power_law", "bending_power_law"],
                            "routing_strategy": "mixed",
                            "confidence": 0.95
                        }
                    },
                    {
                        "step": "analysis",
                        "analysis_result": {
                            "status": "completed",
                            "results": {
                                "power_law": {
                                    "fitted_parameters": {
                                        "A": 2158.57,
                                        "b": 0.809,
                                        "n": 48007.69
                                    }
                                }
                            }
                        }
                    },
                    {
                        "step": "astrosage",
                        "response": "The power law fit shows...",
                        "success": True
                    }
                ],
                "expertise_level": "intermediate"
            }
        }


class RewriteResponse(BaseModel):
    """
    Output from Rewrite Agent
    
    Contains the final formatted response with metadata
    """
    final_response: str = Field(..., description="Final formatted response in markdown")
    summary: Optional[str] = Field(None, description="2-3 sentence executive summary")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the response (word count, LaTeX validation, etc.)"
    )
    validation_passed: bool = Field(
        default=True,
        description="Whether LaTeX validation passed"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Any warnings during processing"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "final_response": "# Analysis Summary\n\n**I calculated** the power law fit...",
                "summary": "Power law fit yielded b=0.809, indicating red noise variability.",
                "metadata": {
                    "word_count": 1234,
                    "has_latex": True,
                    "latex_validated": True,
                    "model_used": "gpt-4o-mini",
                    "response_time": 2.34,
                    "tokens_used": 850
                },
                "validation_passed": True,
                "warnings": []
            }
        }


class PlotInfo(BaseModel):
    """Information about a plot generated during analysis"""
    plot_id: UUID
    plot_type: str  # "psd", "power_law", "bending_power_law"
    plot_url: str
    created_at: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "plot_id": "7e7eed4f-d81d-414c-bcf7-0df9f6f18eff",
                "plot_type": "power_law",
                "plot_url": "storage/plots/power_law/power_law_70dcfb23.png",
                "created_at": "2025-11-03T14:26:31.772123"
            }
        }