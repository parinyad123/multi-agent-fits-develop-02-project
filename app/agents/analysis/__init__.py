# app/agents/analysis/__init__.py

"""
Analysis Agent Package

Provides unified analysis capabilities for FITS files including:
- Statistics computation
- Power Spectral Density (PSD) analysis  
- Power law model fitting
- Bending power law model fitting
- Metadata extraction
"""

from app.agents.analysis.unified_analysis_agent import UnifiedAnalysisAgent
from app.agents.analysis.models import AnalysisRequest, AnalysisResult, PlotInfo

__all__ = [
    'UnifiedAnalysisAgent',
    'AnalysisRequest',
    'AnalysisResult',
    'PlotInfo'
]