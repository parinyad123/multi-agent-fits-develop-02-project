"""
multi-agent-fits-dev-02/app/api/v1/analysis.py
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional
from uuid import uuid4, UUID
from datetime import datetime
from pydantic import BaseModel
import logging

from app.orchestration.orchestrator import (
    DynamicWorkflowOrchestrator,
    UserRequest,
    WorkflowStatus
)

from app.main import get_orchestrator

from app.core.constants import AgentNames

router = APIRouter()
logger = logging.getLogger(__name__)

# =======================================
# Request Models
# =======================================

class ClassificationRequest(BaseModel):
    user_input: str
    context: Optional[Dict[str, Any]] = None

class AnalyzeRequest(BaseModel):
    user_query: str
    fits_file_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

# =======================================
# TEST ENDPOINT - Direct Classification
# =======================================

@router.post("/test/classify")
async def test_classify(
#    user_input: str,
#    context: Optional[Dict[str, Any]] = None,
   request: ClassificationRequest,
   orchestrator: DynamicWorkflowOrchestrator = Depends(get_orchestrator)
):
    """
    Test endpoint - Directly classify input using the classification agent.
    Not part of the main workflow.
    """

    classification_agent = orchestrator.agents.get(AgentNames.CLASSIFICATION)

    if not classification_agent:
        raise HTTPException(status_code=503, detail="Classification agent not available")

    # if context is None:
    #     context = {}
    context = request.context or {}

    try:
        result = await classification_agent.process_request(
            user_input=request.user_input,
            context=context
        )

        return {
            "success": True,
            "classificartion": {
                "primary_intent": result.primary_intent,
                "analysis_types": result.analysis_types,
                "routing_strategy": result.routing_strategy,
                "confidence": result.confidence,
                "question_category": result.question_category,
                "complexity_level": result.complexity_level,
                "reasoning": result.reasoning
            },
            "parameters": result.parameters,
            "parameter_confidence": result.parameter_confidence,
            "parameter_source": result.parameter_source,
            "workflow": {
                "suggested_workflow": result.suggested_workflow,
                "parameter_explanations": result.parameter_explanations,
                "potential_issues": result.potential_issues
            },
            "metadata": {
                "processing_time": result.processing_time,
                "tokens_used": result.tokens_used,
                "cost_estimate": result.cost_estimate,
                "model_used": result.model_used
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =======================================
# FULL WORKFLOW ENDPOINT
# =======================================   

@router.post("/workflow/analyze")
async def submit_analysis(
    request: AnalyzeRequest,
    orchestrator: DynamicWorkflowOrchestrator = Depends(get_orchestrator)
):
    """
    Submit a user request for analysis.
    This will initiate the full multi-agent workflow.
    """

    context = request.context or {}

    # simulate data api from user (frontend)
    user_request = UserRequest(
        user_id = UUID('00000000-0000-0000-0000-000000000001'),
        session_id = str(uuid4()),
        request_id = str(uuid4()),
        fits_file_id = request.fits_file_id,
        user_query = request.user_query,
        context = context,
    )

    try:
        task_id = await orchestrator.submit_request(user_request)

        return {
            "success": True,
            "task_id": task_id,
            "status": "submitted",
            "message": "Request submitted successfully for processing."
        }

    except Exception as e:
        logger.error(f"Error submitting analysis request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analyze/{task_id}")
async def get_analysis_status(
    task_id: str,
    orchestrator: DynamicWorkflowOrchestrator = Depends(get_orchestrator)
):
    """
    Get the status and result of a submitted analysis task.
    Check the status 
    """
    try:
        status = await orchestrator.get_workflow_status(task_id)

        if not status:
            raise HTTPException(status_code=404, detail="Task not found")

        return {
            "task_id": status.task_id,
            "status": status.status,
            "routing_strategy": status.routing_strategy,
            "current_step": status.current_step,
            "progress": status.progress,
            "completed_steps": status.completed_steps,
            "created_at": status.created_at.isoformat() if status.created_at else None,
            "completed_at": status.completed_at.isoformat() if status.completed_at else None,
            "error": status.error,
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error retrieving analysis status for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_classification_stats(
    orchestrator: DynamicWorkflowOrchestrator = Depends(get_orchestrator)
):
    """
    Get statistics on classification Agent.
    """
    classification_agent = orchestrator.agents.get(AgentNames.CLASSIFICATION)

    if not classification_agent:
        raise HTTPException(status_code=503, detail="Classification agent not available")
    
    return classification_agent.get_comprehensive_stats()