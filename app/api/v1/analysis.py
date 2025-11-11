"""
Analysis API Endpoints
Full workflow submission and status tracking

multi-agent-fits-dev-02/app/api/v1/analysis.py
"""

from fastapi import APIRouter, HTTPException, Depends, status
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
from app.core.auth import get_current_active_user
from app.db.models import User

router = APIRouter()
logger = logging.getLogger(__name__)

# =======================================
# Request/Response Models
# =======================================

class ClassificationRequest(BaseModel):
    """Request for testing classification agent"""
    user_input: str
    context: Optional[Dict[str, Any]] = None

class AnalyzeRequest(BaseModel):
    """Request for full workflow analysis"""
    session_id: Optional[str] = None
    user_query: str
    fits_file_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class AnalyzeResponse(BaseModel):
    """Response for workflow submission"""
    success: bool
    task_id: str
    user_id: str
    session_id: str
    is_new_session: bool
    status: str
    message: str
    check_status_url: str

class StatusResponse(BaseModel):
    """Response for workflow status"""
    task_id: str
    status: str
    routing_strategy: Optional[str]
    current_step: Optional[str]
    progress: str
    completed_steps: list
    created_at: Optional[str]
    completed_at: Optional[str]
    error: Optional[str]

# =======================================
# TEST ENDPOINT - Direct Classification
# =======================================

@router.post(
    "/test/classify",
    summary="Test Classification Agent",
    description="Directly test the classification agent without full workflow"
)
async def test_classify(
    request: ClassificationRequest,
    current_user: User = Depends(get_current_active_user),
    orchestrator: DynamicWorkflowOrchestrator = Depends(get_orchestrator)
):
    """
    Test endpoint - Directly classify input using the classification agent.
    Not part of the main workflow.

    **Requires Authentication**
    """

    classification_agent = orchestrator.agents.get(AgentNames.CLASSIFICATION)

    if not classification_agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Classification agent not available"
        )
    
    context = request.context or {}
    context['user_id'] = str(current_user.user_id)
    context['user_email'] = current_user.email

    try:
        result = await classification_agent.process_request(
            user_input=request.user_input,
            context=context
        )

        return {
            "success": True,
            "user_id": str(current_user.user_id),
            "classification": {
                "primary_intent": result.primary_intent,
                "analysis_types": result.analysis_types,
                "routing_strategy": result.routing_strategy,
                "confidence": result.confidence,
                "question_category": result.question_category,
                "complexity_level": result.complexity_level,
                "reasoning": result.reasoning,
                "is_mixed_request": result.is_mixed_request,
                "astrosage_required": result.astrosage_required
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
        logger.error(f"Classification test failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification failed: {str(e)}"
        )


# =======================================
# FULL WORKFLOW ENDPOINT
# =======================================   

@router.post("/workflow/analyze")
async def submit_analysis(
    request: AnalyzeRequest,
    current_user: User = Depends(get_current_active_user),  # Require auth
    orchestrator: DynamicWorkflowOrchestrator = Depends(get_orchestrator)
):
    """
    Submit a user request for analysis.
    This will initiate the full multi-agent workflow.
    
    **Requires Authentication**

    **Flow:**
    1. Classification → Determine routing strategy
    2. Route to Analysis Agent and/or AstroSage
    3. Rewrite Agent → Final response
    4. Save all results to database
    
    **Parameters:**
    - **session_id**: Optional session ID (generates new if null for new chat)
    - **user_query**: User's question or analysis request
    - **fits_file_id**: Optional FITS file UUID (required for analysis workflows)
    - **context**: Optional additional context
    
    **Returns:**
    - **task_id**: Use this to check status at `/api/v1/analyze/{task_id}`
    - **session_id**: Use this for continuing the conversation
    - **is_new_session**: Whether this is a new chat session
    """

    # Prepare context
    context = request.context or {}

    # Add user info to context
    context['user_id'] = str(current_user.user_id)
    context['user_email'] = current_user.email

    # Generate session_id if null (new chat)
    session_id = request.session_id or str(uuid4())
    is_new_session = request.session_id is None

    # Build user request
    user_request = UserRequest(
        user_id=current_user.user_id,  # From JWT token (secure)
        session_id=request.session_id,  # Will be generated in orchestrator if None
        request_id=str(uuid4()),
        fits_file_id=request.fits_file_id,
        user_query=request.user_query,
        context=context
    )

    try:
        # Submit to orchestrator
        task_id = await orchestrator.submit_request(user_request)

        # Get final session_id (might be generated)
        final_session_id = user_request.session_id
        is_new_session = request.session_id is None

        logger.info(
            f"Analysis submitted: user={current_user.user_id}, "
            f"task={task_id}, session={final_session_id}, "
            f"is_new={is_new_session}"
        )

        return AnalyzeResponse(
            success=True,
            task_id=task_id,
            user_id=str(current_user.user_id),
            session_id=final_session_id,
            is_new_session=is_new_session,
            status="submitted",
            message="Request submitted successfully for processing.",
            check_status_url=f"/api/v1/analyze/{task_id}"
        )

    except ValueError as e:
        # Validation errors
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}"
        )
        
    except RuntimeError as e:
        # Runtime errors (session/workflow creation)
        logger.error(f"Runtime error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process request: {str(e)}"
        )
        
    except Exception as e:
        # Unexpected errors
        logger.error(f"Unexpected error submitting analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your request."
        )

@router.get(
    "/analyze/{task_id}",
    response_model=StatusResponse,
    summary="Get Workflow Status",
    description="Check status and results of submitted analysis task"
)
async def get_analysis_status(
    task_id: str,
    current_user: User = Depends(get_current_active_user),
    orchestrator: DynamicWorkflowOrchestrator = Depends(get_orchestrator)
):
    """
    Get the status and result of a submitted analysis task.
    
    **Requires Authentication**
    
    **Status Values:**
    - `queued`: Waiting in queue
    - `in_progress`: Currently processing
    - `completed`: Finished successfully
    - `failed`: Encountered error
    
    **Workflow Steps:**
    - `classification`: Analyzing user intent
    - `analysis`: Running FITS analysis
    - `astrosage`: Querying astronomy LLM
    - `rewrite`: Formatting final response
    
    Note: Users can only view their own tasks (optional: add ownership check)
    """
    try:
        workflow_status = await orchestrator.get_workflow_status(task_id)

        if not workflow_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task not found: {task_id}"
            )

        logger.info(f"Status check: user={current_user.user_id}, task={task_id}")

        return StatusResponse(
            task_id=workflow_status.task_id,
            status=workflow_status.status,
            routing_strategy=workflow_status.routing_strategy,
            current_step=workflow_status.current_step,
            progress=workflow_status.progress,
            completed_steps=workflow_status.completed_steps,
            created_at=workflow_status.created_at.isoformat() if workflow_status.created_at else None,
            completed_at=workflow_status.completed_at.isoformat() if workflow_status.completed_at else None,
            error=workflow_status.error
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error retrieving status for task {task_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve task status: {str(e)}"
        )

@router.get(
    "/stats",
    summary="Get Classification Statistics",
    description="Get performance statistics for the classification agent"
)
async def get_classification_stats(
    current_user: User = Depends(get_current_active_user),
    orchestrator: DynamicWorkflowOrchestrator = Depends(get_orchestrator)
):
    """
    Get statistics on Classification Agent performance.
    
    **Requires Authentication**
    
    Returns metrics like:
    - Total requests processed
    - Average processing time
    - Confidence distribution
    - Routing strategy breakdown
    """
    
    classification_agent = orchestrator.agents.get(AgentNames.CLASSIFICATION)

    if not classification_agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Classification agent not available"
        )
    
    try:
        stats = classification_agent.get_comprehensive_stats()
        return {
            "success": True,
            "user_id": str(current_user.user_id),
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error retrieving stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve statistics: {str(e)}"
        )