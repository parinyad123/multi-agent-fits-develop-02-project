# app/api/v2/analysis.py

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import logging
import json
import asyncio
from uuid import UUID
from typing import Optional

# ✅ แก้แค่บรรทัดนี้
from app.db.base import get_async_session
from app.db.models import User, Session as ChatSession, FITSFile, WorkflowExecution
from app.core.auth import get_current_active_user  # ✅ ใช้จาก core.auth
from app.api.v2.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    WorkflowStatusLight,
    WorkflowStatusFull,
    AnalysisResultLight,
    PlotInfo
)
from app.main import get_orchestrator
from app.orchestration.orchestrator import UserRequest

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/analyze", response_model=AnalyzeResponse)
async def submit_analysis(
    request: AnalyzeRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Submit analysis request (optimized for V2).
    
    **Flow:**
    1. Validate FITS file (if provided)
    2. Load or create session
    3. Submit to orchestrator
    4. Return task_id for tracking
    
    **Tracking:**
    - Poll: GET /api/v2/analyze/{task_id}/status (lightweight)
    - Stream: GET /api/v2/analyze/{task_id}/stream (SSE)
    - Full: GET /api/v2/analyze/{task_id} (when completed)
    """
    try:
        orchestrator = get_orchestrator()
        
        # ✅ Validate FITS file if provided
        fits_file = None
        if request.fits_file_id:
            # Remove .fits extension if present
            file_id_str = request.fits_file_id.replace('.fits', '').replace('.fit', '')
            
            try:
                file_id = UUID(file_id_str)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid FITS file ID format: {request.fits_file_id}"
                )
            
            query = select(FITSFile).where(
                FITSFile.file_id == file_id,
                FITSFile.user_id == current_user.user_id,
                FITSFile.is_deleted == False
            )
            result = await db.execute(query)
            fits_file = result.scalar_one_or_none()
            
            if not fits_file:
                raise HTTPException(status_code=404, detail="FITS file not found")
            
            if not fits_file.is_valid:
                raise HTTPException(
                    status_code=400,
                    detail=f"FITS file is invalid: {fits_file.validation_error}"
                )
        
        # ✅ Load or create session
        session = None
        if request.session_id:
            try:
                session_id = UUID(request.session_id)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid session ID format: {request.session_id}"
                )
            
            query = select(ChatSession).where(
                ChatSession.session_id == session_id,
                ChatSession.user_id == current_user.user_id,
                ChatSession.is_active == True
            )
            result = await db.execute(query)
            session = result.scalar_one_or_none()
            
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
        
        # Build UserRequest for orchestrator
        user_request = UserRequest(
            user_id=current_user.user_id,
            session_id=session.session_id if session else None,
            request_id=None,  # Will be generated
            fits_file_id=request.fits_file_id,
            user_query=request.query,
            context={
                "user_email": current_user.email,
                "user_expertise": request.user_expertise,
                # "analysis_types": request.analysis_types,
                # "parameters": request.parameters
            }
        )
        
        # Submit to orchestrator
        task_id = await orchestrator.submit_request(user_request)
        
        # Get final session_id
        final_session_id = user_request.session_id
        
        logger.info(
            f"Analysis submitted via V2: user={current_user.user_id}, "
            f"task={task_id}, session={final_session_id}"
        )
        
        return AnalyzeResponse(
            task_id=task_id,
            session_id=str(final_session_id),
            status="queued"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to submit analysis")


@router.get("/analyze/{task_id}/status")
async def get_analysis_status_light(
    task_id: str,
    current_user: User = Depends(get_current_active_user),
    orchestrator = Depends(get_orchestrator)
):
    """Get lightweight status for polling"""
    
    workflow_status = await orchestrator.get_workflow_status(task_id)
    
    if not workflow_status:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    workflow = await orchestrator._get_from_memory(task_id)
    user_request = workflow.get('user_request')
    request_user_id = user_request.get('user_id') if isinstance(user_request, dict) else user_request.user_id
    
    if request_user_id != current_user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return {
        "task_id": task_id,
        "status": workflow_status.status,
        "progress": workflow_status.progress,
        "current_step": workflow_status.current_step,
        "error": workflow_status.error
    }

# @router.get("/analyze/{task_id}", response_model=WorkflowStatusFull)
# async def get_workflow_status_full(
#     task_id: str,
#     current_user: User = Depends(get_current_active_user),
#     db: AsyncSession = Depends(get_async_session)
# ):
#     """
#     Get full workflow status with completed_steps (call once when completed).
    
#     **Use for:**
#     - Viewing complete workflow details
#     - Debugging failed workflows
#     - Analysis result inspection
    
#     **Warning:** Response can be large (5-50 KB with all steps)
    
#     **Response includes:**
#     - All fields from lightweight status
#     - routing_strategy
#     - completed_steps (full history)
#     - timestamps
#     """
#     try:
#         # ✅ Convert string to UUID
#         task_uuid = UUID(task_id)
#     except ValueError:
#         raise HTTPException(status_code=400, detail="Invalid task ID format")
    
#     # ✅ Query with UUID
#     result = await db.execute(
#         select(WorkflowExecution).where(WorkflowExecution.workflow_id == task_uuid)
#     )
#     workflow = result.scalar_one_or_none()
    
#     # Check exists
#     if not workflow:
#         raise HTTPException(status_code=404, detail="Analysis not found")
    
#     # Check ownership
#     if workflow.user_id != current_user.user_id:
#         raise HTTPException(status_code=403, detail="Access denied")
    
#     # Check completed
#     if workflow.status != "completed":
#         raise HTTPException(
#             status_code=400,
#             detail=f"Analysis not completed yet. Status: {workflow.status}"
#         )
    
#     # Extract rewrite step
#     completed_steps = workflow.completed_steps or []
#     rewrite_step = next(
#         (s for s in completed_steps if s.get("step") == "rewrite"),
#         None
#     )
    
#     if not rewrite_step:
#         raise HTTPException(status_code=404, detail="Result not found")
    
#     # Build response
#     return AnalysisResultLight(
#         task_id=str(workflow.workflow_id),
#         status=workflow.status,
#         content=rewrite_step["result"]["content"],
#         plots=[PlotInfo(**plot) for plot in rewrite_step["result"]["plots"]],
#         completed_at=workflow.completed_at.isoformat() if workflow.completed_at else None
    # )

@router.get("/analyze/{task_id}/stream")
async def stream_workflow_status(
    task_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Server-Sent Events (SSE) endpoint for real-time workflow updates.
    
    **Use for:**
    - Real-time progress without polling
    - Live status updates
    - Automatic completion detection
    
    **Connection:**
```javascript
    const eventSource = new EventSource('/api/v2/analyze/{task_id}/stream');
    
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('Status:', data.status, data.progress);
        
        if (data.status === 'completed' || data.status === 'failed') {
            eventSource.close();
        }
    };
```
    
    **Updates sent when:**
    - Status changes
    - Progress updates
    - Step transitions
    - Completion or failure
    """
    async def event_generator():
        orchestrator = get_orchestrator()
        last_status = None
        retry_count = 0
        max_retries = 300  # 5 minutes at 1s intervals
        
        try:
            while retry_count < max_retries:
                # Get current workflow status
                workflow = await orchestrator.get_workflow_status(task_id)
                
                if not workflow:
                    yield f"data: {json.dumps({'error': 'Workflow not found'})}\n\n"
                    break
                
                # Create lightweight status
                current_status = {
                    'task_id': workflow.task_id,
                    'status': workflow.status,
                    'progress': workflow.progress,
                    'current_step': workflow.current_step,
                    'error': workflow.error
                }
                
                # Only send if status changed
                if current_status != last_status:
                    yield f"data: {json.dumps(current_status)}\n\n"
                    last_status = current_status
                    retry_count = 0  # Reset on update
                
                # Stop streaming if completed or failed
                if workflow.status in ['completed', 'failed']:
                    logger.info(f"SSE stream ended: task={task_id}, status={workflow.status}")
                    break
                
                # Wait before next check
                await asyncio.sleep(1)
                retry_count += 1
            
            # Timeout reached
            if retry_count >= max_retries:
                yield f"data: {json.dumps({'error': 'Stream timeout', 'status': 'timeout'})}\n\n"
                
        except asyncio.CancelledError:
            logger.info(f"SSE stream cancelled: task={task_id}")
        except Exception as e:
            logger.error(f"Error in SSE stream: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )

@router.get("/analyze/{task_id}")
async def get_analysis_status(
    task_id: str,
    detail: str = Query(
        "light", 
        description="Response detail level: 'light' (default) or 'full'"
    ),
    current_user: User = Depends(get_current_active_user),
    orchestrator = Depends(get_orchestrator)
):
    """
    Get full workflow details (50-100 KB)
    For debugging or advanced users
    """
    
    # ✅ Get from orchestrator memory
    workflow_status = await orchestrator.get_workflow_status(task_id)
    
    if not workflow_status:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Get full workflow dict
    workflow = await orchestrator._get_from_memory(task_id)
    
    # Check ownership
    user_request = workflow.get('user_request')
    if isinstance(user_request, dict):
        request_user_id = user_request.get('user_id')
    else:
        request_user_id = user_request.user_id
    
    if request_user_id != current_user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Return full workflow details
    return {
        "task_id": task_id,
        "status": workflow_status.status,
        "routing_strategy": workflow_status.routing_strategy,
        "current_step": workflow_status.current_step,
        "progress": workflow_status.progress,
        "completed_steps": workflow.get('completed_steps', []),
        "created_at": workflow_status.created_at.isoformat() if workflow_status.created_at else None,
        "completed_at": workflow_status.completed_at.isoformat() if workflow_status.completed_at else None,
        "error": workflow_status.error
    }
    
@router.get("/analyze/{task_id}/result", response_model=AnalysisResultLight)
async def get_analysis_result_only(
    task_id: str,
    current_user: User = Depends(get_current_active_user),
    orchestrator = Depends(get_orchestrator)
):
    """Get lightweight analysis result"""
    
    workflow_status = await orchestrator.get_workflow_status(task_id)
    
    if not workflow_status:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    workflow = await orchestrator._get_from_memory(task_id)
    
    # Check ownership
    user_request = workflow.get('user_request')
    request_user_id = user_request.get('user_id') if isinstance(user_request, dict) else user_request.user_id
    
    if request_user_id != current_user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if workflow_status.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Analysis not completed yet. Status: {workflow_status.status}"
        )
    
    # Extract rewrite step
    completed_steps = workflow.get('completed_steps', [])
    rewrite_step = next((s for s in completed_steps if s.get('step') == 'rewrite'), None)
    
    if not rewrite_step:
        raise HTTPException(status_code=404, detail="Result not found")
    
    return AnalysisResultLight(
        task_id=task_id,
        status=workflow_status.status,
        content=rewrite_step["result"]["content"],
        plots=[PlotInfo(**plot) for plot in rewrite_step["result"]["plots"]],
        completed_at=workflow_status.completed_at.isoformat() if workflow_status.completed_at else None
    )


@router.delete("/analyze/{task_id}")
async def cancel_workflow(
    task_id: str,
    current_user: User = Depends(get_current_active_user),
    orchestrator = Depends(get_orchestrator)
):
    """
    Cancel a running workflow (if possible).
    
    **Note:** Cancellation may not be immediate if workflow is in progress.
    """
    try:
        workflow_status = await orchestrator.get_workflow_status(task_id)
    
        if not workflow_status:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        workflow = await orchestrator._get_from_memory(task_id)
        user_request = workflow.get('user_request')
        request_user_id = user_request.get('user_id') if isinstance(user_request, dict) else user_request.user_id
        
        if request_user_id != current_user.user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # TODO: Implement actual cancellation logic
        return {
            "message": "Cancellation not yet implemented",
            "task_id": task_id,
            "current_status": workflow_status.status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling workflow: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to cancel workflow")
    
