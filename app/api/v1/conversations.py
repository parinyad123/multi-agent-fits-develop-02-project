"""
app/api/v1/conversations.py
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID
from typing import List

from app.db.base import get_async_session
from app.services.conversation_service import ConversationService

router = APIRouter()

@router.get("/sessions/{session_id}/conversations")
async def get_conversation_history(
    session_id: str,
    limit: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get conversation history for a session
    
    Returns:
        List of messages in chronological order
    """

    """
    Frontend: Load conversation when user opens chat

    async function loadConversation(sessionId) {
        const response = await fetch(
            `/api/v1/sessions/${sessionId}/conversations?limit=20`
        );
        const data = await response.json();
        
        // Display messages in chat UI
        displayMessages(data.messages);
    }
    """
    
    try:
        messages = await ConversationService.get_conversation_history(
            session=session,
            session_id=session_id,
            limit=limit
        )
        
        return {
            "session_id": session_id,
            "total_messages": len(messages),
            "messages": messages
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workflows/{workflow_id}")
async def get_workflow_details(
    workflow_id: UUID,
    session: AsyncSession = Depends(get_async_session)
):
    """Get complete workflow execution details"""

    """ Frontend: View workflow details

        async function viewWorkflowDetails(workflowId) {
            const response = await fetch(
                `/api/v1/workflows/${workflowId}`
            );
            const data = await response.json();
            
            // Show workflow execution timeline
            showWorkflowTimeline(data.completed_steps);
            
            // Show performance metrics
            showMetrics({
                executionTime: data.execution_time_seconds,
                tokensUsed: data.total_tokens_used,
                cost: data.estimated_cost
            });
        }
    """
    
    workflow = await ConversationService.get_workflow_by_id(
        session=session,
        workflow_id=workflow_id
    )
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return workflow

