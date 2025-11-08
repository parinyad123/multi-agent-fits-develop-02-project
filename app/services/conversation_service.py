"""
app/services/conversation_service.py
"""

from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from sqlalchemy.orm import selectinload

from app.db.models import ConversationMessage, WorkflowExecution, Session as SessionModel
from app.core.constants import WorkflowStatusType

logger = logging.getLogger(__name__)

class ConversationService:
    """Manage conversation history and workflow tracking"""

    async def create_workflow_execution(
        session: AsyncSession,
        user_id: UUID,
        session_id: str,
        user_query: str,
        request_context: Dict[str, Any],
        file_id: Optional[UUID] = None
    ) -> UUID:
        """
        Create a new workflow execution record
        
        Returns:
            workflow_id
        """
        
        workflow = WorkflowExecution(
            user_id=user_id,
            session_id=session_id,
            file_id=file_id,
            user_query=user_query,
            request_context=request_context,
            status=WorkflowStatusType.PENDING,
            started_at=datetime.now()
        )
        
        session.add(workflow)
        await session.flush()
        
        logger.info(f"Created workflow execution: {workflow.workflow_id}")
        
        return workflow.workflow_id
    
    @staticmethod
    async def update_workflow_status(
        session: AsyncSession,
        workflow_id: UUID,
        status: str,
        current_step: Optional[str] = None,
        progress: Optional[str] = None,
        error: Optional[str] = None
    ):
        """Update workflow execution status"""
        
        result = await session.execute(
            select(WorkflowExecution).where(WorkflowExecution.workflow_id == workflow_id)
        )
        workflow = result.scalar_one_or_none()
        
        if not workflow:
            logger.error(f"Workflow not found: {workflow_id}")
            return
        
        workflow.status = status
        if current_step:
            workflow.current_step = current_step
        if progress:
            workflow.progress = progress
        if error:
            workflow.error = error

            if status == WorkflowStatusType.COMPLETED:
                workflow.completed_at = datetime.now()
            if workflow.started_at:
                duration = (workflow.completed_at - workflow.started_at).total_seconds()
                workflow.execution_time_seconds = int(duration)
        
        await session.flush()

    @staticmethod
    async def save_workflow_results(
        session: AsyncSession,
        workflow_id: UUID,
        completed_steps: List[Dict[str, Any]],
        routing_strategy: str,
        analysis_id: Optional[UUID] = None,
        total_tokens: int = 0,
        estimated_cost: float = 0.0
    ):
        """Update workflow execution status"""

        result = await session.execute(
            select(WorkflowExecution).where(WorkflowExecution.workflow_id == workflow_id)
        )
        workflow = result.scalar_one_or_none()

        if not workflow:
            logger.error(f"Workflow not found: {workflow_id}")
            return
        
        workflow.completed_steps = completed_steps
        workflow.routing_strategy = routing_strategy
        workflow.analysis_id = analysis_id
        workflow.total_tokens_used = total_tokens
        workflow.estimated_cost = estimated_cost

        await session.flush()

    @staticmethod
    async def save_user_message(
        session: AsyncSession,
        session_id: str,
        user_id: UUID,
        workflow_id: UUID,
        content: str
    ) -> UUID:
        """Save user message"""

        # Get next sequence number
        result = await session.execute(
            select(func.max(ConversationMessage.sequence_number))
            .where(ConversationMessage.session_id == session_id)
        )
        max_seq = result.scalar() or 0

        message = ConversationMessage(
            session_id=session_id,
            workflow_id=workflow_id,
            user_id=user_id,
            role="user",
            content=content,
            sequence_number=max_seq + 1
        )

        session.add(message)
        await session.flush()

        logger.debug(f"Saved user message: {message.message_id}")

        return message.message_id
    
    @staticmethod
    async def save_assistant_message(
        session: AsyncSession,
        session_id: str,
        user_id: UUID,
        workflow_id: UUID,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UUID:
        """Save assistant response"""
        
        # Get next sequence number
        result = await session.execute(
            select(func.max(ConversationMessage.sequence_number))
            .where(ConversationMessage.session_id == session_id)
        )
        max_seq = result.scalar() or 0
        
        message = ConversationMessage(
            session_id=session_id,
            workflow_id=workflow_id,
            user_id=user_id,
            role="assistant",
            content=content,
            sequence_number=max_seq + 1,
            message_metadata=metadata or {}
        )
        
        session.add(message)
        await session.flush()
        
        logger.debug(f"Saved assistant message: {message.message_id}")
        
        return message.message_id

    @staticmethod
    async def get_conversation_history(
        session: AsyncSession,
        session_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""

        result = await session.execute(
            select(ConversationMessage)
            .where(ConversationMessage.session_id == session_id)
            .order_by(ConversationMessage.sequence_number.desc())
            .limit(limit)
        )

        messages = result.scalars().all()
        
        # Reverse to get chronological order
        messages = list(reversed(messages))
        
        return [
            {
                "message_id": str(msg.message_id),
                "role": msg.role,
                "content": msg.content,
                "metadata": msg.message_metadata,
                "created_at": msg.created_at.isoformat()
            }
            for msg in messages
        ]
    
    @staticmethod
    async def get_workflow_by_id(
        session: AsyncSession,
        workflow_id: UUID
    ) -> Optional[Dict[str, Any]]:
        """Get workflow execution by ID"""
        
        result = await session.execute(
            select(WorkflowExecution)
            .where(WorkflowExecution.workflow_id == workflow_id)
        )
        
        workflow = result.scalar_one_or_none()
        
        if not workflow:
            return None
        
        return {
            "workflow_id": str(workflow.workflow_id),
            "user_query": workflow.user_query,
            "routing_strategy": workflow.routing_strategy,
            "status": workflow.status,
            "completed_steps": workflow.completed_steps,
            "created_at": workflow.created_at.isoformat(),
            "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None,
            "execution_time_seconds": workflow.execution_time_seconds
        }