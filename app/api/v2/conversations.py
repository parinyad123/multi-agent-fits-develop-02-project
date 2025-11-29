# app/api/v2/conversations.py

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import Optional
import logging
from uuid import UUID

from app.db.base import get_async_session
from app.db.models import User, Session as ChatSession, ConversationMessage
from app.core.auth import get_current_active_user  # ✅ ใช้จาก core.auth
from app.api.v2.models import ConversationResponse, ConversationMessageLight, PlotInfo

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/conversations/{session_id}", response_model=ConversationResponse)
async def get_conversation_history(
    session_id: UUID,  # ✅ เปลี่ยนเป็น UUID
    offset: int = Query(0, ge=0, description="Number of messages to skip"),
    limit: int = Query(50, ge=1, le=100, description="Maximum messages to return"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Get conversation history for a session.
    Returns lightweight messages with plots array (NO raw metadata).
    
    **Pagination:**
    - offset: Skip first N messages
    - limit: Max messages per page (1-100)
    
    **Response:**
    - Sorted chronologically (oldest first)
    - has_more: True if more messages exist
    - next_offset: Use this for next page
    """
    try:
        # ✅ Verify session exists and belongs to user
        session_query = select(ChatSession).where(
            ChatSession.session_id == session_id,
            ChatSession.user_id == current_user.user_id,
            ChatSession.is_active == True  # ✅ เช็ค is_active แทน is_deleted
        )
        session_result = await db.execute(session_query)
        session = session_result.scalar_one_or_none()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # ✅ Query messages with proper ordering
        messages_query = (
            select(ConversationMessage)
            .where(ConversationMessage.session_id == session_id)
            .order_by(ConversationMessage.sequence_number.asc())  # ✅ ใช้ sequence_number
            .offset(offset)
            .limit(limit + 1)  # Get one extra to check has_more
        )
        
        messages_result = await db.execute(messages_query)
        messages = messages_result.scalars().all()
        
        # Check if there are more results
        has_more = len(messages) > limit
        if has_more:
            messages = messages[:limit]
        
        # ✅ Convert to response model
        message_lights = []
        for msg in messages:
            # Extract plots from metadata if exists
            plots = []
            if msg.message_metadata and isinstance(msg.message_metadata, dict):
                if "plots" in msg.message_metadata:
                    plot_list = msg.message_metadata["plots"]
                    if isinstance(plot_list, list):
                        for plot_data in plot_list:
                            try:
                                plots.append(PlotInfo(
                                    plot_id=str(plot_data.get("plot_id", "")),
                                    plot_type=plot_data.get("plot_type", ""),
                                    plot_url=plot_data.get("plot_url", ""),
                                    title=plot_data.get("title", ""),
                                    created_at=plot_data.get("created_at", "")
                                ))
                            except Exception as e:
                                logger.warning(f"Failed to parse plot data: {e}")
                                continue
            
            message_lights.append(ConversationMessageLight(
                message_id=str(msg.message_id),
                role=msg.role,
                content=msg.content,
                created_at=msg.created_at.isoformat(),
                plots=plots if plots else None
            ))
        
        return ConversationResponse(
            session_id=str(session_id),
            messages=message_lights,
            total=len(message_lights),
            has_more=has_more,
            next_offset=offset + limit if has_more else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch conversation history")


@router.get("/conversations/{session_id}/count")
async def get_message_count(
    session_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Get total message count for a session.
    Useful for pagination calculations.
    """
    try:
        # Verify session belongs to user
        session_query = select(ChatSession).where(
            ChatSession.session_id == session_id,
            ChatSession.user_id == current_user.user_id,
            ChatSession.is_active == True
        )
        session_result = await db.execute(session_query)
        session = session_result.scalar_one_or_none()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Count messages
        count_query = select(func.count()).select_from(ConversationMessage).where(
            ConversationMessage.session_id == session_id
        )
        count_result = await db.execute(count_query)
        total_messages = count_result.scalar() or 0
        
        return {
            "session_id": str(session_id),
            "total_messages": total_messages
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error counting messages: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to count messages")