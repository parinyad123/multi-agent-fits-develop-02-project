# app/api/v2/sessions.py

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, desc, func
from typing import Optional
import logging
from uuid import UUID

from app.db.base import get_async_session
from app.db.models import User, Session as ChatSession, ConversationMessage 
from app.core.auth import get_current_active_user
from app.api.v2.models import SessionInfo, SessionListResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/sessions", response_model=SessionListResponse)
async def get_user_sessions(
    offset: int = Query(0, ge=0, description="Number of sessions to skip"),
    limit: int = Query(20, ge=1, le=100, description="Maximum sessions to return"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Get all sessions for the current user.
    Returns lightweight session list sorted by last activity DESC.
    """
    try:
        query = (
            select(ChatSession)
            .where(ChatSession.user_id == current_user.user_id)
            .where(ChatSession.is_active == True)
            .order_by(desc(ChatSession.last_activity_at))
            .offset(offset)
            .limit(limit + 1)
        )
        
        result = await db.execute(query)
        sessions = result.scalars().all()
        
        has_more = len(sessions) > limit
        if has_more:
            sessions = sessions[:limit]
        
        # Convert to response model
        session_infos = []
        for session in sessions:
            # Count messages for this session
            msg_count_query = select(func.count()).select_from(ConversationMessage).where(
                ConversationMessage.session_id == session.session_id
            )
            msg_count_result = await db.execute(msg_count_query)
            message_count = msg_count_result.scalar() or 0
            
            # Use dedicated title field (with fallback)
            title = session.title

            if not title:
                # Fallback 1: metadata (for backward compatibility)
                if session.session_metadata and isinstance(session.session_metadata, dict):
                    title = session.session_metadata.get("title")

                # Fallback 2: default
                if not title:
                    title = "Untitled Session"
            
            session_infos.append(SessionInfo(
                session_id=str(session.session_id),
                title=title,
                created_at=session.created_at.isoformat(),
                last_message_at=session.last_activity_at.isoformat(),
                message_count=message_count
            ))
        
        return SessionListResponse(
            sessions=session_infos,
            total=len(session_infos),
            has_more=has_more,
            next_offset=offset + limit if has_more else None
        )
        
    except Exception as e:
        logger.error(f"Error fetching sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch sessions")


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """Soft delete a session (set is_active=False)"""
    try:
        query = select(ChatSession).where(
            ChatSession.session_id == session_id,
            ChatSession.user_id == current_user.user_id
        )
        result = await db.execute(query)
        session = result.scalar_one_or_none()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Soft delete
        session.is_active = False
        await db.commit()
        
        logger.info(f"Session deleted: user={current_user.user_id}, session={session_id}")
        
        return {
            "success": True,
            "message": "Session deleted successfully",
            "session_id": str(session_id)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {e}", exc_info=True)
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete session")


@router.patch("/sessions/{session_id}/title")
async def update_session_title(
    session_id: UUID,
    title: str = Query(..., min_length=1, max_length=200),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """Update session title (user-defined)"""
    try:
        query = select(ChatSession).where(
            ChatSession.session_id == session_id,
            ChatSession.user_id == current_user.user_id
        )
        result = await db.execute(query)
        session = result.scalar_one_or_none()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Update metadata
        # if not session.session_metadata:
        #     session.session_metadata = {}
        
        # session.session_metadata["title"] = title
        
        # # Mark as modified (for SQLAlchemy to detect change)
        # from sqlalchemy.orm import attributes
        # attributes.flag_modified(session, "session_metadata")
        
        # await db.commit()
        
        # logger.info(
        #     f"Session title updated: user={current_user.user_id}, "
        #     f"session={session_id}, title={title}"
        # )

        # Update dedicated title field
        session.title = title
        session.is_title_user_defined = True    # Mark as user-defined

        await db.commit()

        # logger.into(
        #     f"Session title updated by user: user={current_user.user_id}, "
        #     f"session={session_id}, title='{title}'"
        # )
        
        return {
            "success": True,
            "message": "Session title updated successfully",
            "session_id": str(session_id),
            "title": title
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating session title: {e}", exc_info=True)
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update session title")