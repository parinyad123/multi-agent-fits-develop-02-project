"""
multi-agent-fits-dev-02/app/services/astrosage/conversation_manager.py

Manage conversation history for AstroSage
"""

import logging
from typing import List
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from sqlalchemy.orm import aliased

from app.services.astrosage.models import ConversationPair
from app.db.models import Session as SessionModel

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manage conversation history storage and retrieval
    
    Note: Conversations are stored in sessions.session_metadata JSONB field
    Format: {
        "conversations": [
            {"role": "user", "content": "...", "timestamp": "..."},
            {"role": "assistant", "content": "...", "timestamp": "..."}
        ]
    }
    """
    @staticmethod
    async def get_last_conversations(
        session_id: str,
        db_session: AsyncSession,
        limit: int = 10
    ) -> List[ConversationPair]:
        """
        Get last N conversation pairs from session
        
        Args:
            session_id: Session identifier
            db_session: Database session
            limit: Number of conversation pairs to retrieve (default: 10)
        
        Returns:
            List of ConversationPair objects (oldest first)
        """
        try:
            # Get session record
            result = await db_session.execute(
                select(SessionModel).where(SessionModel.session_id == session_id)
            )
            session_record = result.scalar_one_or_none()

            if not session_record:
                logger.warning(f"Session {session_id} not found")
                return []

            # Get conversations from metadata
            metadata = session_record.session_metadata or {}
            conversations = metadata.get('conversations', [])

            if not conversations:
                logger.info(f"No conversation history for session {session_id}")
                return []
            
            # Convert to ConversationPair objects
            pairs = []

            # Process conversations in pairs (user + assistant)
            for i in range(0, len(conversations) - 1, 2):
                if i + 1 >= len(conversations):
                    break  # Incomplete pair
                
                user_msg = conversations[i]
                assistant_msg = conversations[i + 1]
                
                # Validate roles
                if user_msg.get('role') != 'user' or assistant_msg.get('role') != 'assistant':
                    logger.warning(f"Invalid conversation pair at index {i}")
                    continue
                
                # Parse timestamp
                try:
                    timestamp = datetime.fromisoformat(user_msg.get('timestamp', datetime.now().isoformat()))
                except (ValueError, TypeError):
                    timestamp = datetime.now()
                
                pair = ConversationPair(
                    user_message=user_msg.get('content', ''),
                    assistant_message=assistant_msg.get('content', ''),
                    timestamp=timestamp
                )
                pairs.append(pair)
            
            # Return last N pairs (oldest first)
            result_pairs = pairs[-limit:] if len(pairs) > limit else pairs
            
            logger.info(
                f"Retrieved {len(result_pairs)} conversation pairs "
                f"for session {session_id}"
            )
            
            return result_pairs

        except Exception as e:
            logger.error(
                f"error retrieving conversations for session {session_id}: {e}",
                exc_info=True
            )
            return []
        
    @staticmethod
    async def save_conversation(
        session_id: str,
        user_message: str,
        assistant_message: str,
        db_session: AsyncSession
    ) -> bool:
        """
        Save a conversation pair to session metadata
        
        Args:
            session_id: Session identifier
            user_message: User's message
            assistant_message: Assistant's response
            db_session: Database session
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get session record
            result = await db_session.execute(
                select(SessionModel).where(SessionModel.session_id == session_id)
            )
            session_record = result.scalar_one_or_none()
            
            if not session_record:
                logger.error(f"Session {session_id} not found, cannot save conversation")
                return False
            
            # Initialize metadata if needed
            if session_record.session_metadata is None:
                session_record.session_metadata = {}
            
            # Initialize conversations list
            if 'conversations' not in session_record.session_metadata:
                session_record.session_metadata['conversations'] = []
            
            # Get current timestamp
            timestamp = datetime.now().isoformat()
            
            # Append user message
            session_record.session_metadata['conversations'].append({
                'role': 'user',
                'content': user_message,
                'timestamp': timestamp
            })

            # Append assistant message
            session_record.session_metadata['conversations'].append({
                'role': 'assistant',
                'content': assistant_message,
                'timestamp': timestamp
            })
            
            # Update last activity
            session_record.last_activity_at = datetime.now()
            
            # Flush changes
            await db_session.flush()
            
            logger.info(f"Saved conversation pair to session {session_id}")
            return True
            
        except Exception as e:
            logger.error(
                f"Error saving conversation to session {session_id}: {e}",
                exc_info=True
            )
            return False
        
    @staticmethod
    def format_for_prompt(conversations: List[ConversationPair]) -> str:
        """
        Format conversation history for prompt
        
        Args:
            conversations: List of ConversationPair objects
        
        Returns:
            Formatted string for inclusion in prompt
        """
        if not conversations:
            return ""
        
        lines = ["PREVIOUS CONVERSATION (Last 10 exchanges):\n"]
        
        for pair in conversations:
            # Format timestamp
            time_ago = ConversationManager._format_time_ago(pair.timestamp)
            
            # Add user message
            lines.append(f"[User - {time_ago}]: {pair.user_message}")
            
            # Add assistant message
            lines.append(f"[AstroSage]: {pair.assistant_message}\n")
        
        return "\n".join(lines)

    @staticmethod
    def _format_time_ago(timestamp: datetime) -> str:
        """
        Format timestamp as 'X minutes/hours ago'
        
        Args:
            timestamp: Datetime object
        
        Returns:
            Human-readable time string
        """
        now = datetime.now()
        
        # Handle timezone-aware timestamps
        if timestamp.tzinfo is not None:
            from datetime import timezone
            now = now.replace(tzinfo=timezone.utc)
        
        delta = now - timestamp
        
        seconds = delta.total_seconds()
        
        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        else:
            days = int(seconds / 86400)
            return f"{days} day{'s' if days != 1 else ''} ago"   