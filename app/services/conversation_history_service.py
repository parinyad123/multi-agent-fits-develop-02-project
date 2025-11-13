"""
Conversation History Service
Centralized service for loading and formatting conversation history

Queries from:
- conversation_messages: For conversation context
- analysis_history: For parameters and results

app/services/conversation_history_service.py
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import UUID
import logging

from sqlalchemy import select, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ConversationMessage, AnalysisHistory

logger = logging.getLogger(__name__)


class ConversationMessageDTO:
    """
    Structured conversation message (Data Transfer Object)
    Represents a message for in-memory processing
    """
    
    def __init__(
        self,
        message_id: UUID,
        role: str,
        content: str,
        created_at: datetime,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.message_id = message_id
        self.role = role  # 'user' or 'assistant'
        self.content = content
        self.created_at = created_at
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "message_id": str(self.message_id),
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }
    
    def to_openai_format(self) -> Dict[str, str]:
        """Convert to OpenAI chat format"""
        return {
            "role": self.role,
            "content": self.content
        }
    
    def __repr__(self) -> str:
        return f"<Message {self.role}: {self.content[:50]}...>"
    

class ConversationHistoryService:
    """
    Centralized service for conversation history management v2.1
    
    Features:
    - Load recent messages from conversation_messages
    - Extract parameters from analysis_history with BACKWARD SEARCH
    - Extract results from analysis_history
    - Format for different agents
    - Optimize token usage
    """
    
    # Token estimation (rough approximation)
    CHARS_PER_TOKEN = 4  # Average: 1 token ≈ 4 characters
    
    @staticmethod
    async def get_recent_messages(
        session: AsyncSession,
        session_id: str,
        limit: int = 20,
        include_system: bool = False,
        max_tokens: Optional[int] = None
    ) -> List[ConversationMessageDTO]:
        """
        Get recent messages in chronological order (oldest first)
        
        Args:
            session: Database session
            session_id: Session identifier
            limit: Maximum number of messages (default: 20)
            include_system: Include system messages (default: False)
            max_tokens: Maximum tokens to load (optional truncation)
            
        Returns:
            List of ConversationMessage objects in chronological order
            
        Example:
            messages = await get_recent_messages(session, "sess-123", limit=10)
            for msg in messages:
                print(f"{msg.role}: {msg.content}")
        """
        
        try:
            # Build query
            query = select(ConversationMessage).where(
                ConversationMessage.session_id == session_id
            )
            
            # Filter out system messages if needed
            if not include_system:
                query = query.where(ConversationMessage.role != 'system')
            
            # Order by sequence_number DESC, then limit
            query = query.order_by(desc(ConversationMessage.sequence_number)).limit(limit)
            
            # Execute query
            result = await session.execute(query)
            db_messages = result.scalars().all()
            
            # Convert to ConversationMessage objects
            messages = [
                ConversationMessageDTO(
                    message_id=msg.message_id,
                    role=msg.role,
                    content=msg.content,
                    created_at=msg.created_at,
                    metadata=msg.message_metadata or {}
                )
                for msg in db_messages
            ]
            
            # Reverse to get chronological order (oldest first)
            messages.reverse()
            
            # Apply token limit if specified
            if max_tokens:
                messages = ConversationHistoryService._truncate_by_tokens(
                    messages, 
                    max_tokens
                )
            
            logger.info(
                f"Loaded {len(messages)} messages for session {session_id} "
                f"(requested: {limit})"
            )
            
            return messages
            
        except Exception as e:
            logger.error(f"Failed to load messages for session {session_id}: {e}", exc_info=True)
            return []
        
    @staticmethod
    async def get_last_parameters(
        session: AsyncSession,
        session_id: str,
        file_id: Optional[UUID] = None,
        analysis_type: Optional[str] = None,
        scope: str = "session",
        search_depth: int = 10
    ) -> Dict[str, Any]:
        """
        Extract last used parameters from analysis_history with BACKWARD SEARCH
        
        - Searches through multiple analyses (backward search)
        - File-aware querying (scope: "session" or "file")
        - Returns first match found
        
        Args:
            session: Database session
            session_id: Session identifier
            file_id: Optional file UUID for file-specific queries
            analysis_type: Optional filter by specific analysis type
            scope: Query scope - "session" (all files) or "file" (specific file only)
            search_depth: Number of recent analyses to search (default: 10)
            
        Returns:
            Dictionary of parameters with metadata:
            {
                "power_law": {
                    "A0": 1.0,
                    "b0": 1.0,
                    ...
                },
                "_metadata": {
                    "analysis_id": "uuid-123",
                    "timestamp": "2025-11-10T10:00:00Z",
                    "source": "analysis_history",
                    "search_position": 1  # Found in 1st analysis
                }
            }
            
        Scope Behavior:
            - "session": Search across all files in session (default)
            - "file": Search only within specified file_id
            
        Example 1 - Find specific type across session:
            params = await get_last_parameters(
                session, "sess-123",
                analysis_type="power_law",
                scope="session",
                search_depth=10
            )
            
        Example 2 - Find any parameters for specific file:
            params = await get_last_parameters(
                session, "sess-123",
                file_id=UUID("file-456"),
                scope="file"
            )
            
        Example 3 - Backward search scenario:
            History:
              Analysis 1: power_law @ 10:00
              Analysis 2: bending_power_law @ 10:05
              Analysis 3: metadata @ 10:10 (latest, no power_law)
            
            Query: analysis_type="power_law"
            Result: Returns Analysis 1 parameters 
        """
        
        try:
            # ========================================
            # STEP 1: Build query with scope filter
            # ========================================
            query = select(AnalysisHistory).where(
                AnalysisHistory.session_id == session_id
            )
            
            # Apply file scope filter
            if scope == "file":
                if not file_id:
                    logger.warning(
                        f"scope='file' specified but no file_id provided. "
                        f"Falling back to session scope."
                    )
                else:
                    query = query.where(AnalysisHistory.file_id == file_id)
                    logger.debug(
                        f"Applying file filter: file_id={file_id}, scope={scope}"
                    )
            
            # Order by most recent first and limit search depth
            query = query.order_by(
                desc(AnalysisHistory.completed_at)
            ).limit(search_depth)
            
            # ========================================
            # STEP 2: Execute query
            # ========================================
            result = await session.execute(query)
            analyses = result.scalars().all()
            
            if not analyses:
                logger.debug(
                    f"No analysis history found for session {session_id} "
                    f"(scope={scope}, file_id={file_id})"
                )
                return {}
            
            logger.info(
                f"Searching {len(analyses)} analyses for parameters "
                f"(session_id={session_id}, analysis_type={analysis_type or 'any'}, "
                f"scope={scope})"
            )
            
            # ========================================
            # STEP 3: Backward search through analyses
            # ========================================
            for position, analysis in enumerate(analyses, start=1):
                parameters = analysis.parameters or {}
                
                if not parameters:
                    logger.debug(
                        f"Position {position}: Analysis {analysis.analysis_id} "
                        f"has no parameters, skipping"
                    )
                    continue
                
                # Filter by analysis_type if specified
                if analysis_type:
                    if analysis_type in parameters:
                        # FOUND! Return specific type parameters
                        logger.info(
                            f"Found {analysis_type} parameters at position {position} "
                            f"(analysis_id={analysis.analysis_id}, "
                            f"timestamp={analysis.completed_at})"
                        )
                        
                        return {
                            analysis_type: parameters[analysis_type],
                            "_metadata": {
                                "analysis_id": str(analysis.analysis_id),
                                "timestamp": analysis.completed_at.isoformat(),
                                "analysis_type": analysis_type,
                                "source": "analysis_history",
                                "scope": scope,
                                "file_id": str(analysis.file_id) if analysis.file_id else None,
                                "search_position": position,
                                "search_depth": len(analyses)
                            }
                        }
                    else:
                        logger.debug(
                            f"Position {position}: Analysis {analysis.analysis_id} "
                            f"has parameters {list(parameters.keys())} "
                            f"but not '{analysis_type}', continuing search"
                        )
                else:
                    # FOUND! Return all available parameters
                    logger.info(
                        f"Found parameters at position {position} "
                        f"(analysis_id={analysis.analysis_id}, "
                        f"types={list(parameters.keys())})"
                    )
                    
                    result_params = {**parameters}
                    result_params["_metadata"] = {
                        "analysis_id": str(analysis.analysis_id),
                        "timestamp": analysis.completed_at.isoformat(),
                        "analysis_types": analysis.analysis_types,
                        "source": "analysis_history",
                        "scope": scope,
                        "file_id": str(analysis.file_id) if analysis.file_id else None,
                        "search_position": position,
                        "search_depth": len(analyses)
                    }
                    
                    return result_params
            
            # ========================================
            # STEP 4: No match found
            # ========================================
            logger.info(
                f"No matching parameters found after searching {len(analyses)} analyses "
                f"(session_id={session_id}, analysis_type={analysis_type or 'any'}, "
                f"scope={scope})"
            )
            
            return {}
            
        except Exception as e:
            logger.error(
                f"Failed to extract parameters for session {session_id}: {e}", 
                exc_info=True
            )
            return {}

    @staticmethod
    async def get_last_analysis_results(
        session: AsyncSession,
        session_id: str,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get recent analysis results from analysis_history
        
        Args:
            session: Database session
            session_id: Session identifier
            limit: Number of recent results (default: 3)
            
        Returns:
            List of analysis results with metadata
            [
                {
                    "analysis_id": "uuid-123",
                    "timestamp": "2025-11-10T10:00:00Z",
                    "analysis_types": ["power_law", "psd"],
                    "results": {
                        "completed": {
                            "power_law": {
                                "fitted_parameters": {...}
                            }
                        },
                        "errors": {}
                    },
                    "status": "completed"
                }
            ]
            
        Example:
            results = await get_last_analysis_results(session, "sess-123")
            for result in results:
                print(f"Analysis: {result['analysis_types']}")
                print(f"Results: {result['results']}")
        """
        
        try:
            # Query recent analyses for this session
            query = select(AnalysisHistory).where(
                AnalysisHistory.session_id == session_id
            ).order_by(
                desc(AnalysisHistory.completed_at)
            ).limit(limit)
            
            result = await session.execute(query)
            analyses = result.scalars().all()
            
            analysis_results = []
            for analysis in analyses:
                # Extract results from JSONB
                results = analysis.results or {}
                
                analysis_results.append({
                    "analysis_id": str(analysis.analysis_id),
                    "timestamp": analysis.completed_at.isoformat(),
                    "analysis_types": analysis.analysis_types,
                    "results": results,  # {"completed": {...}, "errors": {...}}
                    "status": analysis.status,
                    "execution_time": analysis.execution_time_seconds
                })
            
            logger.info(
                f"Found {len(analysis_results)} analysis results "
                f"in session {session_id}"
            )
            
            return analysis_results
            
        except Exception as e:
            logger.error(
                f"Failed to load analysis results for session {session_id}: {e}",
                exc_info=True
            )
            return []

    @staticmethod
    def format_for_classification(
        messages: List[ConversationMessageDTO],
        last_parameters: Optional[Dict[str, Any]] = None,
        max_tokens: int = 2000
    ) -> str:
        """
        Format conversation history for Classification Agent
        
        Focus on:
        - User queries
        - Analysis parameters (from analysis_history)
        - Previous analysis types
        
        Args:
            messages: List of ConversationMessage objects
            last_parameters: Parameters from get_last_parameters()
            max_tokens: Maximum tokens for context (default: 2000)
            
        Returns:
            Formatted string for Classification Agent prompt
            
        Example:
            messages = await get_recent_messages(session, session_id)
            params = await get_last_parameters(session, session_id)
            context = format_for_classification(messages, params)
            
            prompt = f'''
                Previous conversation:
                {context}

                Current query: {user_query}

                Task: Extract parameters, using previous parameters as defaults.
            '''
        """
        
        if not messages and not last_parameters:
            return "No previous conversation or parameters."
        
        # Truncate messages if needed
        if messages:
            messages = ConversationHistoryService._truncate_by_tokens(
                messages,
                max_tokens
            )
        
        formatted_parts = []
        formatted_parts.append("=== PREVIOUS CONVERSATION ===\n")
        
        # Add messages
        if messages:
            for msg in messages:
                # Format based on role
                if msg.role == 'user':
                    formatted_parts.append(f"User: {msg.content}")
                
                elif msg.role == 'assistant':
                    # Summarize assistant response
                    content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                    formatted_parts.append(f"Assistant: {content_preview}")
                    
                    # Add routing strategy if available
                    routing = msg.metadata.get('routing_strategy')
                    if routing:
                        formatted_parts.append(f"  → Routing: {routing}")
                
                formatted_parts.append("")  # Empty line between messages
        else:
            formatted_parts.append("(No previous messages)\n")
        
        # Add last used parameters
        if last_parameters:
            formatted_parts.append("\n=== LAST USED PARAMETERS ===\n")
            
            metadata = last_parameters.get("_metadata", {})
            if metadata:
                timestamp = metadata.get("timestamp", "unknown")
                search_position = metadata.get("search_position")
                formatted_parts.append(f"Source: analysis_history")
                formatted_parts.append(f"Timestamp: {timestamp}")
                if search_position:
                    formatted_parts.append(f"Search Position: {search_position}")
                formatted_parts.append("")
            
            # Format parameters by analysis type
            for key, value in last_parameters.items():
                if key.startswith("_"):
                    continue  # Skip metadata
                
                formatted_parts.append(f"{key.upper()}:")
                if isinstance(value, dict):
                    for param_name, param_value in value.items():
                        formatted_parts.append(f"  - {param_name}: {param_value}")
                else:
                    formatted_parts.append(f"  {value}")
                formatted_parts.append("")
        
        formatted_parts.append("=== END HISTORY ===")
        
        return "\n".join(formatted_parts)

    @staticmethod
    def format_for_astrosage(
        messages: List[ConversationMessageDTO],
        last_results: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4000,
        include_analysis_results: bool = True
    ) -> List[Dict[str, str]]:
        """
        Format conversation history for AstroSage (OpenAI format)
        
        Args:
            messages: List of ConversationMessage objects
            last_results: Analysis results from get_last_analysis_results()
            max_tokens: Maximum tokens for context (default: 4000)
            include_analysis_results: Include analysis results in context
            
        Returns:
            List of messages in OpenAI format:
            [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
                ...
            ]
            
        Example:
            messages = await get_recent_messages(session, session_id)
            results = await get_last_analysis_results(session, session_id)
            history = format_for_astrosage(messages, results)
            
            payload = {
                "model": "astrosage",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *history,  # ← Include conversation history
                    {"role": "user", "content": current_query}
                ]
            }
        """
        
        if not messages:
            return []
        
        # Truncate if needed (reserve some tokens for results summary)
        reserved_for_results = 1000 if (include_analysis_results and last_results) else 0
        messages = ConversationHistoryService._truncate_by_tokens(
            messages,
            max_tokens - reserved_for_results
        )
        
        formatted_messages = []
        
        for msg in messages:
            content = msg.content
            
            # Enhance assistant messages with routing info
            if msg.role == 'assistant':
                routing = msg.metadata.get('routing_strategy')
                if routing:
                    content += f"\n\n[Routing: {routing}]"
            
            formatted_messages.append({
                "role": msg.role,
                "content": content
            })
        
        # Append analysis results summary if available
        if include_analysis_results and last_results:
            results_summary = ConversationHistoryService._format_results_summary(
                last_results
            )
            if results_summary:
                formatted_messages.append({
                    "role": "system",
                    "content": results_summary
                })
        
        logger.debug(
            f"Formatted {len(formatted_messages)} messages for AstroSage "
            f"(~{ConversationHistoryService._estimate_tokens_from_messages(formatted_messages)} tokens)"
        )
        
        return formatted_messages

    @staticmethod
    def _format_results_summary(results: List[Dict[str, Any]]) -> str:
        """Format analysis results into a concise summary"""
        
        if not results:
            return ""
        
        summary_parts = []
        summary_parts.append("=== RECENT ANALYSIS RESULTS ===\n")
        
        for i, result in enumerate(results, 1):
            analysis_types = result.get("analysis_types", [])
            timestamp = result.get("timestamp", "unknown")
            completed_results = result.get("results", {}).get("completed", {})
            
            summary_parts.append(f"Analysis {i} ({', '.join(analysis_types)}):")
            summary_parts.append(f"  Timestamp: {timestamp}")
            
            # Summarize key results
            for analysis_type, data in completed_results.items():
                if analysis_type == "power_law":
                    fitted = data.get("fitted_parameters", {})
                    A = fitted.get("A")
                    b = fitted.get("b")
                    if A and b:
                        summary_parts.append(f"  - Power Law: A={A:.3e}, b={b:.3f}")
                
                elif analysis_type == "bending_power_law":
                    fitted = data.get("fitted_parameters", {})
                    fb = fitted.get("fb")
                    sh = fitted.get("sh")
                    if fb and sh:
                        summary_parts.append(f"  - Bending Power Law: fb={fb:.3e}, sh={sh:.3f}")
                
                elif analysis_type == "statistics":
                    stats = data.get("statistics", {})
                    mean = stats.get("mean")
                    std = stats.get("std")
                    if mean and std:
                        summary_parts.append(f"  - Statistics: mean={mean:.3f}, std={std:.3f}")
            
            summary_parts.append("")
        
        summary_parts.append("=== END RESULTS SUMMARY ===")
        
        return "\n".join(summary_parts)

    @staticmethod
    def _truncate_by_tokens(
        messages: List[ConversationMessageDTO],
        max_tokens: int
    ) -> List[ConversationMessageDTO]:
        """
        Truncate messages to fit within token limit
        Keeps most recent messages
        
        Args:
            messages: List of messages (oldest first)
            max_tokens: Maximum tokens
            
        Returns:
            Truncated list of messages
        """
        
        if not messages:
            return []
        
        # Estimate tokens for each message
        message_tokens = [
            ConversationHistoryService._estimate_message_tokens(msg)
            for msg in messages
        ]
        
        # Start from most recent and work backwards
        selected_messages = []
        total_tokens = 0
        
        for msg, tokens in zip(reversed(messages), reversed(message_tokens)):
            if total_tokens + tokens > max_tokens:
                break
            selected_messages.insert(0, msg)  # Insert at beginning
            total_tokens += tokens
        
        if len(selected_messages) < len(messages):
            logger.info(
                f"Truncated history: {len(selected_messages)}/{len(messages)} messages "
                f"(~{total_tokens} tokens)"
            )
        
        return selected_messages
    
    @staticmethod
    def _estimate_message_tokens(message: ConversationMessageDTO) -> int:
        """Estimate tokens for a single message"""
        content_tokens = len(message.content) // ConversationHistoryService.CHARS_PER_TOKEN
        
        # Add overhead for role, metadata
        overhead = 10
        
        return content_tokens + overhead
    
    @staticmethod
    def _estimate_tokens_from_messages(messages: List[Dict[str, str]]) -> int:
        """Estimate total tokens for OpenAI format messages"""
        total = 0
        for msg in messages:
            content = msg.get('content', '')
            total += len(content) // ConversationHistoryService.CHARS_PER_TOKEN + 10
        return total
    
    @staticmethod
    async def get_context_summary(
        session: AsyncSession,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Get comprehensive context summary for a session
        
        Returns:
            {
                "total_messages": 15,
                "last_activity": "2025-11-10T10:30:00Z",
                "analysis_history": ["power_law", "bending_power_law"],
                "last_parameters": {...},
                "last_results": {...}
            }
        """
        
        try:
            # Get message count
            count_query = select(ConversationMessage).where(
                ConversationMessage.session_id == session_id
            )
            result = await session.execute(count_query)
            total_messages = len(result.scalars().all())
            
            # Get last activity
            last_msg_query = select(ConversationMessage).where(
                ConversationMessage.session_id == session_id
            ).order_by(desc(ConversationMessage.sequence_number)).limit(1)
            
            result = await session.execute(last_msg_query)
            last_msg = result.scalar_one_or_none()
            
            last_activity = last_msg.created_at if last_msg else None
            
            # Get analysis history
            analysis_results = await ConversationHistoryService.get_last_analysis_results(
                session, session_id, limit=10
            )
            
            analysis_types = []
            for result in analysis_results:
                analysis_types.extend(result.get('analysis_types', []))
            
            # Remove duplicates, preserve order
            analysis_history = list(dict.fromkeys(analysis_types))
            
            # Get last parameters
            last_parameters = await ConversationHistoryService.get_last_parameters(
                session, session_id
            )
            
            # Get last results
            last_results = analysis_results[0] if analysis_results else None
            
            summary = {
                "total_messages": total_messages,
                "last_activity": last_activity.isoformat() if last_activity else None,
                "analysis_history": analysis_history,
                "last_parameters": last_parameters,
                "last_results": last_results,
                "session_id": session_id
            }
            
            logger.info(f"Generated context summary for session {session_id}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate context summary: {e}", exc_info=True)
            return {
                "total_messages": 0,
                "last_activity": None,
                "analysis_history": [],
                "last_parameters": {},
                "last_results": None,
                "session_id": session_id
            }