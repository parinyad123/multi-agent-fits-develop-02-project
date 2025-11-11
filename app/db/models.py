"""
multi-agent-fits-dev-02/app/db/models.py
"""

from datetime import datetime
from uuid import uuid4
from typing import Optional
from sqlalchemy import (
    Column, String, BigInteger, Boolean, DateTime, Text, Integer,
    ForeignKey, Index, CheckConstraint, ARRAY, Numeric
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from uuid import uuid4

from app.db.base import Base

from passlib.context import CryptContext

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ============================================
# Users Table
# ============================================

class User(Base):
    __tablename__ = "users"
    
    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    username = Column(String(100), unique=True, nullable=True)
    email = Column(String(255), unique=True, nullable=False)

    # Authentication fields
    password_hash = Column(String(255), nullable=True)  # NULL = OAuth users (future)
    is_active = Column(Boolean, default=True)

    # Profile & timestamps
    preferences = Column(JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    # updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    fits_files = relationship("FITSFile", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    analyses = relationship("AnalysisHistory", back_populates="user", cascade="all, delete-orphan")
    workflows = relationship("WorkflowExecution", back_populates="user", cascade="all, delete-orphan")
    conversation_messages = relationship("ConversationMessage", back_populates="user", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index("idx_user_username", "username"),
        Index("idx_user_email", "email"),
        Index("idx_user_is_active", "is_active"),
    )

    # Helper methods
    def set_password(self, password: str):
        """Hash and set password"""
        self.password_hash = pwd_context.hash(password)
    
    def verify_password(self, password: str) -> bool:
        """Verify password"""
        if not self.password_hash:
            return False
        return pwd_context.verify(password, self.password_hash)
    
    def __repr__(self):
        return f"<User(user_id={self.user_id}, username={self.username})>"


# ============================================
# FITS Files Table
# ============================================

class FITSFile(Base):
    __tablename__ = "fits_files"
    
    # Primary identifiers
    file_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    
    # File information
    original_filename = Column(String(255), nullable=False)
    metadata_filename = Column(String(255), nullable=True)
    file_size = Column(BigInteger, nullable=False)
    storage_path = Column(String(500), nullable=False, unique=True)
    
    # Validation & Quality
    is_valid = Column(Boolean, default=False)
    validation_status = Column(String(50), default="pending")  # pending, valid, invalid, corrupted
    validation_error = Column(Text, nullable=True)
    
    # Timestamps
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    last_accessed_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Soft Delete 
    is_deleted = Column(Boolean, default=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    deleted_by = Column(UUID(as_uuid=True), nullable=True)

    # Metadata (JSONB for flexibility)
    fits_metadata = Column(JSONB, default=dict)
    data_info = Column(JSONB, default=dict)
    
    # Relationships
    user = relationship("User", back_populates="fits_files")
    file_sessions = relationship("FileSession", back_populates="fits_file", cascade="all, delete-orphan")
    analyses = relationship("AnalysisHistory", back_populates="fits_file", cascade="all, delete-orphan")
    plots = relationship("PlotFile", back_populates="fits_file", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_fits_user_id", "user_id"),
        Index("idx_fits_uploaded_at", "uploaded_at"),
        Index("idx_fits_last_accessed", "last_accessed_at"),
        Index("idx_fits_validation_status", "validation_status"),
        Index("idx_fits_is_deleted", "is_deleted"),
        Index("idx_fits_is_valid", "is_valid"),
        CheckConstraint("file_size > 0", name="check_file_size_positive"),
    )
    
    def __repr__(self):
        return f"<FITSFile(file_id={self.file_id}, filename={self.original_filename})>"


# ============================================
# Sessions Table
# ============================================

class Session(Base):
    __tablename__ = "sessions"
    
    # session_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id = Column(String(255), primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    
    # Session info
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_activity_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Context
    current_file_id = Column(UUID(as_uuid=True), ForeignKey("fits_files.file_id", ondelete="SET NULL"), nullable=True)
    session_metadata = Column(JSONB, default=dict)
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    current_file = relationship("FITSFile", foreign_keys=[current_file_id])
    file_sessions = relationship("FileSession", back_populates="session", cascade="all, delete-orphan")
    analyses = relationship("AnalysisHistory", back_populates="session", cascade="all, delete-orphan")
    workflows = relationship("WorkflowExecution", back_populates="session", cascade="all, delete-orphan")
    conversation_messages = relationship("ConversationMessage", back_populates="session", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index("idx_session_user_id", "user_id"),
        Index("idx_session_last_activity", "last_activity_at"),
        Index("idx_session_is_active", "is_active"),
    )
    
    def __repr__(self):
        return f"<Session(session_id={self.session_id}, user_id={self.user_id})>"


# ============================================
# File-Session Junction Table
# ============================================

class FileSession(Base):
    __tablename__ = "file_sessions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    file_id = Column(UUID(as_uuid=True), ForeignKey("fits_files.file_id", ondelete="CASCADE"), nullable=False)
    # session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False)
    session_id = Column(String(255), ForeignKey("sessions.session_id", ondelete="CASCADE"))

    # Usage tracking
    added_at = Column(DateTime(timezone=True), server_default=func.now())
    last_used_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    usage_count = Column(Integer, default=0)
    
    # Context
    is_primary = Column(Boolean, default=False)
    analysis_context = Column(JSONB, default=dict)
    
    # Relationships
    fits_file = relationship("FITSFile", back_populates="file_sessions")
    session = relationship("Session", back_populates="file_sessions")
    
    # Indexes & Constraints
    __table_args__ = (
        Index("idx_file_session_file_id", "file_id"),
        Index("idx_file_session_session_id", "session_id"),
        Index("idx_file_session_unique", "file_id", "session_id", unique=True),
    )
    
    def __repr__(self):
        return f"<FileSession(file_id={self.file_id}, session_id={self.session_id})>"


# ============================================
# Analysis History Table
# ============================================

class AnalysisHistory(Base):
    __tablename__ = "analysis_history"
    
    analysis_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    file_id = Column(UUID(as_uuid=True), ForeignKey("fits_files.file_id", ondelete="CASCADE"), nullable=False)
    # session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False)
    session_id = Column(String(255), ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    
    # Analysis info
    analysis_types = Column(ARRAY(String), nullable=False)
    parameters = Column(JSONB, default=dict)
    results = Column(JSONB, default=dict)
    
    # Workflow info
    workflow_id = Column(UUID(as_uuid=True), nullable=True)
    status = Column(String(50), default="pending")  # pending, running, completed, failed
    
    # Performance
    execution_time_seconds = Column(BigInteger, nullable=True)
    
    # Timestamps
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    fits_file = relationship("FITSFile", back_populates="analyses")
    session = relationship("Session", back_populates="analyses")
    user = relationship("User", back_populates="analyses")
    plots = relationship("PlotFile", back_populates="analysis", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_analysis_file_id", "file_id"),
        Index("idx_analysis_session_id", "session_id"),
        Index("idx_analysis_user_id", "user_id"),
        Index("idx_analysis_started_at", "started_at"),
        Index("idx_analysis_status", "status"),
    )
    
    def __repr__(self):
        return f"<AnalysisHistory(analysis_id={self.analysis_id}, status={self.status})>"


# ============================================
# Plot Files Table
# ============================================

class PlotFile(Base):
    __tablename__ = "plot_files"
    
    plot_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    analysis_id = Column(UUID(as_uuid=True), ForeignKey("analysis_history.analysis_id", ondelete="CASCADE"), nullable=False)
    file_id = Column(UUID(as_uuid=True), ForeignKey("fits_files.file_id", ondelete="CASCADE"), nullable=False)
    
    # Plot info
    plot_type = Column(String(50), nullable=False)  # psd, power_law, bending_power_law
    storage_path = Column(String(500), nullable=False)
    plot_url = Column(String(500), nullable=True)
    
    # Metadata
    plot_metadata = Column(JSONB, default=dict)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    analysis = relationship("AnalysisHistory", back_populates="plots")
    fits_file = relationship("FITSFile", back_populates="plots")
    
    # Indexes
    __table_args__ = (
        Index("idx_plot_analysis_id", "analysis_id"),
        Index("idx_plot_file_id", "file_id"),
        Index("idx_plot_type", "plot_type"),
    )
    
    def __repr__(self):
        return f"<PlotFile(plot_id={self.plot_id}, plot_type={self.plot_type})>"
    
# ==========================================
# Workflow Executions Table
# ==========================================

class WorkflowExecution(Base):
    """
    Track complete workflow executions

    Record:
    - User request
    - Routing strategy
    - All agent outputs
    - Final response
    - Execution metadata
    """
    __tablename__ = "workflow_executions"

    # Primary key
    workflow_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign keys
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    session_id = Column(String(255), ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False)
    file_id = Column(UUID(as_uuid=True), ForeignKey("fits_files.file_id", ondelete="SET NULL"), nullable=True)
    analysis_id = Column(UUID(as_uuid=True), ForeignKey("analysis_history.analysis_id", ondelete="SET NULL"), nullable=True)
    
    # Request info
    user_query = Column(Text, nullable=False)
    request_context = Column(JSONB, default=dict)
    
    # Routing
    routing_strategy = Column(String(50), nullable=True)  # "analysis", "astrosage", "mixed"
    
    # Execution steps (complete history)
    completed_steps = Column(JSONB, default=list)  # All intermediate results
    
    # Status
    status = Column(String(50), default="pending")  # pending, in_progress, completed, failed
    current_step = Column(String(50), nullable=True)
    progress = Column(String(10), default="0%")
    
    # Error handling
    error = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Execution metadata
    execution_time_seconds = Column(Integer, nullable=True)
    total_tokens_used = Column(Integer, default=0)
    estimated_cost = Column(Numeric(10, 6), default=0.0)
    
    # Relationships
    user = relationship("User", back_populates="workflows")
    session = relationship("Session", back_populates="workflows")
    conversation_messages = relationship("ConversationMessage", back_populates="workflow", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_workflow_user_id", "user_id"),
        Index("idx_workflow_session_id", "session_id"),
        Index("idx_workflow_created_at", "created_at"),
        Index("idx_workflow_status", "status"),
    )

    def __repr__(self):
        return f"<WorkflowExecution(workflow_id={self.workflow_id}, status={self.status})>"


# ==========================================
# Conversation Messages Table 
# ==========================================

class ConversationMessage(Base):
    """
    Individual messages in a conversation
    
    บันทึก:
    - User messages
    - Assistant responses
    - Message order
    - Associated workflow
    """
    __tablename__ = "conversation_messages"
    
    # Primary key
    message_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign keys
    session_id = Column(String(255), ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False)
    workflow_id = Column(UUID(as_uuid=True), ForeignKey("workflow_executions.workflow_id", ondelete="CASCADE"), nullable=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    
    # Message info
    role = Column(String(20), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    
    # Message order (for proper sequencing)
    sequence_number = Column(Integer, nullable=False)
    
    # Message metadata
    message_metadata = Column(JSONB, default=dict)  # plots, tokens, model used, etc.
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    session = relationship("Session", back_populates="conversation_messages")
    workflow = relationship("WorkflowExecution", back_populates="conversation_messages")
    user = relationship("User", back_populates="conversation_messages")
    
    # Indexes
    __table_args__ = (
        Index("idx_message_session_id", "session_id"),
        Index("idx_message_workflow_id", "workflow_id"),
        Index("idx_message_created_at", "created_at"),
        Index("idx_message_session_sequence", "session_id", "sequence_number"),
    )

    def __repr__(self):
        return f"<ConversationMessage(message_id={self.message_id}, role={self.role})>"

