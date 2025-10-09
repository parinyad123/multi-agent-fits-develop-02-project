"""
multi-agent-fits-dev-02/app/db/models.py
"""

from datetime import datetime
from uuid import uuid4
from typing import Optional
from sqlalchemy import (
    Column, String, BigInteger, Boolean, DateTime, Text, Integer,
    ForeignKey, Index, CheckConstraint, ARRAY
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base import Base

# ============================================
# Users Table
# ============================================

class User(Base):
    __tablename__ = "users"
    
    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    username = Column(String(100), unique=True, nullable=True)
    email = Column(String(255), unique=True, nullable=True)
    preferences = Column(JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    fits_files = relationship("FITSFile", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    analyses = relationship("AnalysisHistory", back_populates="user", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_user_username", "username"),
        Index("idx_user_email", "email"),
    )
    
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
    
    session_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
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
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False)
    
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
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False)
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