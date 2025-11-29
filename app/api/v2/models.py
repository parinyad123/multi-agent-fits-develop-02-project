"""
app/api/v2/models.py
Optimized Pydantic models for v2 API
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from uuid import UUID


# ============================================
# PLOT MODELS
# ============================================

class PlotInfo(BaseModel):
    """Plot information for display"""
    plot_id: str
    plot_type: str
    plot_url: str
    title: str
    created_at: str

    model_config = ConfigDict(from_attributes=True)  # ✅ แก้ typo: drom → from

# ============================================
# CONVERSATION MODELS
# ============================================

class ConversationMessageLight(BaseModel):
    """Lightweight message for conversation list"""
    message_id: str
    role: str   # "user" or "assistant"
    content: str  # ✅ แก้ typo: conctent → content
    created_at: str
    plots: Optional[List[PlotInfo]] = None

    model_config = ConfigDict(from_attributes=True)


class ConversationResponse(BaseModel):
    """Response for conversation history"""
    session_id: str
    messages: List[ConversationMessageLight]
    total: int
    has_more: bool
    next_offset: Optional[int] = None

# ============================================
# SESSION MODELS
# ============================================

class SessionInfo(BaseModel):
    """Session information for sidebar list"""
    session_id: str
    title: str
    created_at: str
    last_message_at: str
    message_count: int

    model_config = ConfigDict(from_attributes=True)  # ✅ แก้ typo: drom → from

class SessionListResponse(BaseModel):
    """Response for session list"""
    sessions: List[SessionInfo]
    total: int
    has_more: bool
    next_offset: Optional[int] = None

# ============================================
# FILE MODELS
# ============================================

class FileInfoLight(BaseModel):
    """Lightweight file info for list view"""
    file_id: UUID
    original_filename: str  # ✅ แก้ typo: original_file → original_filename
    file_size: int
    is_valid: bool
    validation_status: str
    uploaded_at: str

    model_config = ConfigDict(from_attributes=True)

class FileInfoFull(BaseModel):
    """Full file info with metadata (on-demand)"""
    file_id: UUID
    user_id: UUID
    original_filename: str
    metadata_filename: Optional[str] = None
    file_size: int
    is_valid: bool
    validation_status: str
    validation_error: Optional[str] = None
    uploaded_at: str
    last_accessed_at: Optional[str] = None
    fits_metadata: Optional[dict] = None
    data_info: Optional[dict] = None
    
    model_config = ConfigDict(from_attributes=True)

class UserFilesResponseV2(BaseModel):
    """Response for user files list"""
    user_id: UUID
    files: List[FileInfoLight]
    total: int
    has_more: bool
    next_offset: Optional[int] = None

# ============================================
# WORKFLOW/ANALYSIS MODELS
# ============================================

class WorkflowStatusLight(BaseModel):  # ✅ แก้ typo: LIght → Light
    """Lightweight workflow status for polling"""
    task_id: str
    status: str
    progress: str
    current_step: Optional[str] = None
    error: Optional[str] = None


class WorkflowStatusFull(BaseModel):
    """Full workflow status with all details"""
    task_id: str
    status: str
    routing_strategy: Optional[str] = None
    current_step: Optional[str] = None
    progress: str
    completed_steps: Optional[List[dict]] = None
    created_at: Optional[str] = None  # ✅ เพิ่ม Optional
    completed_at: Optional[str] = None
    error: Optional[str] = None

# ============================================
# REQUEST MODELS
# ============================================

class AnalyzeRequest(BaseModel):
    """Request body for analysis submission"""
    query: str = Field(..., min_length=1)
    fits_file_id: Optional[str] = None
    session_id: Optional[str] = None
    user_expertise: str = Field(
        default="intermediate",
        description="User expertise level: beginner, intermediate, advanced, expert"
    )
    analysis_types: Optional[List[str]] = None
    parameters: Optional[dict] = None


class AnalyzeResponse(BaseModel):
    """Response for analysis submission"""
    task_id: str
    session_id: str
    status: str

# ============================================
# FILE UPLOAD RESPONSE (LIGHTWEIGHT)
# ============================================

class FileUploadResponseLight(BaseModel):
    """
    Lightweight upload response (NO metadata)
    Use this for immediate upload feedback
    """
    success: bool = True
    file_id: UUID
    original_filename: str
    file_size: int
    is_valid: bool
    validation_status: str
    validation_error: Optional[str] = None
    uploaded_at: str
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "file_id": "eb536ea0-cce7-479f-9153-7b3bc189d71d",
                "original_filename": "data.fits",
                "file_size": 3888000,
                "is_valid": True,
                "validation_status": "valid",
                "validation_error": None,
                "uploaded_at": "2025-11-28T07:23:30.376319+00:00",
                "message": "File uploaded successfully. Use GET /files/{file_id} for full details."
            }
        }

# ============================================
# Analysis (Rewrite agent) Result (LIGHTWEIGHT)
# ============================================

class AnalysisResultLight(BaseModel):
    """Lightweight analysis result - only what frontend needs"""
    task_id: str
    status: str
    content: str
    plots: List[PlotInfo]
    completed_at: Optional[str] = None
    
    class Config:
        from_attributes = True