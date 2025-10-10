"""
multi-agent-fits-dev-02/app/models/file_models.py
"""

from pydantic import BaseModel, Field, validator
from uuid import UUID
from datetime import datetime 
from typing import Optional, Dict, Any, List

# ==========================================
# Request Models
# ==========================================

class FileUploadResponse(BaseModel):
    """Response model for file upload"""
    success: bool
    file_id: UUID
    filename: str
    original_filename: str
    file_size: int
    is_valid: bool
    validation_status: str
    validation_details: Optional[Dict[str, Any]] = None
    validation_error: Optional[str] = None
    fits_metadata: Optional[Dict[str, Any]] = None
    uploaded_at: datetime
    message: str

    class Config:
        from_attributes = True


class FileInfoResponse(BaseModel):
    """Response model for file information"""
    file_id: UUID
    user_id: UUID
    original_filename: str
    metadata_filename: Optional[str]
    file_size: int
    is_valid: bool
    validation_status: str
    validation_error: Optional[str]
    uploaded_at: datetime
    last_accessed_at: datetime
    fits_metadata: Optional[Dict[str, Any]]
    data_info: Optional[Dict[str, Any]]

    class Config:
        from_attributes = True


class UserFilesResponse(BaseModel):
    """Response model for listing user's files"""
    user_id: UUID
    total_files: int
    files: List[FileInfoResponse]

    class Config:
        from_attributes = True


class FileDeleteResponse(BaseModel):
    """Response model for file deletion"""
    success: bool
    file_id: UUID
    message: str

    class Config:
        from_attributes = True