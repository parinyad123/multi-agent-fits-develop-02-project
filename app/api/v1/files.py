"""
multi-agent-fits-dev-02/app/api/v1/files.py
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID
import logging

from app.db.base import get_async_session
from app.services.file_service import FileService
from app.models.file_models import (
    FileUploadResponse,
    FileInfoResponse,
    UserFilesResponse,
    FileDeleteResponse
)

router = APIRouter()
logger = logging.getLogger(__name__)


# ==========================================
# POST /upload - Upload FITS file
# ==========================================

@router.post("/upload", response_model=FileUploadResponse)
async def upload_fits_file(
    user_id: UUID,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Upload a FITS file for analysis
    
    Args:
        user_id: User ID (UUID)
        file: FITS file to upload
        
    Returns:
        FileUploadResponse with file metadata and validation status
        
    Raises:
        HTTPException 400: Invalid file type or validation failed
        HTTPException 413: File too large
        HTTPException 500: Server error
    """
    try:
        # Ensure user exists
        await FileService.ensure_user_exists(user_id, session)
        
        # Upload and validate file
        db_file, file_path = await FileService.upload_fits_file(
            file=file,
            user_id=user_id,
            session=session
        )
        
        return FileUploadResponse(
            success=True,
            file_id=db_file.file_id,
            filename=file_path.name,
            original_filename=db_file.original_filename,
            file_size=db_file.file_size,
            is_valid=db_file.is_valid,
            validation_status=db_file.validation_status,
            validation_details=db_file.data_info if db_file.is_valid else None,
            validation_error=db_file.validation_error,
            fits_metadata=db_file.fits_metadata if db_file.is_valid else None,
            uploaded_at=db_file.uploaded_at,
            message="File uploaded successfully" if db_file.is_valid 
                   else "File uploaded but validation failed"
        )
        
    except ValueError as e:
        logger.error(f"Validation error during upload: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


# ==========================================
# GET /{file_id} - Get file information
# ==========================================

@router.get("/{file_id}", response_model=FileInfoResponse)
async def get_file_info(
    file_id: UUID,
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get information about an uploaded FITS file
    
    Args:
        file_id: File ID (UUID)
        
    Returns:
        FileInfoResponse with file metadata
        
    Raises:
        HTTPException 404: File not found
        HTTPException 500: Server error
    """
    try:
        file = await FileService.get_file_by_id(file_id, session)
        
        if not file:
            raise HTTPException(status_code=404, detail=f"File not found: {file_id}")
        
        return FileInfoResponse.from_orm(file)
        
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error retrieving file info for {file_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve file info: {str(e)}")


# ==========================================
# GET /user/{user_id} - List user's files
# ==========================================

@router.get("/user/{user_id}", response_model=UserFilesResponse)
async def get_user_files(
    user_id: UUID,
    skip: int = Query(0, ge=0, description="Number of files to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of files to return"),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get all FITS files for a specific user
    
    Args:
        user_id: User ID (UUID)
        skip: Number of files to skip (pagination)
        limit: Maximum number of files to return
        
    Returns:
        UserFilesResponse with list of files
        
    Raises:
        HTTPException 500: Server error
    """
    try:
        files = await FileService.get_user_files(
            user_id=user_id,
            session=session,
            skip=skip,
            limit=limit
        )
        
        return UserFilesResponse(
            user_id=user_id,
            total_files=len(files),
            files=[FileInfoResponse.from_orm(f) for f in files]
        )
        
    except Exception as e:
        logger.error(f"Error retrieving files for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve user files: {str(e)}")


# ==========================================
# DELETE /{file_id} - Delete file
# ==========================================

@router.delete("/{file_id}", response_model=FileDeleteResponse)
async def delete_file(
    file_id: UUID,
    session: AsyncSession = Depends(get_async_session)
):
    """
    Delete a FITS file (manual deletion)
    
    Args:
        file_id: File ID (UUID)
        
    Returns:
        FileDeleteResponse confirming deletion
        
    Raises:
        HTTPException 404: File not found
        HTTPException 500: Server error
    """
    try:
        success = await FileService.delete_file(file_id, session)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"File not found: {file_id}")
        
        return FileDeleteResponse(
            success=True,
            file_id=file_id,
            message="File deleted successfully"
        )
        
    except HTTPException:
        raise
    
    except ValueError as e:
        logger.error(f"Error deleting file {file_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    except Exception as e:
        logger.error(f"Unexpected error deleting file {file_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")