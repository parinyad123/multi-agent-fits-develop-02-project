"""
File Management API Endpoints with Authentication
Protected routes for FITS file operations

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

from app.core.auth import get_current_active_user
from app.db.models import User

router = APIRouter()
logger = logging.getLogger(__name__)


# ==========================================
# POST /upload - Upload FITS file
# ==========================================

@router.post("/upload", response_model=FileUploadResponse)
async def upload_fits_file(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Upload a FITS file for analysis
    
    **Requires Authentication**
    
    Args:
        file: FITS file to upload (max size defined in config)
        
    Returns:
        FileUploadResponse with file metadata and validation status
        
    Raises:
        HTTPException 400: Invalid file type or validation failed
        HTTPException 413: File too large
        HTTPException 500: Server error
    """
    try:
        # Use user_id from JWT token
        user_id = current_user.user_id  

        # Ensure user exists
        await FileService.ensure_user_exists(user_id, session)
        
        # Upload and validate file
        db_file, file_path = await FileService.upload_fits_file(
            file=file,
            user_id=user_id,
            session=session
        )
        
        logger.info(
            f"File uploaded: user={user_id}, file={db_file.file_id}, "
            f"size={db_file.file_size}, valid={db_file.is_valid}"
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
        logger.error(f"Error uploading file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")



# ==========================================
# GET /{file_id} - Get file information
# ==========================================

@router.get("/{file_id}", response_model=FileInfoResponse)
async def get_file_info(
    file_id: UUID,
    current_user: User = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get information about an uploaded FITS file
    
    **Requires Authentication**
    **Authorization**: Users can only access their own files
    
    Args:
        file_id: File ID (UUID)
        
    Returns:
        FileInfoResponse with file metadata
        
    Raises:
        HTTPException 403: Access denied (file belongs to another user)
        HTTPException 404: File not found
        HTTPException 500: Server error
    """
    try:
        file = await FileService.get_file_by_id(file_id, session)
        
        if not file:
            raise HTTPException(status_code=404, detail=f"File not found: {file_id}")
        
        # Check file ownership
        if file.user_id != current_user.user_id:
            logger.warning(
                f"Access denied: user={current_user.user_id} tried to access "
                f"file={file_id} owned by user={file.user_id}"
            )
            raise HTTPException(
                status_code=403,
                detail="Access denied: You don't have permission to access this file"
            )
        
        logger.info(f"File info retrieved: user={current_user.user_id}, file={file_id}")
        
        return FileInfoResponse.from_orm(file)
        
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error retrieving file info for {file_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve file info: {str(e)}")


# ==========================================
# GET /user/me/files - List current user's files
# ==========================================

@router.get("/user/me/files", response_model=UserFilesResponse)
async def get_my_files(
    skip: int = Query(0, ge=0, description="Number of files to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of files to return"),
    current_user: User = Depends(get_current_active_user),  #  Require auth
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get all FITS files for the current authenticated user
    
    **Requires Authentication**
    
    Args:
        skip: Number of files to skip (pagination)
        limit: Maximum number of files to return
        
    Returns:
        UserFilesResponse with list of files
        
    Raises:
        HTTPException 500: Server error
    """
    try:
        # Use current_user.user_id from JWT
        user_id = current_user.user_id
        
        files = await FileService.get_user_files(
            user_id=user_id,
            session=session,
            skip=skip,
            limit=limit
        )
        
        logger.info(
            f"Files retrieved: user={user_id}, count={len(files)}, "
            f"skip={skip}, limit={limit}"
        )
        
        return UserFilesResponse(
            user_id=user_id,
            total_files=len(files),
            files=[FileInfoResponse.from_orm(f) for f in files]
        )
        
    except Exception as e:
        logger.error(f"Error retrieving files for user {current_user.user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to retrieve user files: {str(e)}"
        )


# ==========================================
# DELETE /{file_id} - Delete file
# ==========================================

@router.delete("/{file_id}", response_model=FileDeleteResponse)
async def delete_file(
    file_id: UUID,
    current_user: User = Depends(get_current_active_user), 
    session: AsyncSession = Depends(get_async_session)
):
    """
    Delete a FITS file (soft delete)
    
    **Requires Authentication**
    **Authorization**: Users can only delete their own files
    
    Args:
        file_id: File ID (UUID)
        
    Returns:
        FileDeleteResponse with success status
        
    Raises:
        HTTPException 403: Access denied (file belongs to another user)
        HTTPException 404: File not found
        HTTPException 500: Server error
    """
    try:
        file = await FileService.get_file_by_id(file_id, session)
        
        if not file:
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {file_id}"
            )
        
        # Check file ownership
        if file.user_id != current_user.user_id:
            logger.warning(
                f"Delete denied: user={current_user.user_id} tried to delete "
                f"file={file_id} owned by user={file.user_id}"
            )
            raise HTTPException(
                status_code=403,
                detail="Access denied: You don't have permission to delete this file"
            )
        
        # Delete file (soft delete)
        await FileService.delete_file(file_id, current_user.user_id, session)
        
        logger.info(f"File deleted: user={current_user.user_id}, file={file_id}")
        
        return FileDeleteResponse(
            success=True,
            file_id=file_id,
            message="File deleted successfully"
        )
        
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error deleting file {file_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete file: {str(e)}"
        )
