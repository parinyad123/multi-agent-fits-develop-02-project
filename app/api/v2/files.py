# app/api/v2/files.py

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func
from typing import Optional
import logging
from uuid import UUID

# ✅ แก้แค่บรรทัดนี้
from app.db.base import get_async_session
from app.db.models import User, FITSFile
from app.core.auth import get_current_active_user  # ✅ ใช้จาก core.auth
from app.api.v2.models import FileInfoLight, FileInfoFull, UserFilesResponseV2, FileUploadResponseLight  
from app.services.file_service import FileService

router = APIRouter()
logger = logging.getLogger(__name__)



@router.get("/files", response_model=UserFilesResponseV2)
async def get_user_files(
    offset: int = Query(0, ge=0, description="Number of files to skip"),
    limit: int = Query(20, ge=1, le=100, description="Maximum files to return"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Get all FITS files for the current user.
    Returns lightweight file list (NO fits_metadata, NO data_info).
    
    **Pagination:**
    - offset: Skip first N files
    - limit: Max files per page (1-100)
    
    **Response:**
    - Sorted by upload date (newest first)
    - Only includes: file_id, filename, size, validation status
    - has_more: True if more files exist
    """
    try:
        # ✅ Query files with proper filtering
        query = (
            select(FITSFile)
            .where(
                FITSFile.user_id == current_user.user_id,
                FITSFile.is_deleted == False  # ✅ Only active files
            )
            .order_by(desc(FITSFile.uploaded_at))
            .offset(offset)
            .limit(limit + 1)
        )
        
        result = await db.execute(query)
        files = result.scalars().all()
        
        # Check if there are more results
        has_more = len(files) > limit
        if has_more:
            files = files[:limit]
        
        # ✅ Convert to lightweight response
        file_infos = []
        for file in files:
            file_infos.append(FileInfoLight(
                file_id=file.file_id,
                original_filename=file.original_filename,
                file_size=file.file_size,
                is_valid=file.is_valid,
                validation_status=file.validation_status or "unknown",
                uploaded_at=file.uploaded_at.isoformat()
            ))
        
        return UserFilesResponseV2(
            user_id=current_user.user_id,
            files=file_infos,
            total=len(file_infos),
            has_more=has_more,
            next_offset=offset + limit if has_more else None
        )
        
    except Exception as e:
        logger.error(f"Error fetching files: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch files")


@router.get("/files/{file_id}", response_model=FileInfoFull)
async def get_file_details(
    file_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Get full file details with metadata (on-demand).
    
    **Use this endpoint when:**
    - User clicks on a file to view details
    - Need FITS metadata for display
    - Need data_info for analysis
    
    **Warning:** Response can be large (1-10 KB with metadata)
    """
    try:
        # Find file
        query = select(FITSFile).where(
            FITSFile.file_id == file_id,
            FITSFile.user_id == current_user.user_id,
            FITSFile.is_deleted == False
        )
        result = await db.execute(query)
        file = result.scalar_one_or_none()
        
        if not file:
            raise HTTPException(status_code=404, detail="File not found")
        
        # ✅ เก็บค่าทั้งหมดก่อน commit (วิธีที่ปลอดภัยที่สุด)
        response_data = FileInfoFull(
            file_id=file.file_id,
            user_id=file.user_id,
            original_filename=file.original_filename,
            metadata_filename=file.metadata_filename,
            file_size=file.file_size,
            is_valid=file.is_valid,
            validation_status=file.validation_status or "unknown",
            validation_error=file.validation_error,
            uploaded_at=file.uploaded_at.isoformat(),
            last_accessed_at=file.last_accessed_at.isoformat() if file.last_accessed_at else None,
            fits_metadata=file.fits_metadata,
            data_info=file.data_info
        )
        
        # ✅ Update last_accessed_at AFTER creating response
        from datetime import datetime, timezone
        file.last_accessed_at = datetime.now(timezone.utc)
        await db.commit()
        
        # Return pre-built response (safe, no lazy loading)
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching file details: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch file details")


@router.get("/files/stats")
async def get_file_statistics(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Get file statistics for current user.
    
    Returns:
    - total_files: Total number of files
    - total_size: Total size in bytes
    - valid_files: Number of valid files
    - invalid_files: Number of invalid files
    """
    try:
        # Count total files
        total_query = select(func.count()).select_from(FITSFile).where(
            FITSFile.user_id == current_user.user_id,
            FITSFile.is_deleted == False
        )
        total_result = await db.execute(total_query)
        total_files = total_result.scalar() or 0
        
        # Sum total size
        size_query = select(func.sum(FITSFile.file_size)).where(
            FITSFile.user_id == current_user.user_id,
            FITSFile.is_deleted == False
        )
        size_result = await db.execute(size_query)
        total_size = size_result.scalar() or 0
        
        # Count valid files
        valid_query = select(func.count()).select_from(FITSFile).where(
            FITSFile.user_id == current_user.user_id,
            FITSFile.is_deleted == False,
            FITSFile.is_valid == True
        )
        valid_result = await db.execute(valid_query)
        valid_files = valid_result.scalar() or 0
        
        invalid_files = total_files - valid_files
        
        return {
            "user_id": str(current_user.user_id),
            "total_files": total_files,
            "total_size_bytes": int(total_size),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "valid_files": valid_files,
            "invalid_files": invalid_files
        }
        
    except Exception as e:
        logger.error(f"Error fetching file statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch file statistics")


# ============================================
# REUSE V1 ENDPOINTS (Upload & Delete)
# ============================================

@router.post("/files/upload", response_model=FileUploadResponseLight)
async def upload_file(
    file: UploadFile = File(..., description="FITS file to upload (max size from config)"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Upload FITS file (lightweight response).
    
    **Process:**
    1. Validate file extension (.fits, .fit)
    2. Save to storage
    3. Validate FITS structure
    4. Extract metadata (saved to database)
    5. Return lightweight response
    
    **Returns:** 
    - Lightweight response (~500 bytes)
    - NO fits_metadata or data_info in response
    - Use GET /files/{file_id} to get full details
    
    **Response includes:**
    - file_id: Use this for subsequent requests
    - is_valid: Whether FITS validation passed
    - validation_status: "valid", "invalid", or "corrupted"
    - message: Next steps or error info
    """
    try:
        # ✅ Reuse V1 service (still saves metadata to database)
        db_file, file_path = await FileService.upload_fits_file(
            file=file,
            user_id=current_user.user_id,
            session=db
        )
        
        await db.commit()
        await db.refresh(db_file)
        
        logger.info(
            f"File uploaded via V2: user={current_user.user_id}, "
            f"file={db_file.file_id}, valid={db_file.is_valid}, "
            f"size={db_file.file_size}"
        )
        
        # ✅ Return LIGHTWEIGHT response (NO metadata)
        return FileUploadResponseLight(
            success=True,
            file_id=db_file.file_id,
            original_filename=db_file.original_filename,
            file_size=db_file.file_size,
            is_valid=db_file.is_valid,
            validation_status=db_file.validation_status or "unknown",
            validation_error=db_file.validation_error,
            uploaded_at=db_file.uploaded_at.isoformat(),
            message=(
                "File uploaded and validated successfully. "
                f"Use GET /files/{db_file.file_id} for full metadata."
                if db_file.is_valid
                else f"File uploaded but validation failed: {db_file.validation_error}"
            )
        )
        
    except ValueError as e:
        logger.error(f"Validation error during upload: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error uploading file: {e}", exc_info=True)
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to upload file")


@router.delete("/files/{file_id}")
async def delete_file(
    file_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Delete FITS file (soft delete - sets is_deleted=True).
    
    **Authorization:** Users can only delete their own files
    """
    try:
        # ✅ Reuse V1 service
        await FileService.delete_file(
            file_id=file_id,
            user_id=current_user.user_id,
            session=db
        )
        
        await db.commit()
        
        logger.info(f"File deleted via V2: user={current_user.user_id}, file={file_id}")
        
        return {
            "success": True,
            "file_id": str(file_id),
            "message": "File deleted successfully"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting file: {e}", exc_info=True)
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete file")