"""
multi-agent-fits-dev-02/app/services/file_service.py
"""

import logging
from pathlib import Path
from typing import Optional, List, Tuple
from uuid import UUID, uuid4
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import UploadFile

from app.db.models import FITSFile, User
from app.utils.file_manager import FileManager
from app.tools.fits.loader import validate_fits_structure, get_fits_header
from app.core.constants import FileExtensions, ResourceLimits

logger = logging.getLogger(__name__)


class FileService:
    """Service for managing FITS file operations"""

    @staticmethod
    async def upload_fits_file(
        file: UploadFile,
        user_id: UUID,
        session: AsyncSession
    ) -> Tuple[FITSFile, Path]:
        """
        Upload and validate FITS file
        
        Returns:
            Tuple of (FITSFile model, file_path)
        
        Raises:
            ValueError: If validation fails
        """
        
        # ==========================================
        # 1. Validate file extension
        # ==========================================
        if not file.filename.lower().endswith(FileExtensions.FITS):
            raise ValueError(
                f"Invalid file type. Only {', '.join(FileExtensions.FITS)} files are allowed"
            )

        # ==========================================
        # 2. Read file content
        # ==========================================
        file_content = await file.read()
        file_size = len(file_content)

        # Check file size
        if file_size > ResourceLimits.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ValueError(
                f"File size ({file_size / (1024*1024):.2f} MB) exceeds maximum limit of "
                f"{ResourceLimits.MAX_FILE_SIZE_MB} MB"
            )

        if file_size == 0:
            raise ValueError("File is empty")

        # ==========================================
        # 3. Generate file ID and save to filesystem
        # ==========================================
        file_id = uuid4()
        
        try:
            file_path = FileManager.save_fits_file(str(file_id), file_content)
        except Exception as e:
            logger.error(f"Failed to save file to filesystem: {e}")
            raise ValueError(f"Failed to save file: {str(e)}")

        # ==========================================
        # 4. Validate FITS structure
        # ==========================================
        validation_status = "pending"
        validation_error = None
        is_valid = False
        fits_metadata = {}
        data_info = {}
        metadata_filename = None

        try:
            # Validate structure
            validation_details = validate_fits_structure(str(file_path))
            is_valid = True
            validation_status = "valid"
            data_info = validation_details

            # Extract header information
            try:
                header_info = get_fits_header(str(file_path))
                metadata_filename = header_info.get("filename", "")
                fits_metadata = header_info.get("header", {})
            except Exception as e:
                logger.warning(f"Failed to extract FITS header: {e}")

        except Exception as e:
            logger.warning(f"FITS validation failed: {e}")
            validation_status = "invalid"
            validation_error = str(e)
            is_valid = False

        # ==========================================
        # 5. Create database record
        # ==========================================
        db_file = FITSFile(
            file_id=file_id,
            user_id=user_id,
            original_filename=file.filename,
            metadata_filename=metadata_filename,
            file_size=file_size,
            storage_path=str(file_path),
            is_valid=is_valid,
            validation_status=validation_status,
            validation_error=validation_error,
            fits_metadata=fits_metadata,
            data_info=data_info
        )

        try:
            session.add(db_file)
            await session.commit()
            await session.refresh(db_file)
            logger.info(f"File saved to database: {file_id}")
        except Exception as e:
            logger.error(f"Database error: {e}")
            # Rollback database
            await session.rollback()
            # Clean up file from filesystem
            FileManager.delete_fits_file(str(file_id))
            raise ValueError(f"Failed to save file metadata: {str(e)}")

        return db_file, file_path

    @staticmethod
    async def get_file_by_id(
        file_id: UUID,
        session: AsyncSession
    ) -> Optional[FITSFile]:
        """Get file information by ID"""
        
        result = await session.execute(
            select(FITSFile).where(FITSFile.file_id == file_id)
        )
        file = result.scalar_one_or_none()
        
        if file:
            # Update last accessed time
            file.last_accessed_at = datetime.now()
            await session.commit()
            
        return file

    @staticmethod
    async def get_user_files(
        user_id: UUID,
        session: AsyncSession,
        skip: int = 0,
        limit: int = 100
    ) -> List[FITSFile]:
        """Get all files for a user"""
        
        result = await session.execute(
            select(FITSFile)
            .where(FITSFile.user_id == user_id)
            .order_by(FITSFile.uploaded_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()

    @staticmethod
    async def delete_file(
        file_id: UUID,
        session: AsyncSession
    ) -> bool:
        """
        Delete file from database and filesystem
        
        Returns:
            True if successful, False if file not found
        """
        
        # Get file from database
        file = await FileService.get_file_by_id(file_id, session)
        
        if not file:
            return False

        try:
            # Delete from filesystem
            FileManager.delete_fits_file(str(file_id))
            
            # Delete from database
            await session.delete(file)
            await session.commit()
            
            logger.info(f"File deleted: {file_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting file {file_id}: {e}")
            await session.rollback()
            raise ValueError(f"Failed to delete file: {str(e)}")

    @staticmethod
    async def ensure_user_exists(
        user_id: UUID,
        session: AsyncSession
    ) -> User:
        """
        Ensure user exists in database, create if not
        
        Returns:
            User model
        """
        
        result = await session.execute(
            select(User).where(User.user_id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            # Create new user
            user = User(user_id=user_id)
            session.add(user)
            await session.commit()
            await session.refresh(user)
            logger.info(f"Created new user: {user_id}")
            
        return user