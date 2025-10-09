"""
multi-agent-fits-dev-02/tests/test_database.py
"""

import pytest
from uuid import uuid4
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.base import get_async_session, init_db
from app.db.models import User, FITSFile

@pytest.mark.asyncio
async def test_database_connection():
    """Test basic database connection"""
    async for session in get_async_session():
        # Simple query
        result = await session.execute("SELECT 1")
        assert result.scalar() == 1

@pytest.mark.asyncio
async def test_create_user():
    """Test creating a user"""
    async for session in get_async_session():
        user = User(
            username="test_user",
            email="test@example.com"
        )
        session.add(user)
        await session.commit()
        
        assert user.user_id is not None
        assert user.created_at is not None

@pytest.mark.asyncio
async def test_create_fits_file():
    """Test creating a FITS file record"""
    async for session in get_async_session():
        # Create user first
        user = User(username="fits_user", email="fits@example.com")
        session.add(user)
        await session.flush()
        
        # Create FITS file
        fits_file = FITSFile(
            user_id=user.user_id,
            original_filename="test.fits",
            file_size=1024000,
            storage_path="/storage/fits_files/test.fits",
            is_valid=True,
            validation_status="valid"
        )
        session.add(fits_file)
        await session.commit()
        
        assert fits_file.file_id is not None
        assert fits_file.uploaded_at is not None