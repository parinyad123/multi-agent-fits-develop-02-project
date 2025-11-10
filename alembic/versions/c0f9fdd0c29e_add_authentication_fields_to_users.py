"""Add authentication fields to users

Revision ID: c0f9fdd0c29e
Revises: 3d68e1934dd3
Create Date: 2025-11-09

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c0f9fdd0c29e'
down_revision = '3d68e1934dd3'
branch_labels = None
depends_on = None


def upgrade():
    # ========================================
    # STEP 1: Add new columns (nullable first)
    # ========================================
    
    # Add password_hash (nullable)
    op.add_column('users', sa.Column('password_hash', sa.String(255), nullable=True))
    
    # Add is_active (with default True)
    op.add_column('users', sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'))
    
    # ========================================
    # STEP 2: Fix existing NULL emails
    # ========================================
    
    # Update NULL emails to a placeholder
    op.execute("""
        UPDATE users 
        SET email = 'noemail_' || user_id::text || '@placeholder.local'
        WHERE email IS NULL
    """)
    
    # ========================================
    # STEP 3: Make email NOT NULL
    # ========================================
    
    # Now it's safe to add NOT NULL constraint
    op.alter_column('users', 'email',
                    existing_type=sa.String(255),
                    nullable=False)
    
    # ========================================
    # STEP 4: Add indexes
    # ========================================
    
    op.create_index('idx_user_is_active', 'users', ['is_active'], unique=False)


def downgrade():
    # Remove index
    op.drop_index('idx_user_is_active', table_name='users')
    
    # Revert email to nullable
    op.alter_column('users', 'email',
                    existing_type=sa.String(255),
                    nullable=True)
    
    # Remove columns
    op.drop_column('users', 'is_active')
    op.drop_column('users', 'password_hash')