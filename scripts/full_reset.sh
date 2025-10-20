#!/bin/bash

set -e

echo " FULL DATABASE RESET"
echo "======================================"

# Database credentials
DB_USER="fits_user"
DB_PASSWORD="fits_password"
DB_NAME="fits_analysis_db"
DB_HOST="localhost"
DB_PORT="5433"

export PGPASSWORD="$DB_PASSWORD"

cd /home/parinya/multi-agent-fits-dev-02

echo ""
echo "  This will DELETE ALL DATA!"
read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "1  Terminating connections..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d postgres << EOF
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = '$DB_NAME'
  AND pid <> pg_backend_pid();
EOF

echo ""
echo "2  Dropping database..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d postgres << EOF
DROP DATABASE IF EXISTS $DB_NAME;
EOF

echo ""
echo "3  Creating database..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d postgres << EOF
CREATE DATABASE $DB_NAME;
EOF

echo ""
echo "4  Removing old migrations..."
rm -f alembic/versions/*.py
touch alembic/versions/.gitkeep

echo ""
echo "5  Creating initial migration..."
alembic revision --autogenerate -m "Initial database schema with all features"

echo ""
echo "6  Running migrations..."
alembic upgrade head

echo ""
echo "7  Verifying schema..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME << EOF
-- List tables
\dt

-- Check fits_files columns
SELECT column_name, data_type, character_maximum_length
FROM information_schema.columns
WHERE table_name = 'fits_files'
ORDER BY ordinal_position;

-- Check sessions.session_id type
SELECT column_name, data_type, character_maximum_length
FROM information_schema.columns
WHERE table_name = 'sessions' AND column_name = 'session_id';
EOF

unset PGPASSWORD

echo ""
echo " Database reset complete!"
echo ""
echo "Next steps:"
echo "  1. Start server: python run.py"
echo "  2. Test upload: curl -X POST ..."
echo "  3. Run tests: python test_analysis_workflow.py"