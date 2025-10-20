#!/bin/bash

set -e

echo " Force Reset Database (with connection termination)"
echo "======================================================="

# Database credentials
DB_USER="fits_user"
DB_PASSWORD="fits_password"
DB_NAME="fits_analysis_db"
DB_HOST="localhost"
DB_PORT="5433"

export PGPASSWORD="$DB_PASSWORD"

# Change to project directory
cd /home/parinya/multi-agent-fits-dev-02

echo ""
echo "1  Checking active connections..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d postgres << EOF
SELECT count(*) as active_connections
FROM pg_stat_activity 
WHERE datname = '$DB_NAME';
EOF

echo ""
echo "2  Terminating all connections to $DB_NAME..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d postgres << EOF
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = '$DB_NAME'
  AND pid <> pg_backend_pid();
EOF

echo ""
echo "3  Dropping database..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d postgres << EOF
DROP DATABASE IF EXISTS $DB_NAME;
EOF

echo ""
echo "4  Creating new database..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d postgres << EOF
CREATE DATABASE $DB_NAME;
EOF

echo ""
echo "5  Removing old migration files..."
rm -f alembic/versions/*.py
touch alembic/versions/.gitkeep

echo ""
echo "6  Creating fresh migration..."
alembic revision --autogenerate -m "Initial database schema"

echo ""
echo "7  Running migrations..."
alembic upgrade head

echo ""
echo " Database reset complete!"
echo ""
echo "8  Verifying tables..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME << EOF
\dt
EOF

echo ""
echo "9  Checking session_id column type..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME << EOF
SELECT 
    table_name,
    column_name, 
    data_type,
    character_maximum_length
FROM information_schema.columns 
WHERE table_name IN ('sessions', 'analysis_history', 'file_sessions')
  AND column_name = 'session_id'
ORDER BY table_name;
EOF

unset PGPASSWORD

echo ""
echo " All done!"
echo ""
echo "Next steps:"
echo "  - Run tests: python test_analysis_workflow.py"
echo "  - Check logs: tail -f storage/logs/app.log"