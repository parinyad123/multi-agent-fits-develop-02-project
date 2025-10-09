#!/bin/bash
# check_db.sh
# ==============================
# ทำให้ execute ได้
# chmod +x check_db.sh

# รัน
# ./check_db.sh
# ==============================

echo "==================================="
echo "PostgreSQL Connection Diagnostic"
echo "==================================="

echo ""
echo "1. Checking Docker containers..."
docker ps | grep postgres

echo ""
echo "2. Checking port 5433..."
sudo lsof -i :5433 || echo "Port 5433 is not in use"

echo ""
echo "3. Testing container health..."
docker exec fits-postgres pg_isready -U fits_user 2>/dev/null || echo "Container not running or not ready"

echo ""
echo "4. Testing database connection..."
docker exec -it fits-postgres psql -U fits_user -d fits_analysis_db -c "SELECT 1;" 2>/dev/null || echo "Cannot connect to database"

echo ""
echo "5. Checking recent logs..."
docker-compose logs --tail=10 postgres

echo ""
echo "==================================="
echo "Diagnostic complete"
echo "==================================="