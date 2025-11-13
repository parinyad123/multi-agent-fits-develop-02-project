"""
Test ConversationHistoryService Phase 1: Backward Search (Fixed v2)

Run: python test_conversation_history_phase1.py
"""

import asyncio
import json
from uuid import uuid4, UUID
from datetime import datetime, timedelta

from app.db.base import AsyncSessionLocal
from app.db.models import AnalysisHistory, FITSFile, Session as SessionModel, User
from app.services.conversation_history_service import ConversationHistoryService


async def get_or_create_test_user(session) -> UUID:
    """Get existing user or create test user"""
    
    from sqlalchemy import select
    
    # Try to get any existing user
    result = await session.execute(
        select(User).limit(1)
    )
    existing_user = result.scalar_one_or_none()
    
    if existing_user:
        print(f"✅ Using existing user: {existing_user.user_id}")
        return existing_user.user_id
    
    # Create test user if none exists
    print("⚠️ No users found, creating test user...")
    test_user = User(
        email=f"test-{uuid4()}@example.com",
        password_hash="test_hash",
        is_active=True,
        is_verified=True
    )
    
    session.add(test_user)
    await session.flush()
    
    print(f"✅ Created test user: {test_user.user_id}")
    return test_user.user_id


async def setup_test_data(session, session_id: str, user_id: UUID):
    """Create test data for backward search testing"""
    
    print("\n" + "=" * 80)
    print("Setting up test data...")
    print("=" * 80)
    
    # ========================================
    # STEP 1: Create dummy FITS file
    # ========================================
    test_file = FITSFile(
        user_id=user_id,
        original_filename="test_file.fits",
        metadata_filename="test_file.fits",
        storage_path="/tmp/test_file.fits",
        file_size=1024000,
        is_valid=True,
        validation_status="valid",
        fits_metadata={},
        data_info={}
    )
    
    session.add(test_file)
    await session.flush()  # Get file_id
    
    file_id = test_file.file_id
    print(f"✅ Created test FITS file: {file_id}")
    
    # ========================================
    # STEP 2: Create session record
    # ========================================
    test_session = SessionModel(
        session_id=session_id,
        user_id=user_id,
        created_at=datetime.now(),
        last_activity_at=datetime.now(),
        is_active=True,
        session_metadata={}
    )
    
    session.add(test_session)
    await session.flush()
    
    print(f"✅ Created test session: {session_id}")
    
    # ========================================
    # STEP 3: Create test analyses
    # ========================================
    base_time = datetime.now() - timedelta(minutes=30)
    
    # Analysis 1: power_law @ -30 min
    analysis1 = AnalysisHistory(
        file_id=file_id,
        user_id=user_id,
        session_id=session_id,
        analysis_types=["power_law"],
        parameters={
            "power_law": {
                "A0": 1.0,
                "b0": 1.0,
                "bins": 5000
            }
        },
        results={"completed": {}, "errors": {}},
        status="completed",
        started_at=base_time,
        completed_at=base_time + timedelta(seconds=10),
        execution_time_seconds=10
    )
    
    # Analysis 2: bending_power_law @ -20 min
    analysis2 = AnalysisHistory(
        file_id=file_id,
        user_id=user_id,
        session_id=session_id,
        analysis_types=["bending_power_law"],
        parameters={
            "bending_power_law": {
                "A0": 10.0,
                "fb0": 0.01,
                "bins": 3500
            }
        },
        results={"completed": {}, "errors": {}},
        status="completed",
        started_at=base_time + timedelta(minutes=10),
        completed_at=base_time + timedelta(minutes=10, seconds=15),
        execution_time_seconds=15
    )
    
    # Analysis 3: metadata @ -10 min (latest, no parameters)
    analysis3 = AnalysisHistory(
        file_id=file_id,
        user_id=user_id,
        session_id=session_id,
        analysis_types=["metadata"],
        parameters={},  # No parameters!
        results={"completed": {}, "errors": {}},
        status="completed",
        started_at=base_time + timedelta(minutes=20),
        completed_at=base_time + timedelta(minutes=20, seconds=2),
        execution_time_seconds=2
    )
    
    session.add_all([analysis1, analysis2, analysis3])
    await session.commit()
    
    print(f"\n✅ Created 3 test analyses:")
    print(f"  1. power_law @ {analysis1.completed_at}")
    print(f"  2. bending_power_law @ {analysis2.completed_at}")
    print(f"  3. metadata (no params) @ {analysis3.completed_at}")
    
    return file_id, analysis1.analysis_id, analysis2.analysis_id, analysis3.analysis_id


async def cleanup_test_data(session, file_id: UUID, session_id: str, aid1: UUID, aid2: UUID, aid3: UUID):
    """Cleanup test data"""
    
    print("\n" + "=" * 80)
    print("Cleaning up test data...")
    print("=" * 80)
    
    try:
        # Delete analyses
        for aid in [aid1, aid2, aid3]:
            analysis = await session.get(AnalysisHistory, aid)
            if analysis:
                await session.delete(analysis)
        
        # Delete session
        test_session = await session.get(SessionModel, session_id)
        if test_session:
            await session.delete(test_session)
        
        # Delete file
        test_file = await session.get(FITSFile, file_id)
        if test_file:
            await session.delete(test_file)
        
        await session.commit()
        
        print("✅ Test data cleaned up")
        
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")
        await session.rollback()


async def test_backward_search():
    """Test backward search functionality"""
    
    print("\n" + "=" * 80)
    print("Testing ConversationHistoryService - Phase 1: Backward Search")
    print("=" * 80)
    
    file_id = None
    aid1 = aid2 = aid3 = None
    test_session_id = None
    
    try:
        async with AsyncSessionLocal() as session:
            
            # Get or create test user
            test_user_id = await get_or_create_test_user(session)
            
            # Generate test session ID
            test_session_id = f"test-session-{uuid4()}"
            
            # Setup test data
            file_id, aid1, aid2, aid3 = await setup_test_data(
                session, 
                test_session_id, 
                test_user_id
            )
            
            # ========================================
            # Test 1: Query specific type (power_law)
            # Should find Analysis 1 (not latest!)
            # ========================================
            print("\n" + "=" * 80)
            print("Test 1: Backward Search for power_law")
            print("=" * 80)
            print("Expected: Find Analysis 1 (position 3, oldest with power_law)")
            
            params = await ConversationHistoryService.get_last_parameters(
                session=session,
                session_id=test_session_id,
                analysis_type="power_law",
                scope="session",
                search_depth=10
            )
            
            if params:
                print("\n✅ SUCCESS: Found power_law parameters!")
                print(json.dumps(params, indent=2, default=str))
                
                # Verify it's from Analysis 1
                metadata = params.get("_metadata", {})
                assert metadata.get("analysis_id") == str(aid1), f"Wrong analysis! Expected {aid1}, got {metadata.get('analysis_id')}"
                assert metadata.get("search_position") == 3, f"Wrong position! Expected 3, got {metadata.get('search_position')}"
                
                power_law_params = params.get("power_law", {})
                assert power_law_params.get("A0") == 1.0, "Wrong A0 value!"
                assert power_law_params.get("bins") == 5000, "Wrong bins value!"
                
                print("✅ All assertions passed!")
            else:
                print("❌ FAIL: No parameters found")
                return False
            
            # ========================================
            # Test 2: Query any parameters
            # Should skip Analysis 3, find Analysis 2
            # ========================================
            print("\n" + "=" * 80)
            print("Test 2: Backward Search for any parameters")
            print("=" * 80)
            print("Expected: Skip Analysis 3 (no params), find Analysis 2")
            
            params = await ConversationHistoryService.get_last_parameters(
                session=session,
                session_id=test_session_id,
                analysis_type=None,  # Any type
                scope="session",
                search_depth=10
            )
            
            if params:
                print("\n✅ SUCCESS: Found parameters!")
                print(json.dumps(params, indent=2, default=str))
                
                # Verify it's from Analysis 2
                metadata = params.get("_metadata", {})
                assert metadata.get("analysis_id") == str(aid2), "Wrong analysis returned!"
                assert metadata.get("search_position") == 2, "Wrong search position!"
                
                bpl_params = params.get("bending_power_law", {})
                assert bpl_params.get("A0") == 10.0, "Wrong A0 value!"
                
                print("✅ All assertions passed!")
            else:
                print("❌ FAIL: No parameters found")
                return False
            
            # ========================================
            # Test 3: Query non-existent type
            # ========================================
            print("\n" + "=" * 80)
            print("Test 3: Search for non-existent type")
            print("=" * 80)
            print("Expected: Empty result after searching all analyses")
            
            params = await ConversationHistoryService.get_last_parameters(
                session=session,
                session_id=test_session_id,
                analysis_type="statistics",  # Doesn't exist
                scope="session",
                search_depth=10
            )
            
            if not params:
                print("✅ SUCCESS: Correctly returned empty for non-existent type")
            else:
                print("❌ FAIL: Should not have found statistics parameters")
                return False
            
            # ========================================
            # Test 4: Limited search depth
            # ========================================
            print("\n" + "=" * 80)
            print("Test 4: Limited search depth (depth=1)")
            print("=" * 80)
            print("Expected: Only check latest (Analysis 3), return empty")
            
            params = await ConversationHistoryService.get_last_parameters(
                session=session,
                session_id=test_session_id,
                analysis_type="power_law",
                scope="session",
                search_depth=1  # Only check latest
            )
            
            if not params:
                print("✅ SUCCESS: Correctly stopped at depth=1")
            else:
                print("❌ FAIL: Should not have found power_law with depth=1")
                return False
            
            # ========================================
            # Test 5: File scope filter
            # ========================================
            print("\n" + "=" * 80)
            print("Test 5: File-specific query")
            print("=" * 80)
            
            params = await ConversationHistoryService.get_last_parameters(
                session=session,
                session_id=test_session_id,
                file_id=file_id,
                scope="file",
                search_depth=10
            )
            
            if params:
                print("✅ SUCCESS: Found file-specific parameters!")
                metadata = params.get("_metadata", {})
                assert metadata.get("file_id") == str(file_id), "Wrong file_id!"
                print(json.dumps(params, indent=2, default=str))
            else:
                print("❌ FAIL: Should have found file-specific parameters")
                return False
            
            # ========================================
            # Cleanup
            # ========================================
            await cleanup_test_data(session, file_id, test_session_id, aid1, aid2, aid3)
        
        print("\n" + "=" * 80)
        print("✅ ALL PHASE 1 TESTS PASSED!")
        print("=" * 80)
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ Assertion failed: {e}")
        
        # Cleanup on error
        if file_id and aid1 and test_session_id:
            try:
                async with AsyncSessionLocal() as session:
                    await cleanup_test_data(session, file_id, test_session_id, aid1, aid2, aid3)
            except:
                pass
        
        return False
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on error
        if file_id and aid1 and test_session_id:
            try:
                async with AsyncSessionLocal() as session:
                    await cleanup_test_data(session, file_id, test_session_id, aid1, aid2, aid3)
            except:
                pass
        
        return False


async def test_with_real_data():
    """Test with existing session data"""
    
    print("\n" + "=" * 80)
    print("Optional: Test with Real Data")
    print("=" * 80)
    
    session_id = input("\nEnter existing session_id (or press Enter to skip): ").strip()
    
    if not session_id:
        print("Skipping real data test")
        return
    
    async with AsyncSessionLocal() as session:
        
        print(f"\nTesting backward search with session: {session_id}")
        
        # Test 1: Find power_law parameters
        print("\n--- Query: power_law ---")
        params = await ConversationHistoryService.get_last_parameters(
            session=session,
            session_id=session_id,
            analysis_type="power_law",
            search_depth=10
        )
        
        if params:
            print("✅ Found power_law parameters:")
            print(json.dumps(params, indent=2, default=str))
        else:
            print("ℹ️ No power_law parameters found in this session")
        
        # Test 2: Find any parameters
        print("\n--- Query: any parameters ---")
        params = await ConversationHistoryService.get_last_parameters(
            session=session,
            session_id=session_id,
            search_depth=10
        )
        
        if params:
            print("✅ Found parameters:")
            print(json.dumps(params, indent=2, default=str))
        else:
            print("ℹ️ No parameters found in this session")


async def main():
    """Run all tests"""
    
    print("=" * 80)
    print("ConversationHistoryService Phase 1 Tests")
    print("=" * 80)
    
    # Test 1: Automated test with synthetic data
    success = await test_backward_search()
    
    if not success:
        print("\n❌ Automated tests FAILED!")
        return
    
    # Test 2: Optional test with real data
    await test_with_real_data()
    
    print("\n" + "=" * 80)
    print("🎉 All Phase 1 tests completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())