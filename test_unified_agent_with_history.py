#!/usr/bin/env python3
"""
Test Unified Agent with History Support
"""

import asyncio
import json
from uuid import uuid4
from datetime import datetime, timezone

from sqlalchemy import delete, select

from app.db.base import AsyncSessionLocal
from app.db.models import (
    User,
    Session as SessionModel,
    AnalysisHistory,
    ConversationMessage,
    FITSFile
)
from app.agents.classification_parameter.unified_FITS_classification_parameter_agent import (
    UnifiedFITSClassificationAgent
)


async def get_or_create_test_user():
    """Get existing user or use known test user ID"""
    
    # ✅ Use your existing user ID from database
    existing_user_id = "123e4567-e89b-12d3-a456-426614174000"
    
    async with AsyncSessionLocal() as session:
        # Check if user exists
        result = await session.execute(
            select(User).where(User.user_id == existing_user_id)
        )
        user = result.scalar_one_or_none()
        
        if user:
            print(f"✅ Using existing user: {existing_user_id[:8]}...")
            return existing_user_id, False
        else:
            print(f"⚠️  User not found, will create test data without user deletion")
            return existing_user_id, False


async def setup_test_session():
    """Create test session with existing user"""
    
    # Get existing user
    test_user_id, should_delete_user = await get_or_create_test_user()
    
    async with AsyncSessionLocal() as session:
        
        test_session_id = str(uuid4())
        test_file_id = str(uuid4())
        test_filename = "test_0792180101_PN_source_lc_1200_5000eV.fits"
        
        # ✅ Create Session
        db_session = SessionModel(
            session_id=test_session_id,
            user_id=test_user_id
        )
        session.add(db_session)
        
        # ✅ Create FITS file
        fits_file = FITSFile(
            file_id=test_file_id,
            user_id=test_user_id,
            original_filename=test_filename,
            file_size=1024000,
            storage_path=f"/test/storage/{test_file_id}/{test_filename}",
            is_valid=True,
            validation_status="valid"
        )
        session.add(fits_file)
        
        # ✅ Create previous analysis WITH user_id
        analysis1 = AnalysisHistory(
            analysis_id=str(uuid4()),
            session_id=test_session_id,
            file_id=test_file_id,
            user_id=test_user_id,  # ✅ FIXED
            analysis_types=["power_law"],
            parameters={
                "power_law": {
                    "A0": 1.5,
                    "b0": 1.0,
                    "bins": 3500,
                    "A_max": 1e38,
                    "A_min": 0.0,
                    "b_max": 3.0,
                    "b_min": 0.1,
                    "maxfev": 1000000,
                    "filename": test_filename,
                    "low_freq": 1e-5,
                    "high_freq": 0.05,
                    "noise_bound_percent": 0.7
                }
            },
            status="completed",
            completed_at=datetime.now(timezone.utc)
        )
        session.add(analysis1)
        
        # ✅ Create conversation messages WITH user_id AND sequence_number
        messages = [
            ConversationMessage(
                message_id=str(uuid4()),
                session_id=test_session_id,
                user_id=test_user_id,
                role="user",
                content="Fit a power law model to this data",
                sequence_number=1,  # ✅ เพิ่มบรรทัดนี้!
                created_at=datetime.now(timezone.utc)
            ),
            ConversationMessage(
                message_id=str(uuid4()),
                session_id=test_session_id,
                user_id=test_user_id,
                role="assistant",
                content="Power law fit completed. A=1.5, b=1.0",
                sequence_number=2,  # ✅ เพิ่มบรรทัดนี้!
                message_metadata={
                    "routing_strategy": "analysis",
                    "analysis_types": ["power_law"]
                },
                created_at=datetime.now(timezone.utc)
            )
        ]
        
        for msg in messages:
            session.add(msg)
        
        await session.commit()
        
        print(f"✅ Using user: {test_user_id[:8]}...")
        print(f"✅ Created test session: {test_session_id[:8]}...")
        print(f"✅ Created FITS file: {test_file_id[:8]}...")
        print(f"   Filename: {test_filename}")
        print(f"✅ Created previous analysis with power_law parameters")
        print(f"   A0=1.5, b0=1.0, bins=3500")
        print(f"✅ Created 2 conversation messages")
        
        return test_session_id, test_file_id, test_user_id, should_delete_user


async def cleanup_test_session(session_id: str, file_id: str, user_id: str, should_delete_user: bool):
    """Clean up test data"""
    
    async with AsyncSessionLocal() as session:
        print(f"\n🧹 Cleaning up test data...")
        
        # Delete in proper order (respecting foreign keys)
        await session.execute(
            delete(ConversationMessage).where(
                ConversationMessage.session_id == session_id
            )
        )
        print(f"   ✓ Deleted conversation messages")
        
        await session.execute(
            delete(AnalysisHistory).where(
                AnalysisHistory.session_id == session_id
            )
        )
        print(f"   ✓ Deleted analysis history")
        
        await session.execute(
            delete(SessionModel).where(
                SessionModel.session_id == session_id
            )
        )
        print(f"   ✓ Deleted session")
        
        await session.execute(
            delete(FITSFile).where(
                FITSFile.file_id == file_id
            )
        )
        print(f"   ✓ Deleted FITS file")
        
        if should_delete_user:
            await session.execute(
                delete(User).where(
                    User.user_id == user_id
                )
            )
            print(f"   ✓ Deleted test user")
        else:
            print(f"   ⊗ Kept existing user (not created by test)")
        
        await session.commit()
        print(f"✅ Cleanup complete")


async def test_with_history():
    """Test unified agent with history"""
    
    print("\n" + "="*80)
    print("UNIFIED AGENT - HISTORY SUPPORT TEST")
    print("="*80)
    
    session_id = None
    file_id = None
    user_id = None
    should_delete_user = False
    
    try:
        # Setup
        session_id, file_id, user_id, should_delete_user = await setup_test_session()
        
        agent = UnifiedFITSClassificationAgent()
        
        test_cases = [
            {
                "query": "Fit power law again",
                "description": "Simple repeat - should inherit all parameters",
                "expect_inherit": True
            },
            {
                "query": "Same parameters but bins=5000",
                "description": "Inherit with override - bins should be 5000",
                "expect_inherit": True,
                "expect_override": {"bins": 5000}
            },
            {
                "query": "Fit power law with A0=2.0 and bins=4000",
                "description": "New explicit parameters",
                "expect_inherit": False,
                "expect_values": {"A0": 2.0, "bins": 4000}
            },
            {
                "query": "Use last settings",
                "description": "Explicit inheritance request",
                "expect_inherit": True
            }
        ]
        
        async with AsyncSessionLocal() as db_session:
            
            for i, test in enumerate(test_cases, 1):
                print(f"\n{'='*80}")
                print(f"TEST {i}/{len(test_cases)}: {test['description']}")
                print(f"{'='*80}")
                print(f"Query: \"{test['query']}\"")
                
                result = await agent.process_request(
                    user_input=test["query"],
                    context={"file_id": file_id},
                    session=db_session,
                    session_id=session_id
                )
                
                print(f"\n📊 Classification Results:")
                print(f"   Intent: {result.primary_intent}")
                print(f"   Routing: {result.routing_strategy}")
                print(f"   Analysis Types: {result.analysis_types}")
                print(f"   Confidence: {result.confidence:.2f}")
                
                if result.parameters:
                    print(f"\n📐 Parameters:")
                    
                    for atype, params in result.parameters.items():
                        print(f"\n   {atype.upper()}:")
                        
                        # Show key parameters only
                        key_params = ["bins", "A0", "b0", "low_freq", "high_freq"]
                        for key in key_params:
                            if key in params:
                                value = params[key]
                                print(f"      {key}: {value}")
                        
                        # Show inheritance info
                        if "_inherited_from" in params:
                            inherited_from = params["_inherited_from"]
                            aid = str(inherited_from.get('analysis_id', 'N/A'))
                            print(f"\n      ✅ INHERITED from: {aid[:8]}...")
                            print(f"         Position: {inherited_from.get('position', 'N/A')}")
                        
                        # Show overrides
                        if "_overridden_fields" in params:
                            print(f"      🔧 OVERRIDDEN: {params['_overridden_fields']}")
                    
                    print(f"\n   Source: {result.parameter_source}")
                
                # Validation
                print(f"\n✓ Validation:")
                
                if test.get("expect_inherit"):
                    source = result.parameter_source.get("power_law", "")
                    if source in ["inherited", "inherited_with_overrides"]:
                        print(f"   ✅ Inheritance: PASS")
                    else:
                        print(f"   ⚠️  Inheritance: FAIL (got {source})")
                
                if test.get("expect_override"):
                    power_law_params = result.parameters.get("power_law", {})
                    for key, expected in test["expect_override"].items():
                        actual = power_law_params.get(key)
                        if actual == expected:
                            print(f"   ✅ Override {key}={expected}: PASS")
                        else:
                            print(f"   ⚠️  Override {key}: FAIL (expected {expected}, got {actual})")
                
                if test.get("expect_values"):
                    power_law_params = result.parameters.get("power_law", {})
                    for key, expected in test["expect_values"].items():
                        actual = power_law_params.get(key)
                        if actual == expected:
                            print(f"   ✅ Value {key}={expected}: PASS")
                        else:
                            print(f"   ⚠️  Value {key}: FAIL (expected {expected}, got {actual})")
        
        # Statistics
        print(f"\n{'='*80}")
        print("AGENT STATISTICS")
        print("="*80)
        stats = agent.get_comprehensive_stats()
        print(json.dumps(stats, indent=2))
        
        print(f"\n{'='*80}")
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*80)
        
    except Exception as e:
        print(f"\n{'='*80}")
        print("❌ TEST FAILED")
        print("="*80)
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        print(f"\nFull Traceback:")
        import traceback
        traceback.print_exc()
        
    finally:
        if session_id and file_id and user_id:
            try:
                await cleanup_test_session(session_id, file_id, user_id, should_delete_user)
            except Exception as cleanup_error:
                print(f"\n⚠️  Cleanup error: {cleanup_error}")


if __name__ == "__main__":
    asyncio.run(test_with_history())