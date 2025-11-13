"""
Test ConversationHistoryService with Real Session Data

Run: python test_real_data.py
"""

import asyncio
import json
from sqlalchemy import select, func

from app.db.base import AsyncSessionLocal
from app.db.models import Session as SessionModel, AnalysisHistory
from app.services.conversation_history_service import ConversationHistoryService


async def find_best_test_session():
    """Find a session with multiple analyses for testing"""
    
    print("\n" + "=" * 80)
    print("Finding Best Test Session...")
    print("=" * 80)
    
    async with AsyncSessionLocal() as session:
        # Find sessions with multiple analyses
        result = await session.execute(
            select(
                SessionModel.session_id,
                func.count(AnalysisHistory.analysis_id).label('analysis_count')
            ).join(
                AnalysisHistory,
                SessionModel.session_id == AnalysisHistory.session_id
            ).group_by(
                SessionModel.session_id
            ).having(
                func.count(AnalysisHistory.analysis_id) >= 2  # At least 2 analyses
            ).order_by(
                func.count(AnalysisHistory.analysis_id).desc()
            ).limit(5)
        )
        
        sessions = result.all()
        
        if not sessions:
            print("❌ No sessions with multiple analyses found!")
            return None
        
        print(f"\n✅ Found {len(sessions)} sessions with multiple analyses:")
        for i, (sid, count) in enumerate(sessions, 1):
            print(f"  {i}. {sid} ({count} analyses)")
        
        # Return session with most analyses
        best_session = sessions[0][0]
        print(f"\n✅ Selected session: {best_session}")
        
        return best_session


async def inspect_session_history(session_id: str):
    """Inspect what's in the session history"""
    
    print("\n" + "=" * 80)
    print(f"Inspecting Session: {session_id}")
    print("=" * 80)
    
    async with AsyncSessionLocal() as session:
        
        # Get all analyses for this session
        result = await session.execute(
            select(AnalysisHistory)
            .where(AnalysisHistory.session_id == session_id)
            .order_by(AnalysisHistory.completed_at.desc())
        )
        
        analyses = result.scalars().all()
        
        print(f"\n📊 Found {len(analyses)} analyses:")
        print("-" * 80)
        
        for i, analysis in enumerate(analyses, 1):
            print(f"\n{i}. Analysis ID: {analysis.analysis_id}")
            print(f"   Timestamp: {analysis.completed_at}")
            print(f"   Types: {analysis.analysis_types}")
            print(f"   Status: {analysis.status}")
            
            # Show parameters
            params = analysis.parameters or {}
            if params:
                print(f"   Parameters:")
                for key, value in params.items():
                    if isinstance(value, dict):
                        print(f"     - {key}:")
                        for pk, pv in value.items():
                            print(f"       • {pk}: {pv}")
                    else:
                        print(f"     - {key}: {value}")
            else:
                print(f"   Parameters: (empty)")
        
        return analyses


async def test_backward_search_real(session_id: str, analyses: list):
    """Test backward search with real data"""
    
    print("\n" + "=" * 80)
    print("Test 1: Backward Search for Specific Type")
    print("=" * 80)
    
    # Find what analysis types exist
    all_types = set()
    for analysis in analyses:
        all_types.update(analysis.analysis_types)
    
    print(f"\nAvailable analysis types: {list(all_types)}")
    
    # Test each type
    async with AsyncSessionLocal() as session:
        
        for analysis_type in all_types:
            print(f"\n--- Searching for: {analysis_type} ---")
            
            params = await ConversationHistoryService.get_last_parameters(
                session=session,
                session_id=session_id,
                analysis_type=analysis_type,
                scope="session",
                search_depth=10
            )
            
            if params:
                metadata = params.get("_metadata", {})
                position = metadata.get("search_position", "?")
                depth = metadata.get("search_depth", "?")
                
                print(f"✅ Found at position {position}/{depth}")
                print(f"   Analysis ID: {metadata.get('analysis_id')}")
                print(f"   Timestamp: {metadata.get('timestamp')}")
                
                # Show parameter preview
                param_data = params.get(analysis_type, {})
                if param_data:
                    print(f"   Parameters: {list(param_data.keys())}")
            else:
                print(f"❌ Not found")


async def test_any_parameters_real(session_id: str):
    """Test finding any parameters (no specific type)"""
    
    print("\n" + "=" * 80)
    print("Test 2: Find Any Parameters (No Type Filter)")
    print("=" * 80)
    
    async with AsyncSessionLocal() as session:
        
        params = await ConversationHistoryService.get_last_parameters(
            session=session,
            session_id=session_id,
            analysis_type=None,  # Any type
            scope="session",
            search_depth=10
        )
        
        if params:
            metadata = params.get("_metadata", {})
            position = metadata.get("search_position", "?")
            depth = metadata.get("search_depth", "?")
            
            print(f"✅ Found parameters at position {position}/{depth}")
            print(f"   Analysis ID: {metadata.get('analysis_id')}")
            print(f"   Analysis Types: {metadata.get('analysis_types')}")
            print(f"   Timestamp: {metadata.get('timestamp')}")
            
            # Show all parameter types
            param_types = [k for k in params.keys() if not k.startswith("_")]
            print(f"   Available: {param_types}")
            
            # Show full structure
            print(f"\n📄 Full Parameters:")
            print(json.dumps(params, indent=2, default=str))
        else:
            print(f"❌ No parameters found")


async def test_file_scope_real(session_id: str, analyses: list):
    """Test file-specific parameter query"""
    
    print("\n" + "=" * 80)
    print("Test 3: File-Specific Query")
    print("=" * 80)
    
    # Get unique file_ids
    file_ids = list(set(a.file_id for a in analyses if a.file_id))
    
    if not file_ids:
        print("⚠️ No file_ids found in analyses, skipping file scope test")
        return
    
    print(f"\nFound {len(file_ids)} unique files:")
    for i, fid in enumerate(file_ids, 1):
        print(f"  {i}. {fid}")
    
    async with AsyncSessionLocal() as session:
        
        # Test with first file
        test_file_id = file_ids[0]
        print(f"\n--- Testing with file: {test_file_id} ---")
        
        params = await ConversationHistoryService.get_last_parameters(
            session=session,
            session_id=session_id,
            file_id=test_file_id,
            scope="file",  # File-specific only
            search_depth=10
        )
        
        if params:
            metadata = params.get("_metadata", {})
            
            print(f"✅ Found file-specific parameters")
            print(f"   File ID: {metadata.get('file_id')}")
            print(f"   Analysis ID: {metadata.get('analysis_id')}")
            print(f"   Position: {metadata.get('search_position')}")
            
            param_types = [k for k in params.keys() if not k.startswith("_")]
            print(f"   Types: {param_types}")
        else:
            print(f"❌ No parameters found for this file")


async def test_search_depth_real(session_id: str):
    """Test different search depths"""
    
    print("\n" + "=" * 80)
    print("Test 4: Search Depth Variations")
    print("=" * 80)
    
    async with AsyncSessionLocal() as session:
        
        for depth in [1, 3, 5, 10]:
            print(f"\n--- Search Depth: {depth} ---")
            
            params = await ConversationHistoryService.get_last_parameters(
                session=session,
                session_id=session_id,
                analysis_type=None,
                scope="session",
                search_depth=depth
            )
            
            if params:
                metadata = params.get("_metadata", {})
                position = metadata.get("search_position", "?")
                actual_depth = metadata.get("search_depth", "?")
                
                print(f"✅ Found at position {position}/{actual_depth}")
                
                if position > depth:
                    print(f"⚠️ Found beyond requested depth!")
            else:
                print(f"❌ Nothing found in depth {depth}")


async def test_format_for_classification(session_id: str):
    """Test formatting for Classification Agent"""
    
    print("\n" + "=" * 80)
    print("Test 5: Format for Classification Agent")
    print("=" * 80)
    
    async with AsyncSessionLocal() as session:
        
        # Load messages
        messages = await ConversationHistoryService.get_recent_messages(
            session=session,
            session_id=session_id,
            limit=10
        )
        
        # Load parameters
        params = await ConversationHistoryService.get_last_parameters(
            session=session,
            session_id=session_id
        )
        
        # Format
        formatted = ConversationHistoryService.format_for_classification(
            messages=messages,
            last_parameters=params,
            max_tokens=2000
        )
        
        print(f"\n✅ Formatted context ({len(formatted)} chars):")
        print("-" * 80)
        print(formatted[:1000])  # Show first 1000 chars
        if len(formatted) > 1000:
            print(f"\n... ({len(formatted) - 1000} more characters)")


async def main():
    """Run all real data tests"""
    
    print("=" * 80)
    print("ConversationHistoryService - Real Data Testing")
    print("=" * 80)
    
    # Find best test session
    session_id = await find_best_test_session()
    
    if not session_id:
        print("\n❌ No suitable test session found!")
        print("Please create some analyses first.")
        return
    
    # Inspect session
    analyses = await inspect_session_history(session_id)
    
    if not analyses:
        print("\n❌ No analyses found in session!")
        return
    
    # Run tests
    await test_backward_search_real(session_id, analyses)
    await test_any_parameters_real(session_id)
    await test_file_scope_real(session_id, analyses)
    await test_search_depth_real(session_id)
    await test_format_for_classification(session_id)
    
    print("\n" + "=" * 80)
    print("✅ All Real Data Tests Completed!")
    print("=" * 80)
    
    # Summary
    print("\n📊 Summary:")
    print(f"  Session tested: {session_id}")
    print(f"  Analyses found: {len(analyses)}")
    print(f"  Analysis types: {list(set(t for a in analyses for t in a.analysis_types))}")


if __name__ == "__main__":
    asyncio.run(main())