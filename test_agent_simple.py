#!/usr/bin/env python3
"""
Simple test without database - just test the agent classification
"""

import asyncio
from app.agents.classification_parameter.unified_FITS_classification_parameter_agent import (
    UnifiedFITSClassificationAgent
)


async def simple_test():
    """Test agent without any database dependencies"""
    
    print("\n" + "="*80)
    print("SIMPLE AGENT TEST (NO DATABASE)")
    print("="*80)
    
    try:
        # Create agent
        agent = UnifiedFITSClassificationAgent()
        print("✅ Agent created successfully")
        
        # Test queries
        test_queries = [
            "Fit power law with bins=3500",
            "Calculate statistics",
            "What is a black hole?",
            "Fit PSD and explain turbulence"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*80}")
            print(f"TEST {i}: {query}")
            print("="*80)
            
            # Call WITHOUT session/session_id (should work with defaults)
            result = await agent.process_request(
                user_input=query,
                context={}
                # ❌ NO session
                # ❌ NO session_id
            )
            
            print(f"✅ Classification successful!")
            print(f"   Intent: {result.primary_intent}")
            print(f"   Routing: {result.routing_strategy}")
            print(f"   Analysis Types: {result.analysis_types}")
            print(f"   Confidence: {result.confidence:.2f}")
            
            if result.parameters:
                print(f"   Has Parameters: Yes")
                for atype in result.parameters.keys():
                    print(f"      - {atype}")
        
        print(f"\n{'='*80}")
        print("✅ ALL SIMPLE TESTS PASSED!")
        print("="*80)
        
        # Show stats
        stats = agent.get_comprehensive_stats()
        print(f"\nTotal Requests: {stats['usage']['total_requests']}")
        print(f"Total Cost: ${stats['usage']['total_cost']:.4f}")
        
    except Exception as e:
        print(f"\n❌ SIMPLE TEST FAILED")
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(simple_test())