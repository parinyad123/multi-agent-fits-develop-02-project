"""
tests/test_classification_agent.py

ทดสอบ UnifiedFITSClassificationAgent กับ Groq API
รัน: python -m tests.test_classification_agent
"""

import asyncio
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

from app.agents.classification_parameter.unified_FITS_classification_parameter_agent import (
    UnifiedFITSClassificationAgent,
    UnifiedFITSResult,
)

# ==================================
# Test cases
# ==================================

TEST_CASES = [
    {
        "label": "Pure analysis",
        "query": "Fit both power law and bending power law with 5000 bins",
        "expected_intent": "analysis",
        "expected_strategy": "analysis",
    },
    {
        "label": "Pure question",
        "query": "What is IRAS 13224-3809 and why is it important for AGN studies?",
        "expected_intent": "general",
        "expected_strategy": "groq",
    },
    {
        "label": "Mixed request",
        "query": (
            "Compute PSD and fit power law with 5000 bins, "
            "then discuss how the spectral indices reflect accretion regimes."
        ),
        "expected_intent": "mixed",
        "expected_strategy": "mixed",
    },
    {
        "label": "Metadata only",
        "query": "Show me the FITS file header and observation metadata",
        "expected_intent": "analysis",
        "expected_strategy": "analysis",
    },
]


def _pass_fail(result: UnifiedFITSResult, expected_intent: str, expected_strategy: str) -> str:
    intent_ok = result.primary_intent == expected_intent
    strategy_ok = result.routing_strategy == expected_strategy
    if intent_ok and strategy_ok:
        return "PASS"
    parts = []
    if not intent_ok:
        parts.append(f"intent: got '{result.primary_intent}', want '{expected_intent}'")
    if not strategy_ok:
        parts.append(f"strategy: got '{result.routing_strategy}', want '{expected_strategy}'")
    return "FAIL  (" + ", ".join(parts) + ")"


async def run_tests():
    agent = UnifiedFITSClassificationAgent()

    print("\n" + "=" * 70)
    print("  UnifiedFITSClassificationAgent  —  Groq integration test")
    print("=" * 70)

    passed = 0
    for tc in TEST_CASES:
        print(f"\n[{tc['label']}]")
        print(f"  Query   : {tc['query'][:80]}")

        result: UnifiedFITSResult = await agent.process_request(
            user_input=tc["query"],
            context={"has_uploaded_files": True},
        )

        status = _pass_fail(result, tc["expected_intent"], tc["expected_strategy"])
        if status == "PASS":
            passed += 1

        print(f"  Intent  : {result.primary_intent}  (confidence={result.confidence:.2f})")
        print(f"  Strategy: {result.routing_strategy}")
        print(f"  Types   : {result.analysis_types}")
        print(f"  Model   : {result.model_used}")
        print(f"  Tokens  : {result.tokens_used}  |  Time: {result.processing_time:.2f}s")
        print(f"  Result  : {status}")

    print("\n" + "=" * 70)
    print(f"  {passed}/{len(TEST_CASES)} tests passed")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(run_tests())
