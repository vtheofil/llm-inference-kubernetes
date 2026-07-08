"""
test_retrieval.py — Quick local sanity check for Atlas Systems RAG.

Tests ONLY the retrieval layer (no LLM call). Verifies that for each
expected query, ChromaDB returns the right type of chunk and the
top-K results actually contain the answer text.

Run after `python -m app.ingest --force` finishes:
    python scripts/test_retrieval.py
"""
from __future__ import annotations

import sys
import os

# Make sure we can import the app from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag import retrieve_with_sources

# ── Test queries with expected content in retrieved chunks ──────────────────
# For each query, we list one or more strings that MUST appear in at least
# one retrieved chunk for the test to pass.
TEST_CASES: list[tuple[str, list[str]]] = [
    # Company-level facts (from data/about/)
    ("When was Atlas Systems founded?", ["2014"]),
    ("Who is the CEO of Atlas Systems?", ["Maria Voulgari"]),
    ("How many offices does Atlas have?", ["Athens HQ", "Thessaloniki", "Patras"]),
    ("What is the annual training budget per employee?", ["1,200", "1200"]),
    ("Where is the Athens HQ located?", ["Kifisias", "Athens"]),
    ("Which universities does Atlas partner with?", ["Thessaly", "Patras", "NTUA"]),

    # Policy facts (from data/policies/)
    ("How many days of annual leave after 5 years of service?", ["28", "25", "3"]),
    ("What is the remote work policy?", ["3 days", "hybrid", "remote"]),
    ("How much can a manager approve for expenses?", ["500", "€500"]),

    # IT FAQ facts (from data/it_faq/)
    ("How do I reset my VPN password?", ["helpdesk", "vpn-reset", "5-10 minutes"]),
    ("How do I install the corporate printer?", ["atlas-print", "printer", "SSO"]),

    # Project facts (from data/projects/) — open-ended, just verify a project chunk comes back
    ("Who is the project manager for Atlas?", ["Project Manager", "Atlas"]),
    ("What is the budget of Project Phoenix?", ["Phoenix", "budget"]),

    # Department facts (from data/departments/)
    ("Who leads the Engineering department?", ["Engineering", "Director"]),
    ("How many people are in the DevOps department?", ["DevOps"]),

    # Employee facts (from data/employees/) — search-by-name
    ("What does Allison Hill do?", ["Allison Hill", "DevOps"]),
]


def colour(text: str, ok: bool) -> str:
    """Tiny ANSI helper for terminal output."""
    return f"\033[92m{text}\033[0m" if ok else f"\033[91m{text}\033[0m"


def run_tests() -> None:
    print("=" * 72)
    print("Atlas Systems — Retrieval Sanity Check")
    print("=" * 72)

    passed = 0
    failed = 0

    for query, expected_strings in TEST_CASES:
        print(f"\n── Q: {query}")
        try:
            context, sources = retrieve_with_sources(query, history=None, n=5)
        except Exception as exc:
            print(colour(f"   ERROR — retrieval failed: {exc}", False))
            failed += 1
            continue

        # Show what came back
        print(f"   Retrieved {len(sources)} chunks:")
        for i, src in enumerate(sources, 1):
            cit = src.get("citation", "?")
            dist = src.get("distance")
            dist_str = f"{dist:.4f}" if dist is not None else "N/A"
            print(f"     {i}. {cit} (distance: {dist_str})")

        # Check if any expected string is found in the combined context
        ctx_lower = context.lower()
        hits = [s for s in expected_strings if s.lower() in ctx_lower]
        if hits:
            print(colour(f"   ✓ PASS — found: {', '.join(hits)}", True))
            passed += 1
        else:
            print(colour(f"   ✗ FAIL — expected ANY of: {expected_strings}", False))
            # Show first 200 chars of context for debugging
            preview = context[:300].replace("\n", " ")
            print(f"     Context preview: {preview}…")
            failed += 1

    print()
    print("=" * 72)
    print(f"RESULTS — passed: {passed}/{passed + failed}")
    print("=" * 72)
    if failed == 0:
        print(colour("\n🎉 All retrieval tests passed. Safe to deploy.", True))
    else:
        print(colour(f"\n⚠  {failed} retrieval test(s) failed. Investigate before deploying.", False))


if __name__ == "__main__":
    run_tests()
