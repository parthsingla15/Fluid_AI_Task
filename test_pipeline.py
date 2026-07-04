"""
test_pipeline.py
-----------------
Quick sanity check that runs the two required test cases directly
(bypassing HTTP) so you can verify everything works before/without
starting the API server. Run: python test_pipeline.py
"""

from planner import generate_plan
from executor import execute_plan
from reflect import reflect_on_draft
from docgen import build_docx
import uuid
import json

TEST_CASES = [
    # 1. Standard, clear business request
    "Write meeting minutes for our weekly engineering standup where we discussed "
    "sprint progress, a blocked API integration, and next week's release date.",

    # 2. Complex / ambiguous request (missing info, agent must assume)
    "We need something for tomorrow's client meeting about our new product, "
    "not sure exactly what format, and the budget numbers aren't finalized yet.",
]

for i, req in enumerate(TEST_CASES, 1):
    print(f"\n{'='*70}\nTEST CASE {i}: {req}\n{'='*70}")

    plan = generate_plan(req)
    print(f"[PLAN] doc_type={plan['doc_type']} title='{plan['title']}'")
    for s in plan["steps"]:
        print(f"  - Step {s['id']}: {s['section_heading']} ({s['status']})")

    plan = execute_plan(req, plan)
    assumptions = reflect_on_draft(req, plan)
    print(f"[ASSUMPTIONS/NOTES]")
    for a in assumptions:
        print(f"  - {a}")

    filename = f"test_{i}_{uuid.uuid4().hex[:6]}.docx"
    path = build_docx(plan, assumptions, filename)
    print(f"[DOCX SAVED] {path}")
