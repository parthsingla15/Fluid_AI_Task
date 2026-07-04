"""
planner.py
----------
Turns a free-text user request into a structured execution plan (TODO list)
that the agent decides *itself* — the request never contains hardcoded
steps. This is the "autonomous planning" requirement.

Output contract (JSON):
{
  "doc_type": "...",       # proposal / meeting minutes / project plan / etc.
  "title": "...",          # document title
  "steps": [
     {"id": 1, "section_heading": "...", "title": "..."},
     ...
  ]
}
"""

import json
import re
import logging
from llm_client import call_llm

logger = logging.getLogger("agent")

PLANNER_SYSTEM_PROMPT = """You are an autonomous planning module inside a document-generation agent.

Given a user's natural language request, decide:
1. The most appropriate business document type to produce (choose one: proposal,
   meeting minutes, project plan, business report, technical design, SOP,
   product specification, or another suitable type if none fit).
2. A short, professional document title.
3. A list of 4-7 logical sections/steps needed to fully satisfy the request.
   If the request is vague, ambiguous, or missing information, make
   reasonable, explicitly stated assumptions rather than asking a question
   (you cannot ask questions - you must decide and proceed).

Return ONLY valid JSON, no markdown fences, no commentary, in exactly this shape:
{
  "doc_type": "string",
  "title": "string",
  "steps": [
    {"id": 1, "section_heading": "string", "title": "string describing what this step should produce"}
  ]
}
"""

FALLBACK_PLAN = {
    "doc_type": "Business Report",
    "title": "Generated Report",
    "steps": [
        {"id": 1, "section_heading": "Executive Summary", "title": "Summarize the request and objective"},
        {"id": 2, "section_heading": "Details", "title": "Address the core content of the request"},
        {"id": 3, "section_heading": "Next Steps", "title": "Outline recommended next steps"},
    ],
}


def _extract_json(raw: str) -> dict:
    """Strip markdown fences / stray text and parse the first JSON object found."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned.strip(), flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned.strip()).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Last resort: grab the substring between the first { and the last }
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError("Could not parse JSON from planner output")


def generate_plan(user_request: str) -> dict:
    """
    Calls the LLM to produce a plan. On malformed output, retries once with a
    stricter instruction; if that still fails, degrades to a safe fallback
    plan instead of crashing the request (error handling & recovery).
    """
    raw = call_llm(PLANNER_SYSTEM_PROMPT, user_request, temperature=0.3)

    try:
        plan = _extract_json(raw)
        _validate_plan(plan)
        return plan
    except Exception as e:
        logger.warning(f"Plan parsing failed ({e}), retrying with stricter prompt")

    # Retry once with an even stricter reminder
    retry_prompt = user_request + "\n\nReminder: respond with ONLY the raw JSON object, nothing else."
    raw2 = call_llm(PLANNER_SYSTEM_PROMPT, retry_prompt, temperature=0.1)
    try:
        plan = _extract_json(raw2)
        _validate_plan(plan)
        return plan
    except Exception as e:
        logger.error(f"Plan parsing failed again ({e}), using fallback plan")
        return FALLBACK_PLAN


def _validate_plan(plan: dict):
    assert "steps" in plan and isinstance(plan["steps"], list) and len(plan["steps"]) > 0
    assert "doc_type" in plan and "title" in plan
    for step in plan["steps"]:
        assert "id" in step and "section_heading" in step and "title" in step
        step["status"] = "pending"  # agent tracks its own progress
