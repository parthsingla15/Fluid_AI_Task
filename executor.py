"""
executor.py
-----------
Walks the plan produced by planner.py and executes each step in order,
generating the actual content for that section. Tracks status on each
step ("pending" -> "in_progress" -> "done") so the agent's progress is
observable (printed to console/logs as it works, and returned in the API
response) - this is what makes the planning "autonomous" rather than
just a list that's shown once and ignored.
"""

import logging
from llm_client import call_llm

logger = logging.getLogger("agent")

EXECUTOR_SYSTEM_PROMPT = """You are the execution module of an autonomous document-writing agent.

You will be given:
- The original user request
- The overall document type and title
- The specific section you must write right now

Write clear, professional business content for ONLY this section
(3-6 sentences, or a short bulleted list if more appropriate).
Do not repeat the section heading in your answer. Do not add
markdown formatting like ** or #. Where information wasn't given
by the user, make a reasonable business assumption and state it
briefly and naturally in the text.
"""


def execute_plan(user_request: str, plan: dict) -> dict:
    """
    Executes every step in plan['steps'] sequentially. Mutates each step's
    'status' field in place and attaches generated 'content'. Returns the
    same plan dict, now fully populated.
    """
    doc_type = plan.get("doc_type", "Business Report")
    title = plan.get("title", "Generated Document")

    for step in plan["steps"]:
        step["status"] = "in_progress"
        logger.info(f"[EXECUTING] Step {step['id']}: {step['section_heading']}")

        user_prompt = (
            f"Original request: {user_request}\n"
            f"Document type: {doc_type}\n"
            f"Document title: {title}\n"
            f"Section to write now: {step['section_heading']} - {step['title']}"
        )

        try:
            content = call_llm(EXECUTOR_SYSTEM_PROMPT, user_prompt, temperature=0.5)
            step["content"] = content.strip()
            step["status"] = "done"
        except Exception as e:
            logger.error(f"Step {step['id']} failed: {e}")
            step["content"] = (
                f"[Content generation failed for this section: {e}. "
                f"Placeholder inserted so document generation can still complete.]"
            )
            step["status"] = "failed_recovered"

        logger.info(f"[COMPLETE] Step {step['id']} -> status={step['status']}")

    return plan
