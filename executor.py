
import logging
from llm_client import call_llm_with_tools
from tools import TOOLS_SCHEMA, AVAILABLE_TOOLS

logger = logging.getLogger("agent")

EXECUTOR_SYSTEM_PROMPT = """You are the execution module of an autonomous document-writing agent.

You will be given:
- The original user request
- The overall document type and title
- The specific section you must write right now

Write clear, professional business content for ONLY this section
(3-6 sentences, or a short bulleted list if more appropriate).
Do not repeat the section heading in your answer. Do not add
markdown formatting like ** or #.

You have access to a lookup_benchmark_data tool that returns real-ish
industry benchmark figures (budget range, timeline, team size). If this
section needs concrete numbers that the user's request did not provide
(for example a budget, timeline, or team size section), call the tool
first and use its result. Do not call it for sections that don't need
numeric data (e.g. an introduction or summary). Where information is
still missing after that, make a reasonable business assumption and
state it briefly and naturally in the text.
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
            content, tools_used = call_llm_with_tools(
                EXECUTOR_SYSTEM_PROMPT, user_prompt,
                tools=TOOLS_SCHEMA, tool_functions=AVAILABLE_TOOLS, temperature=0.5,
            )
            step["content"] = content.strip()
            step["tools_used"] = tools_used
            step["status"] = "done"
        except Exception as e:
            logger.error(f"Step {step['id']} failed: {e}")
            step["content"] = (
                f"[Content generation failed for this section: {e}. "
                f"Placeholder inserted so document generation can still complete.]"
            )
            step["tools_used"] = []
            step["status"] = "failed_recovered"

        tool_note = f" (used tool: {step['tools_used']})" if step.get("tools_used") else ""
        logger.info(f"[COMPLETE] Step {step['id']} -> status={step['status']}{tool_note}")

    return plan
