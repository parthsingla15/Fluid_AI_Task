"""
reflect.py
----------
THE MANDATORY "ONE REAL ENGINEERING IMPROVEMENT": Reflection / Self-check.

After the executor drafts all sections, the agent re-reads its own output
against the ORIGINAL request and asks itself:
  - What did I have to assume because the user didn't specify it?
  - What's still missing or risky?

Why this improvement, over the alternatives:
- Conversation memory / RAG don't fit a single-shot document-generation
  task with no prior turns or knowledge base.
- Multi-step planning is already covered by planner.py.
- Reflection directly targets the spec's "complex/ambiguous request"
  test case: instead of silently guessing or failing, the agent surfaces
  its assumptions explicitly, which is exactly what a competent human
  analyst would do when given an under-specified brief.

The output is appended to the document as an "Assumptions & Notes"
section and also returned in the API response, so nothing is hidden.
"""

import logging
from llm_client import call_llm

logger = logging.getLogger("agent")

REFLECTION_SYSTEM_PROMPT = """You are the reflection / self-check module of an autonomous agent.

You will see the user's original request and the full draft document the
agent produced. Compare them critically and list, as short bullet points:
- Any assumptions the agent made because information was missing or ambiguous
- Any gaps, risks, or follow-ups the user should be aware of

Be concise (3-6 bullets). If truly nothing is missing, say so in one bullet.
Do not rewrite the document. Do not use markdown headers.
"""


def reflect_on_draft(user_request: str, plan: dict) -> list[str]:
    """
    Runs one extra LLM pass over the completed draft. Returns a list of
    bullet-point strings (assumptions/gaps) to attach to the final doc.
    """
    draft_text = "\n\n".join(
        f"{s['section_heading']}: {s.get('content', '')}" for s in plan["steps"]
    )

    user_prompt = (
        f"Original user request:\n{user_request}\n\n"
        f"Draft document (type: {plan.get('doc_type')}, title: {plan.get('title')}):\n{draft_text}"
    )

    logger.info("[REFLECTING] Running self-check against original request")
    raw = call_llm(REFLECTION_SYSTEM_PROMPT, user_prompt, temperature=0.3)

    bullets = []
    for line in raw.splitlines():
        line = line.strip().lstrip("-•*").strip()
        if line:
            bullets.append(line)

    if not bullets:
        bullets = ["No significant gaps identified on self-check."]

    logger.info(f"[REFLECTION DONE] {len(bullets)} notes generated")
    return bullets
