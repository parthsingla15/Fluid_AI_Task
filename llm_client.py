"""
llm_client.py
--------------
Thin wrapper around the Groq API (free tier, OpenAI-compatible).

Design decision: all LLM access goes through this single function so that
retry/fallback logic and mock-mode live in exactly one place, instead of
being duplicated across planner/executor/reflect.

If GROQ_API_KEY is not set, or the API call fails for any reason (network,
rate limit, bad key), we fall back to a deterministic MOCK response so the
rest of the pipeline (planning -> execution -> reflection -> docx) can still
be developed/tested/demoed offline. This is the "Retry & fallback logic"
half of the mandatory improvement; "Reflection/self-check" (see reflect.py)
is the primary one we lean on in the writeup.
"""

import os
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("agent")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

_client = None
if GROQ_API_KEY:
    try:
        from groq import Groq
        _client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:  # pragma: no cover
        logger.warning(f"Could not init Groq client, falling back to mock mode: {e}")
        _client = None


def call_llm(system_prompt: str, user_prompt: str, max_retries: int = 2, temperature: float = 0.4) -> str:
    """
    Calls the LLM with a simple retry loop. Returns raw text content.
    Falls back to a mock response if no client is configured or all
    retries fail, so the agent degrades gracefully instead of crashing.
    """
    if _client is None:
        return _mock_response(system_prompt, user_prompt)

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = _client.chat.completions.create(
                model=GROQ_MODEL,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_err = e
            logger.warning(f"LLM call failed (attempt {attempt}/{max_retries}): {e}")
            time.sleep(0.6 * attempt)  # small backoff

    logger.error(f"All LLM attempts failed, using mock fallback. Last error: {last_err}")
    return _mock_response(system_prompt, user_prompt)


def _mock_response(system_prompt: str, user_prompt: str) -> str:
    """
    Deterministic offline stand-in so the pipeline is fully runnable/testable
    without an API key. Detects which stage is calling (plan / section /
    reflect) from the system prompt and returns plausible structured text.
    """
    sp = system_prompt.lower()

    if "return only valid json" in sp and "steps" in sp:
        return """
        {
          "doc_type": "Business Report",
          "title": "Auto-Generated Report",
          "steps": [
            {"id": 1, "section_heading": "Executive Summary", "title": "Summarize the request and objective"},
            {"id": 2, "section_heading": "Background", "title": "Explain context and current situation"},
            {"id": 3, "section_heading": "Approach", "title": "Describe the approach or plan"},
            {"id": 4, "section_heading": "Timeline & Next Steps", "title": "Outline timeline and next steps"},
            {"id": 5, "section_heading": "Risks & Assumptions", "title": "List risks and assumptions"}
          ]
        }
        """

    if "reflect" in sp or "self-check" in sp:
        return (
            "- Assumption: Specific budget figures were not provided, so placeholder "
            "estimates are used and should be validated.\n"
            "- Assumption: Timeline assumes standard business days and no major blockers.\n"
            "- Gap: Stakeholder sign-off process was not specified; a generic approval "
            "step has been added.\n"
        )

    # Default: section content generation
    return (
        "This section was generated in offline mock mode (no GROQ_API_KEY configured). "
        "In a live run, this would contain LLM-generated content addressing: "
        f"\"{user_prompt.strip()[:200]}\". Set GROQ_API_KEY to enable real generation."
    )
