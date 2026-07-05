
import logging

logger = logging.getLogger("agent")

# --- Tool schema, in OpenAI/Groq function-calling format ---
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "lookup_benchmark_data",
            "description": (
                "Look up mock industry benchmark data (typical budget range, "
                "timeline, and team size) for a given project or industry type. "
                "Call this when writing a section that needs concrete numbers "
                "(budget, timeline, team size, market size) but the user's "
                "request did not provide them."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": (
                            "The project type or industry to look up, e.g. "
                            "'mobile app launch', 'SaaS marketing', 'client proposal'"
                        ),
                    }
                },
                "required": ["topic"],
            },
        },
    }
]

# --- Mock backing "database" the tool reads from ---
_MOCK_BENCHMARK_DB = {
    "mobile app": {"budget_range": "$40,000 - $120,000", "timeline": "3-6 months", "team_size": "4-7 people"},
    "marketing": {"budget_range": "$8,000 - $25,000", "timeline": "4-8 weeks", "team_size": "2-4 people"},
    "saas": {"budget_range": "$60,000 - $200,000", "timeline": "4-9 months", "team_size": "5-10 people"},
    "client proposal": {"budget_range": "$5,000 - $50,000", "timeline": "1-3 months", "team_size": "2-5 people"},
    "product launch": {"budget_range": "$15,000 - $75,000", "timeline": "2-4 months", "team_size": "3-6 people"},
}

_DEFAULT_BENCHMARK = {"budget_range": "$10,000 - $50,000", "timeline": "6-12 weeks", "team_size": "3-5 people"}


def lookup_benchmark_data(topic: str) -> dict:
    """The actual tool implementation. Simple keyword match against the mock DB."""
    topic_lower = topic.lower()
    for key, data in _MOCK_BENCHMARK_DB.items():
        if key in topic_lower:
            logger.info(f"[TOOL CALL] lookup_benchmark_data('{topic}') -> matched '{key}'")
            return {"topic": topic, "source": "mock_benchmark_db", **data}

    logger.info(f"[TOOL CALL] lookup_benchmark_data('{topic}') -> no match, using default")
    return {"topic": topic, "source": "mock_benchmark_db_default", **_DEFAULT_BENCHMARK}


# Registry mapping tool name -> callable, so llm_client can dispatch generically
AVAILABLE_TOOLS = {
    "lookup_benchmark_data": lookup_benchmark_data,
}
