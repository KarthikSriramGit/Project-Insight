"""
Prompt templates for telemetry natural-language queries.
"""

SYSTEM_PROMPT = (
    "You are an AI assistant analyzing fleet telemetry data from autonomous vehicles "
    "with ROS2 and NVIDIA DRIVE sensors.\n\n"
    "IMPORTANT RULES:\n"
    "- All numeric columns contain real, usable values. Compute statistics directly from them.\n"
    "- '_pct' columns are percentages (0-100). Report them as percentages, e.g. 'average brake "
    "pressure: 48.5%'. Do NOT say the data is insufficient because values are percentages.\n"
    "- '_kmh' = km/h, '_deg' = degrees, '_m' = meters, '_ns' = nanoseconds.\n"
    "- gear_position is an integer (0-7). Report the most common value.\n"
    "- When asked for averages, max, min, or counts, compute them from the data provided.\n"
    "- Use the 'Summary statistics' section if provided; it has pre-computed stats.\n"
    "- Be concise. Give specific numbers. Do not hedge or refuse when the data is available."
)


def format_user_query(
    user_query: str,
    data_context: str,
    max_context_chars: int = 8000,
) -> str:
    """
    Format a user query with telemetry data context for the LLM.

    Args:
        user_query: Natural language question.
        data_context: String representation of retrieved data (e.g. CSV snippet, stats).
        max_context_chars: Truncate context if longer.

    Returns:
        Formatted user message.
    """
    if len(data_context) > max_context_chars:
        data_context = data_context[:max_context_chars] + "\n... (truncated)"
    return (
        f"Telemetry data:\n{data_context}\n\n"
        f"Question: {user_query}\n\n"
        "Provide a concise answer based on the data above."
    )
