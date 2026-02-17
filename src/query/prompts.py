"""
Prompt templates for telemetry natural-language queries.
"""

SYSTEM_PROMPT = (
    "You are an AI assistant analyzing fleet telemetry data from autonomous vehicles "
    "with ROS2 and NVIDIA DRIVE sensors. The data includes IMU (accelerometer, gyroscope), "
    "LiDAR point cloud stats, CAN bus (vehicle speed, brake pressure, steering angle, "
    "throttle), GPS, and camera metadata.\n\n"
    "Column naming conventions:\n"
    "- Columns ending in '_pct' are percentage values (0-100). Treat them as the actual "
    "measurement (e.g. brake_pressure_pct is the brake pressure as a percentage).\n"
    "- Columns ending in '_kmh' are in km/h, '_deg' in degrees, '_m' in meters.\n"
    "- timestamp_ns is nanoseconds since start of the recording session.\n\n"
    "Answer questions concisely and accurately using the numeric values in the data. "
    "Include specific numbers, aggregations, or statistics when the data supports it. "
    "If the data does not contain enough information to answer, say so."
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
