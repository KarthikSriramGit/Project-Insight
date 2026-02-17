"""
Shared query configuration for notebooks 02, 03, and 04 (comparison).

Each entry: query, optional vehicle_ids, sensor_type, brake_threshold, skip_data.
Use skip_data=True for general-knowledge questions (no telemetry context).
"""

from typing import Optional

QUERY_CONFIG = [
    {
        "id": "q1",
        "label": "Peak brake % (V001)",
        "query": "What was the peak brake pressure percentage in vehicle V001?",
        "vehicle_ids": ["V001"],
        "sensor_type": "can",
    },
    {
        "id": "q2",
        "label": "Max brake % (all)",
        "query": "What is the maximum brake_pressure_pct value across all vehicles? Which vehicle had it?",
        "sensor_type": "can",
    },
    {
        "id": "q3",
        "label": "Avg speed per vehicle",
        "query": "What is the average vehicle_speed_kmh for each vehicle? List from fastest to slowest.",
        "sensor_type": "can",
    },
    {
        "id": "q4",
        "label": "Fleet health summary",
        "query": "Provide a brief fleet health summary: average speed, average throttle, average brake pressure percentage.",
        "sensor_type": "can",
    },
    {
        "id": "q5",
        "label": "Hard braking events",
        "query": "How many rows have brake_pressure_pct above 90? What is the average vehicle_speed_kmh during these events?",
        "sensor_type": "can",
        "brake_threshold": 90.0,
    },
]
